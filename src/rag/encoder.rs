// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.

/// Document embedding encoder — the RAG sidecar.
///
/// Mirrors the VisionEncoder pattern: a second GGUF loaded alongside the main model.
/// Runs the full transformer prefill on a text input, then pools the hidden states to
/// produce a single embedding vector.
///
/// Pooling strategy is selected from `general.architecture` in the sidecar GGUF:
///   - `bert` / `nomic-bert` / `roberta` / `xlm-roberta`: CLS pooling (position 0)
///   - all other architectures: mean pooling over all token positions
///
/// The result is always L2-normalised so dot-product == cosine similarity at query time.
use crate::engine::io::parse_gguf_file;
use crate::engine::kernels::{
    accum, axpy_inplace, dot_f32_simd, l2_norm, layernorm_inplace,
    matmul_quantized_batch_with_scratch, silu_and_mul_inplace, softmax,
};
use crate::engine::profiling::{
    PROF_ATTN_NS, PROF_FFN_NS, PROF_TRANSFORMER_NS, prof_end, prof_start, record_forward_pass,
};
use crate::engine::runtime::{malloc_run_state, transformer};
use crate::engine::tokenizer::init_tokenizer_from_gguf;
use crate::engine::types::{Config, GGUFFile, RunState, Tokenizer, TransformerWeights};
use crate::engine::weights::init_weights_from_gguf;
use crate::vendors::build_config_from_gguf;

// ---------------------------------------------------------------------------
// Pooling strategy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PoolingStrategy {
    /// Use the hidden state at position 0 (CLS token) after the final layer.
    Cls,
    /// Average the hidden states across all token positions.
    Mean,
}

fn pooling_strategy_from_architecture(arch: &str) -> PoolingStrategy {
    let lower = arch.to_ascii_lowercase();
    if lower.contains("bert") || lower.contains("roberta") {
        PoolingStrategy::Cls
    } else {
        PoolingStrategy::Mean
    }
}

// ---------------------------------------------------------------------------
// EmbedContext — the read-only parts needed for a single embed call
// ---------------------------------------------------------------------------

/// Borrowed, read-only view of the inference-time parts of an encoder.
///
/// Contains only `Config`, `TransformerWeights`, and the mmap `&[u8]` — all of which
/// are `Send + Sync` without any `unsafe`, so this struct can be shared freely across
/// rayon worker threads.
///
/// Tokenization is intentionally excluded: `bpe_encode` requires `&mut Tokenizer`
/// (lazy hashmap init), so it is done in a sequential pre-pass before the parallel phase.
pub(crate) struct EmbedContext<'a> {
    pub(crate) config: &'a Config,
    pub(crate) weights: &'a TransformerWeights,
    pub(crate) mapped: &'a [u8],
    pub(crate) pooling: PoolingStrategy,
    pub(crate) dim: usize,
}

/// Embed a pre-tokenised chunk using the supplied context and a caller-owned `RunState`.
///
/// For BERT-family models this dispatches to `embed_prefill_bert`, which processes all
/// tokens in a single batch-GEMM forward pass (each weight row dequantised once regardless
/// of sequence length) with full bidirectional attention.  For all other architectures the
/// original sequential token-by-token path is used.
///
/// Splitting tokenisation from inference lets the parallel index-builder run the
/// transformer on many chunks concurrently (each with its own `RunState`) while
/// sharing the read-only `EmbedContext` across threads.
pub(crate) fn embed_raw(
    token_ids: &[i32],
    ctx: &EmbedContext<'_>,
    run_state: &mut RunState,
    bert_state: &mut BertPrefillState,
) -> Result<Vec<f32>, String> {
    if token_ids.is_empty() {
        return Ok(vec![0f32; ctx.dim]);
    }
    let prof_t0 = prof_start();

    // Truncate to the KV-cache capacity (seq_len was capped at load time).
    let token_ids = if token_ids.len() > ctx.config.seq_len {
        &token_ids[..ctx.config.seq_len]
    } else {
        token_ids
    };

    // BERT-family: batch prefill with bidirectional attention (faster and more correct).
    if ctx.config.is_bert_family {
        let result = (|| {
            let mut emb = embed_prefill_bert(token_ids, ctx, bert_state)?;
            l2_normalize(&mut emb);
            Ok(emb)
        })();
        prof_end(&PROF_TRANSFORMER_NS, prof_t0);
        if result.is_ok() {
            record_forward_pass();
        }
        return result;
    }

    let n = token_ids.len();
    let dim = ctx.dim;
    let mut acc = vec![0f32; dim];

    for (pos, &tok) in token_ids.iter().enumerate() {
        transformer(
            tok as usize,
            pos,
            ctx.config,
            run_state,
            ctx.weights,
            ctx.mapped,
        )?;
        match ctx.pooling {
            PoolingStrategy::Cls => {
                if pos == 0 {
                    acc.copy_from_slice(&run_state.x[..dim]);
                }
            }
            PoolingStrategy::Mean => {
                for (a, &x) in acc.iter_mut().zip(run_state.x[..dim].iter()) {
                    *a += x;
                }
            }
        }
    }

    if ctx.pooling == PoolingStrategy::Cls {
        acc = run_state.x[..dim].to_vec();
    } else {
        let scale = 1.0 / n as f32;
        acc.iter_mut().for_each(|v| *v *= scale);
    }

    l2_normalize(&mut acc);
    prof_end(&PROF_TRANSFORMER_NS, prof_t0);
    record_forward_pass();
    Ok(acc)
}

// ---------------------------------------------------------------------------
// BertPrefillState — reusable scratch buffers for embed_prefill_bert
// ---------------------------------------------------------------------------

/// Pre-allocated scratch buffers for the BERT batch-prefill forward pass.
///
/// Buffers grow on demand (first time a longer sequence is seen) but never shrink.
/// Keeping them alive across calls eliminates ~10 heap allocations per chunk — the
/// dominant overhead when indexing thousands of small documents.
pub(crate) struct BertPrefillState {
    pub(crate) all_cos: Vec<f32>,
    pub(crate) all_sin: Vec<f32>,
    pub(crate) rope_inv_freq: Vec<f32>,
    pub(crate) rope_cached_tokens: usize,
    pub(crate) rope_cached_half: usize,
    pub(crate) rope_cached_dim: usize,
    pub(crate) rope_cached_theta_bits: u32,
    pub(crate) x: Vec<f32>,
    pub(crate) qkv_buf: Vec<f32>,
    pub(crate) q_buf: Vec<f32>,
    pub(crate) k_buf: Vec<f32>,
    pub(crate) v_buf: Vec<f32>,
    pub(crate) k_head_buf: Vec<f32>,
    pub(crate) v_head_buf: Vec<f32>,
    pub(crate) xb_attn: Vec<f32>,
    pub(crate) xb2: Vec<f32>,
    pub(crate) hb: Vec<f32>,
    pub(crate) hb2: Vec<f32>,
    pub(crate) att: Vec<f32>,
    pub(crate) dequant_row: Vec<f32>,
}

impl BertPrefillState {
    pub(crate) fn new() -> Self {
        Self {
            all_cos: Vec::new(),
            all_sin: Vec::new(),
            rope_inv_freq: Vec::new(),
            rope_cached_tokens: 0,
            rope_cached_half: 0,
            rope_cached_dim: 0,
            rope_cached_theta_bits: 0,
            x: Vec::new(),
            qkv_buf: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            k_head_buf: Vec::new(),
            v_head_buf: Vec::new(),
            xb_attn: Vec::new(),
            xb2: Vec::new(),
            hb: Vec::new(),
            hb2: Vec::new(),
            att: Vec::new(),
            dequant_row: Vec::new(),
        }
    }

    /// Ensure all buffers can hold at least `m` tokens.  Only reallocates when `m`
    /// exceeds the previous maximum — O(1) on the steady-state hot path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ensure(
        &mut self,
        m: usize,
        rope_half: usize,
        dim: usize,
        total_qkv: usize,
        q_dim: usize,
        kv_dim: usize,
        hidden_dim: usize,
    ) {
        grow(&mut self.all_cos, m * rope_half);
        grow(&mut self.all_sin, m * rope_half);
        grow(&mut self.x, m * dim);
        grow(&mut self.qkv_buf, m * total_qkv);
        grow(&mut self.q_buf, m * q_dim);
        grow(&mut self.k_buf, m * kv_dim);
        grow(&mut self.v_buf, m * kv_dim);
        grow(&mut self.k_head_buf, m * kv_dim);
        grow(&mut self.v_head_buf, m * kv_dim);
        grow(&mut self.xb_attn, m * q_dim);
        grow(&mut self.xb2, m * dim);
        grow(&mut self.hb, m * hidden_dim);
        grow(&mut self.hb2, m * hidden_dim);
        grow(&mut self.att, m);
    }

    fn ensure_rope_tables(&mut self, m: usize, rope_half: usize, rope_dim: usize, rope_theta: f32) {
        if rope_half == 0 {
            return;
        }

        let rope_theta_bits = rope_theta.to_bits();
        let rope_params_changed = self.rope_cached_half != rope_half
            || self.rope_cached_dim != rope_dim
            || self.rope_cached_theta_bits != rope_theta_bits;
        if rope_params_changed {
            if self.rope_inv_freq.len() < rope_half {
                self.rope_inv_freq.resize(rope_half, 0.0);
            }
            for i in 0..rope_half {
                self.rope_inv_freq[i] = 1.0_f32 / rope_theta.powf((2 * i) as f32 / rope_dim as f32);
            }
            self.rope_cached_tokens = 0;
            self.rope_cached_half = rope_half;
            self.rope_cached_dim = rope_dim;
            self.rope_cached_theta_bits = rope_theta_bits;
        }

        if self.rope_cached_tokens >= m {
            return;
        }

        for pos in self.rope_cached_tokens..m {
            let base = pos * rope_half;
            for i in 0..rope_half {
                let val = pos as f32 * self.rope_inv_freq[i];
                self.all_cos[base + i] = val.cos();
                self.all_sin[base + i] = val.sin();
            }
        }
        self.rope_cached_tokens = m;
    }
}

#[inline(always)]
fn grow(buf: &mut Vec<f32>, needed: usize) {
    if buf.len() < needed {
        buf.resize(needed, 0.0);
    }
}

// ---------------------------------------------------------------------------
// BERT batch-prefill forward pass
// ---------------------------------------------------------------------------

/// Full batch forward pass for BERT-family encoder models.
///
/// Processes all `m = token_ids.len()` tokens simultaneously through each transformer layer,
/// using `matmul_quantized_batch` (GEMM) so each weight row is dequantised exactly once per
/// layer regardless of sequence length.  Attention is fully bidirectional (no causal mask),
/// which is both faster and more correct than the sequential causal fallback.
///
/// Returns the pooled hidden-state vector (not yet L2-normalised).
fn embed_prefill_bert(
    token_ids: &[i32],
    ctx: &EmbedContext<'_>,
    state: &mut BertPrefillState,
) -> Result<Vec<f32>, String> {
    let p = ctx.config;
    let w = ctx.weights;
    let mapped = ctx.mapped;
    let m = token_ids.len();
    let dim = p.dim;
    let head_size = if p.head_dim > 0 {
        p.head_dim
    } else {
        dim / p.n_heads
    };
    let kv_dim = p.n_kv_heads * head_size;
    let q_dim = p.n_heads * head_size;
    let total_qkv = q_dim + 2 * kv_dim;
    let hidden_dim = p.hidden_dim;
    let kv_mul = p.n_heads / p.n_kv_heads;
    let attn_scale = 1.0_f32 / (head_size as f32).sqrt();
    let eps = if p.rms_norm_eps > 0.0 {
        p.rms_norm_eps
    } else {
        1e-5
    };

    // RoPE frequencies (adjacent-pair, same formula as the non-Gemma/Qwen branch in inference.rs).
    let rope_dim = if p.rope_dim > 0 {
        p.rope_dim
    } else {
        head_size
    };
    let rope_half = rope_dim / 2;

    // Ensure all scratch buffers are large enough for m tokens.
    state.ensure(m, rope_half, dim, total_qkv, q_dim, kv_dim, hidden_dim);
    state.ensure_rope_tables(m, rope_half, rope_dim, p.rope_theta);

    // Bind named slice views — these are the exact same names as before, now pointing
    // into the pre-allocated buffers instead of freshly heap-allocated Vecs.
    let all_cos = &mut state.all_cos[..m * rope_half];
    let all_sin = &mut state.all_sin[..m * rope_half];
    let x = &mut state.x[..m * dim];
    let qkv_buf = &mut state.qkv_buf[..m * total_qkv];
    let q_buf = &mut state.q_buf[..m * q_dim];
    let k_buf = &mut state.k_buf[..m * kv_dim];
    let v_buf = &mut state.v_buf[..m * kv_dim];
    let k_head_buf = &mut state.k_head_buf[..m * kv_dim];
    let v_head_buf = &mut state.v_head_buf[..m * kv_dim];
    let xb_attn = &mut state.xb_attn[..m * q_dim];
    let xb2 = &mut state.xb2[..m * dim];
    let hb = &mut state.hb[..m * hidden_dim];
    let hb2 = &mut state.hb2[..m * hidden_dim];
    let att = &mut state.att[..m];
    let dequant_row = &mut state.dequant_row;

    // x[m × dim]: token embeddings
    for (pos, &tok) in token_ids.iter().enumerate() {
        let tok = tok as usize;
        let src = &w.token_embedding_table[tok * dim..(tok + 1) * dim];
        x[pos * dim..(pos + 1) * dim].copy_from_slice(src);
    }

    for l in 0..p.n_layers {
        let attn_prof = prof_start();
        // ── QKV projection ─────────────────────────────────────────────────
        // Pre-attn for BERT post-norm: feed x directly (no norm).
        let fused_qkv = !w.wq.is_empty() && w.wq[l].rows == q_dim + 2 * kv_dim;
        if fused_qkv {
            matmul_quantized_batch_with_scratch(
                qkv_buf,
                x,
                &w.wq[l],
                mapped,
                m,
                0,
                total_qkv,
                dequant_row,
            )?;
        } else {
            matmul_quantized_batch_with_scratch(
                q_buf,
                x,
                &w.wq[l],
                mapped,
                m,
                0,
                q_dim,
                dequant_row,
            )?;
            matmul_quantized_batch_with_scratch(
                k_buf,
                x,
                &w.wk[l],
                mapped,
                m,
                0,
                kv_dim,
                dequant_row,
            )?;
            matmul_quantized_batch_with_scratch(
                v_buf,
                x,
                &w.wv[l],
                mapped,
                m,
                0,
                kv_dim,
                dequant_row,
            )?;
        }

        // ── RoPE: adjacent-pair rotation per position ──────────────────────
        if fused_qkv {
            for pos in 0..m {
                let rope_cos = &all_cos[pos * rope_half..(pos + 1) * rope_half];
                let rope_sin = &all_sin[pos * rope_half..(pos + 1) * rope_half];
                let fused_base = pos * total_qkv;
                let q_base = pos * q_dim;
                let q_src = &qkv_buf[fused_base..fused_base + q_dim];
                q_buf[q_base..q_base + q_dim].copy_from_slice(q_src);

                let mut qi = 0;
                while qi < q_dim {
                    let hd = ((qi % head_size) / 2).min(rope_half.saturating_sub(1));
                    let fcr = rope_cos[hd];
                    let fci = rope_sin[hd];
                    let v0 = q_buf[q_base + qi];
                    let v1 = q_buf[q_base + qi + 1];
                    q_buf[q_base + qi] = v0 * fcr - v1 * fci;
                    q_buf[q_base + qi + 1] = v0 * fci + v1 * fcr;
                    qi += 2;
                }

                let k_src_base = fused_base + q_dim;
                let v_src_base = k_src_base + kv_dim;
                for kv_head in 0..p.n_kv_heads {
                    let head_base = kv_head * m * head_size + pos * head_size;
                    let k_src = &qkv_buf
                        [k_src_base + kv_head * head_size..k_src_base + (kv_head + 1) * head_size];
                    let v_src = &qkv_buf
                        [v_src_base + kv_head * head_size..v_src_base + (kv_head + 1) * head_size];
                    k_head_buf[head_base..head_base + head_size].copy_from_slice(k_src);
                    v_head_buf[head_base..head_base + head_size].copy_from_slice(v_src);

                    let mut ki = 0;
                    while ki < head_size {
                        let hd = (ki / 2).min(rope_half.saturating_sub(1));
                        let fcr = rope_cos[hd];
                        let fci = rope_sin[hd];
                        let v0 = k_head_buf[head_base + ki];
                        let v1 = k_head_buf[head_base + ki + 1];
                        k_head_buf[head_base + ki] = v0 * fcr - v1 * fci;
                        k_head_buf[head_base + ki + 1] = v0 * fci + v1 * fcr;
                        ki += 2;
                    }
                }
            }
        } else {
            for pos in 0..m {
                let rope_cos = &all_cos[pos * rope_half..(pos + 1) * rope_half];
                let rope_sin = &all_sin[pos * rope_half..(pos + 1) * rope_half];
                let q_base = pos * q_dim;
                let k_base = pos * kv_dim;
                let mut qi = 0;
                while qi < q_dim {
                    let hd = ((qi % head_size) / 2).min(rope_half.saturating_sub(1));
                    let fcr = rope_cos[hd];
                    let fci = rope_sin[hd];
                    let v0 = q_buf[q_base + qi];
                    let v1 = q_buf[q_base + qi + 1];
                    q_buf[q_base + qi] = v0 * fcr - v1 * fci;
                    q_buf[q_base + qi + 1] = v0 * fci + v1 * fcr;
                    qi += 2;
                }
                let mut ki = 0;
                while ki < kv_dim {
                    let hd = ((ki % head_size) / 2).min(rope_half.saturating_sub(1));
                    let fcr = rope_cos[hd];
                    let fci = rope_sin[hd];
                    let v0 = k_buf[k_base + ki];
                    let v1 = k_buf[k_base + ki + 1];
                    k_buf[k_base + ki] = v0 * fcr - v1 * fci;
                    k_buf[k_base + ki + 1] = v0 * fci + v1 * fcr;
                    ki += 2;
                }
            }

            // Stage K/V in head-major order so the attention inner loops can stream
            // contiguous position slices instead of hopping by kv_dim every step.
            for kv_head in 0..p.n_kv_heads {
                let head_base = kv_head * m * head_size;
                for pos in 0..m {
                    let src_base = pos * kv_dim + kv_head * head_size;
                    let dst_base = head_base + pos * head_size;
                    k_head_buf[dst_base..dst_base + head_size]
                        .copy_from_slice(&k_buf[src_base..src_base + head_size]);
                    v_head_buf[dst_base..dst_base + head_size]
                        .copy_from_slice(&v_buf[src_base..src_base + head_size]);
                }
            }
        }

        // ── Bidirectional attention ─────────────────────────────────────────
        xb_attn.fill(0.0);
        for h in 0..p.n_heads {
            let kv_head = h / kv_mul;
            let kv_head_base = kv_head * m * head_size;

            // Scores are materialized one row at a time and consumed immediately,
            // avoiding the large n_heads × m × m scratch tensor.
            for pos in 0..m {
                let q_start = pos * q_dim + h * head_size;
                let q_head = &q_buf[q_start..q_start + head_size];
                let scores = &mut att[..m];
                for (t, score) in scores.iter_mut().enumerate() {
                    let k_start = kv_head_base + t * head_size;
                    *score = dot_f32_simd(q_head, &k_head_buf[k_start..k_start + head_size])
                        * attn_scale;
                }
                softmax(scores, m);
                let out_start = pos * q_dim + h * head_size;
                let xb_head = &mut xb_attn[out_start..out_start + head_size];
                xb_head.fill(0.0);
                for (t, &a) in scores.iter().enumerate() {
                    let v_start = kv_head_base + t * head_size;
                    axpy_inplace(xb_head, a, &v_head_buf[v_start..v_start + head_size]);
                }
            }
        }

        // ── Output projection ───────────────────────────────────────────────
        // wo[l].rows = dim, wo[l].cols = q_dim
        matmul_quantized_batch_with_scratch(
            xb2,
            xb_attn,
            &w.wo[l],
            mapped,
            m,
            0,
            dim,
            dequant_row,
        )?;

        // Post-attn: residual + LayerNorm per token.
        for pos in 0..m {
            let off = pos * dim;
            accum(&mut x[off..off + dim], &xb2[off..off + dim], dim);
            layernorm_inplace(
                &mut x[off..off + dim],
                &w.attn_post_norm[l * dim..(l + 1) * dim],
                &w.attn_post_norm_bias[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }
        prof_end(&PROF_ATTN_NS, attn_prof);

        // ── FFN (SwiGLU / gated) ────────────────────────────────────────────
        // Pre-FFN for BERT: feed x directly (no norm).
        // w1.rows = hidden_dim, w1.cols = dim  (gate)
        // w3.rows = hidden_dim, w3.cols = dim  (up)
        // w2.rows = dim,        w2.cols = hidden_dim  (down)
        let ffn_prof = prof_start();
        matmul_quantized_batch_with_scratch(
            hb,
            x,
            &w.w1[l],
            mapped,
            m,
            0,
            hidden_dim,
            dequant_row,
        )?;
        matmul_quantized_batch_with_scratch(
            hb2,
            x,
            &w.w3[l],
            mapped,
            m,
            0,
            hidden_dim,
            dequant_row,
        )?;
        silu_and_mul_inplace(&mut hb[..m * hidden_dim], &hb2[..m * hidden_dim]);
        matmul_quantized_batch_with_scratch(xb2, hb, &w.w2[l], mapped, m, 0, dim, dequant_row)?;

        // Post-FFN: residual + LayerNorm per token.
        for pos in 0..m {
            let off = pos * dim;
            accum(&mut x[off..off + dim], &xb2[off..off + dim], dim);
            layernorm_inplace(
                &mut x[off..off + dim],
                &w.ffn_post_norm[l * dim..(l + 1) * dim],
                &w.ffn_post_norm_bias[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }
        prof_end(&PROF_FFN_NS, ffn_prof);
    }

    // Pool hidden states.
    let embedding = match ctx.pooling {
        PoolingStrategy::Cls => x[0..dim].to_vec(),
        PoolingStrategy::Mean => {
            let mut acc = vec![0f32; dim];
            for pos in 0..m {
                let off = pos * dim;
                for (a, &xi) in acc.iter_mut().zip(x[off..off + dim].iter()) {
                    *a += xi;
                }
            }
            let scale = 1.0 / m as f32;
            acc.iter_mut().for_each(|v| *v *= scale);
            acc
        }
    };
    Ok(embedding)
}

// ---------------------------------------------------------------------------
// EmbeddingEncoder — one concrete sidecar model
// ---------------------------------------------------------------------------

pub(crate) struct EmbeddingEncoder {
    gguf: GGUFFile,
    config: Config,
    tokenizer: Tokenizer,
    weights: TransformerWeights,
    run_state: RunState,
    bert_state: BertPrefillState,
    pooling: PoolingStrategy,
    /// Prefix prepended to query strings at retrieval time (e.g. `"search_query: "` for nomic-embed).
    pub(crate) query_prefix: String,
    /// Prefix prepended to document text at index-build time (e.g. `"search_document: "` for nomic-embed).
    pub(crate) document_prefix: String,
}

/// Return task-specific prefixes for architectures that require them.
/// nomic-embed-text-v1.5 uses Matryoshka/instruction-tuned embeddings and requires
/// "search_query:" / "search_document:" to be prepended for retrieval tasks.
fn task_prefixes_for_architecture(arch: &str) -> (&'static str, &'static str) {
    let lower = arch.to_ascii_lowercase();
    if lower == "nomic-bert" {
        ("search_query: ", "search_document: ")
    } else {
        ("", "")
    }
}

impl EmbeddingEncoder {
    pub(crate) fn new(gguf: GGUFFile, debug_mode: bool) -> Result<Self, String> {
        let mut config = build_config_from_gguf(&gguf, debug_mode)?;
        let tokenizer_policy = crate::vendors::tokenizer_policy(&config);
        let mut tokenizer =
            init_tokenizer_from_gguf(&gguf, &mut config, tokenizer_policy, debug_mode)?;
        tokenizer.use_sentencepiece = config.is_gemma3;
        let weights = init_weights_from_gguf(&gguf, &config, debug_mode)?;

        // Embedding never generates — it only prefills up to the chunk length.
        // Cap seq_len so the KV cache is proportional to the actual input, not the
        // model's full context window.  A 32k-context decoder model would otherwise
        // allocate ~4 GB of KV cache *per RunState*, which explodes when we create
        // one RunState per rayon thread.
        // 2048 tokens covers any chunk we will ever embed (max_chars=1800 ≈ 450 tok).
        const MAX_EMBED_SEQ_LEN: usize = 2048;
        if config.seq_len > MAX_EMBED_SEQ_LEN {
            if debug_mode {
                eprintln!(
                    "RAG encoder: capping seq_len {} → {MAX_EMBED_SEQ_LEN} (embedding only)",
                    config.seq_len
                );
            }
            config.seq_len = MAX_EMBED_SEQ_LEN;
        }

        let run_state = malloc_run_state(&config)?;
        let bert_state = BertPrefillState::new();
        let arch = crate::engine::io::get_gguf_string_from_map(&gguf.kv, "general.architecture")
            .unwrap_or_default();
        let pooling = pooling_strategy_from_architecture(arch);
        let (query_prefix, document_prefix) = task_prefixes_for_architecture(arch);
        let (query_prefix, document_prefix) =
            (query_prefix.to_string(), document_prefix.to_string());

        if debug_mode {
            let param_estimate = estimate_params(&config);
            let speed_hint = if config.n_layers > 16 || config.dim > 1024 {
                " ⚠ large model — consider a BERT-style encoder for faster indexing"
            } else {
                ""
            };
            eprintln!(
                "RAG encoder: arch={arch}  layers={}  dim={}  ~{param_estimate}  pooling={:?}{speed_hint}",
                config.n_layers, config.dim, pooling
            );
            eprintln!("  seq_len capped to {}", config.seq_len);
        }
        Ok(Self {
            gguf,
            config,
            tokenizer,
            weights,
            run_state,
            bert_state,
            pooling,
            query_prefix,
            document_prefix,
        })
    }

    /// Embedding dimension (= model `dim`).
    pub(crate) fn dim(&self) -> usize {
        self.config.dim
    }

    pub(crate) fn prepare_tokenizer(&mut self) {
        self.tokenizer.prepare_for_encode();
    }

    pub(crate) fn prepared_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Borrow the inference-time (read-only) parts as an `EmbedContext`.
    ///
    /// Call this after all tokenisation is done.  The returned context is `Send + Sync`
    /// and can be shared freely across rayon worker threads.
    pub(crate) fn embed_context(&self) -> EmbedContext<'_> {
        EmbedContext {
            config: &self.config,
            weights: &self.weights,
            mapped: self.gguf.mapped.as_slice(),
            pooling: self.pooling,
            dim: self.config.dim,
        }
    }

    /// Embed `text` → unit-length f32 vector (single-threaded convenience wrapper).
    pub(crate) fn embed(&mut self, text: &str) -> Result<Vec<f32>, String> {
        let mut ids = Vec::new();
        self.tokenizer.bpe_encode(text, &mut ids);
        let config = &self.config;
        let weights = &self.weights;
        let mapped = self.gguf.mapped.as_slice();
        let pooling = self.pooling;
        let dim = self.config.dim;
        let ctx = EmbedContext {
            config,
            weights,
            mapped,
            pooling,
            dim,
        };
        embed_raw(&ids, &ctx, &mut self.run_state, &mut self.bert_state)
    }
}

/// Rough parameter count for display purposes (vocab embeddings + transformer layers).
fn estimate_params(c: &Config) -> String {
    let embed = c.vocab_size * c.dim;
    let attn = c.n_layers * (4 * c.dim * c.dim); // Q K V O projections
    let ffn = c.n_layers * (3 * c.dim * c.hidden_dim.max(1)); // gate+up+down
    let total = (embed + attn + ffn) as f64;
    if total >= 1e9 {
        format!("{:.1}B params", total / 1e9)
    } else {
        format!("{:.0}M params", total / 1e6)
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 1e-8 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

// ---------------------------------------------------------------------------
// DocumentEncoder dispatch enum — mirrors VisionEncoder
// ---------------------------------------------------------------------------

pub(crate) enum DocumentEncoder {
    Embedding(EmbeddingEncoder),
}

impl DocumentEncoder {
    /// Load an embedding sidecar GGUF from `path`.
    pub(crate) fn load(path: &str, debug_mode: bool) -> Result<Self, String> {
        if debug_mode {
            eprintln!("Loading RAG encoder sidecar: {path}");
        }
        let gguf = parse_gguf_file(path, debug_mode)
            .map_err(|e| format!("failed to load RAG encoder '{}': {e}", path))?;
        let enc = EmbeddingEncoder::new(gguf, debug_mode)?;
        Ok(Self::Embedding(enc))
    }

    pub(crate) fn load_from_bytes(data: &'static [u8]) -> Result<Self, String> {
        use crate::engine::io::parse_gguf_from_bytes;
        let gguf = parse_gguf_from_bytes(data, false)
            .map_err(|e| format!("failed to load embedded RAG encoder: {e}"))?;
        let enc = EmbeddingEncoder::new(gguf, false)?;
        Ok(Self::Embedding(enc))
    }

    /// Embedding dimension.
    pub(crate) fn dim(&self) -> usize {
        match self {
            Self::Embedding(e) => e.dim(),
        }
    }

    pub(crate) fn prepare_tokenizer(&mut self) {
        match self {
            Self::Embedding(e) => e.prepare_tokenizer(),
        }
    }

    pub(crate) fn prepared_tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Embedding(e) => e.prepared_tokenizer(),
        }
    }

    /// Borrow the inference-time read-only state as an `EmbedContext`.
    /// Call only after all tokenisation is complete.
    pub(crate) fn embed_context(&self) -> EmbedContext<'_> {
        match self {
            Self::Embedding(e) => e.embed_context(),
        }
    }

    /// Embed `text` → L2-normalised vector (single-threaded convenience).
    pub(crate) fn embed(&mut self, text: &str) -> Result<Vec<f32>, String> {
        match self {
            Self::Embedding(e) => e.embed(text),
        }
    }

    /// Prefix to prepend to query strings (empty string if none required).
    pub(crate) fn query_prefix(&self) -> &str {
        match self {
            Self::Embedding(e) => &e.query_prefix,
        }
    }

    /// Prefix to prepend to document text when indexing (empty string if none required).
    pub(crate) fn document_prefix(&self) -> &str {
        match self {
            Self::Embedding(e) => &e.document_prefix,
        }
    }
}

// ---------------------------------------------------------------------------
// Sidecar auto-discovery
// ---------------------------------------------------------------------------

/// Try to find an embedding sidecar GGUF next to `model_path`.
///
/// Probe order (first match wins):
///   1. `embed-{filename}`        e.g. `embed-Qwen3-4B-Q4.gguf`
///   2. `{stem}.embed.gguf`
///   3. `encoder.gguf`
///   4. Any `embed*.gguf` or `encoder*.gguf` in the same directory, alphabetically first.
pub(crate) fn discover_embedding_sidecar(model_path: &str) -> Option<String> {
    let model = std::path::Path::new(model_path);
    let dir = model.parent()?;
    let filename = model.file_name()?.to_str()?;
    let stem = model.file_stem()?.to_str()?;

    let candidates = [
        dir.join(format!("embed-{filename}")),
        dir.join(format!("{stem}.embed.gguf")),
        dir.join("encoder.gguf"),
    ];

    for c in &candidates {
        if c.exists() {
            return Some(c.to_string_lossy().into_owned());
        }
    }

    // Glob: any embed*.gguf or encoder*.gguf in dir.
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut matches: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let lower = name.to_ascii_lowercase();
                if (lower.starts_with("embed") || lower.starts_with("encoder"))
                    && lower.ends_with(".gguf")
                    && e.path() != model
                {
                    Some(e.path().to_string_lossy().into_owned())
                } else {
                    None
                }
            })
            .collect();
        matches.sort();
        if let Some(first) = matches.into_iter().next() {
            return Some(first);
        }
    }

    None
}
