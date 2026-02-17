use crate::engine::kernels::{
    accum, dot_f32_simd, finite_or_zero, l2_norm, matmul_f32_embeddings, matmul_quantized,
    matmul_quantized_rows, qwen3next_linear_attention_autoregressive, rmsnorm, rmsnorm_gemma,
    rmsnorm_inplace, rmsnorm_per_head_gemma_inplace, scale_slice_inplace, select_topk_softmax,
    sigmoidf, softmax,
};
use crate::engine::profiling::{prof_end, prof_start, PROF_ATTN_NS, PROF_FFN_NS, PROF_MOE_NS};
use crate::engine::switches::{
    kv_cache_mode, layer_debug_enabled, layer_debug_pos, par_attn_min_heads,
    KvCacheMode as SwitchKvCacheMode,
};
use crate::engine::types::{Config, KvCacheFormat, RunState, TransformerWeights};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

fn alloc_f32(len: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        let bytes = len.saturating_mul(std::mem::size_of::<f32>());
        format!(
            "unable to allocate {label} ({bytes} bytes). Try reducing --context-size and --max-tokens."
        )
    })?;
    out.resize(len, 0.0);
    Ok(out)
}

fn alloc_i8(len: usize, label: &str) -> Result<Vec<i8>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        format!(
            "unable to allocate {label} ({} bytes). Try reducing --context-size and --max-tokens.",
            len
        )
    })?;
    out.resize(len, 0);
    Ok(out)
}

fn alloc_u8(len: usize, label: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        format!(
            "unable to allocate {label} ({} bytes). Try reducing --context-size and --max-tokens.",
            len
        )
    })?;
    out.resize(len, 0);
    Ok(out)
}

fn quantize_row_q8(src: &[f32], dst: &mut [i8], scale_out: &mut f32) {
    let mut max_abs = 0.0f32;
    for &x in src {
        max_abs = max_abs.max(x.abs());
    }
    if max_abs == 0.0 {
        *scale_out = 1.0;
        dst.fill(0);
        return;
    }
    let inv = 127.0 / max_abs;
    let scale = max_abs / 127.0;
    *scale_out = scale;
    for (i, &x) in src.iter().enumerate() {
        let q = (x * inv).round().clamp(-127.0, 127.0) as i8;
        dst[i] = q;
    }
}

fn quantize_row_q4(src: &[f32], dst: &mut [u8], base_elem: usize, scale_out: &mut f32) {
    let mut max_abs = 0.0f32;
    for &x in src {
        max_abs = max_abs.max(x.abs());
    }
    if max_abs == 0.0 {
        *scale_out = 1.0;
        for i in 0..src.len() {
            let elem_idx = base_elem + i;
            let byte_idx = elem_idx / 2;
            if (elem_idx & 1) == 0 {
                dst[byte_idx] &= 0xF0;
            } else {
                dst[byte_idx] &= 0x0F;
            }
        }
        return;
    }
    let inv = 7.0 / max_abs;
    let scale = max_abs / 7.0;
    *scale_out = scale;
    for (i, &x) in src.iter().enumerate() {
        let q = (x * inv).round().clamp(-8.0, 7.0) as i8;
        let nib = (q as i32 & 0x0F) as u8;
        let elem_idx = base_elem + i;
        let byte_idx = elem_idx / 2;
        if (elem_idx & 1) == 0 {
            dst[byte_idx] = (dst[byte_idx] & 0xF0) | nib;
        } else {
            dst[byte_idx] = (dst[byte_idx] & 0x0F) | (nib << 4);
        }
    }
}

#[inline]
fn dequant_q4_at(src: &[u8], elem_idx: usize) -> i8 {
    let byte = src[elem_idx / 2];
    let nib = if (elem_idx & 1) == 0 {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    };
    if nib >= 8 {
        nib as i8 - 16
    } else {
        nib as i8
    }
}

#[inline]
fn dot_q8_row(q: &[f32], cache: &[i8], row_offset: usize, scale: f32) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..q.len() {
        acc += q[i] * (cache[row_offset + i] as f32 * scale);
    }
    acc
}

#[inline]
fn axpy_q8_row(dst: &mut [f32], a: f32, cache: &[i8], row_offset: usize, scale: f32) {
    let scaled = a * scale;
    for i in 0..dst.len() {
        dst[i] += scaled * cache[row_offset + i] as f32;
    }
}

#[inline]
fn dot_q4_row(q: &[f32], cache: &[u8], row_offset: usize, scale: f32) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..q.len() {
        let v = dequant_q4_at(cache, row_offset + i) as f32 * scale;
        acc += q[i] * v;
    }
    acc
}

#[inline]
fn axpy_q4_row(dst: &mut [f32], a: f32, cache: &[u8], row_offset: usize, scale: f32) {
    let scaled = a * scale;
    for i in 0..dst.len() {
        let v = dequant_q4_at(cache, row_offset + i) as f32;
        dst[i] += scaled * v;
    }
}

pub(crate) fn malloc_run_state(p: &Config) -> Result<RunState, String> {
    let head_size = if p.head_dim > 0 {
        p.head_dim
    } else {
        p.dim / p.n_heads
    };
    let kv_dim = p.n_kv_heads * head_size;
    let q_dim = p.n_heads * head_size;
    let ssm_inner = p.ssm_inner_size;
    let ssm_k_heads = p.ssm_group_count;
    let ssm_v_heads = p.ssm_time_step_rank;
    let ssm_head_dim = p.ssm_state_size;
    let deepseek_qk_head_dim = p.deepseek_qk_nope_head_dim + p.deepseek_qk_rope_head_dim;
    let deepseek_qk_dim = p.n_heads.saturating_mul(deepseek_qk_head_dim);
    let deepseek_v_dim = p.n_heads.saturating_mul(p.deepseek_v_head_dim);
    let deepseek_q_abs_dim = p.n_heads.saturating_mul(p.deepseek_kv_lora_rank);
    let deepseek_mla_k_dim = p
        .deepseek_kv_lora_rank
        .saturating_add(p.deepseek_qk_rope_head_dim);
    let ssm_conv_dim = if p.is_qwen3next {
        ssm_inner + 2 * ssm_k_heads * ssm_head_dim
    } else {
        0
    };
    let ssm_conv_hist = if p.is_qwen3next {
        p.ssm_conv_kernel.saturating_sub(1)
    } else {
        0
    };
    let ssm_state_stride = if p.is_qwen3next {
        ssm_v_heads * ssm_head_dim * ssm_head_dim
    } else {
        0
    };
    let ssm_conv_stride = if p.is_qwen3next {
        ssm_conv_hist * ssm_conv_dim
    } else {
        0
    };
    let max_dim = p.dim.max(q_dim).max(deepseek_qk_dim);
    let ffn_dim = p
        .hidden_dim
        .max(p.expert_hidden_dim)
        .max(p.shared_expert_hidden_dim);
    let scratch_dim = ffn_dim
        .max(ssm_conv_dim)
        .max(ssm_inner)
        .max(ssm_head_dim)
        .max(deepseek_v_dim)
        .max(deepseek_q_abs_dim);

    let rope_dim = if p.rope_dim > 0 {
        p.rope_dim
    } else {
        head_size
    };
    let rope_size = rope_dim / 2;
    let mut rope_freqs = vec![0.0f32; rope_size];
    for (i, freq) in rope_freqs.iter_mut().enumerate() {
        *freq = p.rope_freq_scale / p.rope_theta.powf((i * 2) as f32 / rope_dim as f32);
    }

    let swa_theta = if p.rope_theta_swa > 0.0 {
        p.rope_theta_swa
    } else {
        10_000.0
    };
    let mut rope_freqs_swa = vec![0.0f32; rope_size];
    for (i, freq) in rope_freqs_swa.iter_mut().enumerate() {
        *freq = p.rope_freq_scale / swa_theta.powf((i * 2) as f32 / rope_dim as f32);
    }

    let att_len = p
        .n_heads
        .checked_mul(p.seq_len)
        .ok_or_else(|| "overflow while computing attention buffer size".to_string())?;
    let kv_cache_rows = p
        .n_layers
        .checked_mul(p.seq_len)
        .ok_or_else(|| "overflow while computing kv cache rows".to_string())?;
    let deepseek_k_cache_len = kv_cache_rows
        .checked_mul(deepseek_mla_k_dim)
        .ok_or_else(|| "overflow while computing deepseek key cache size".to_string())?;
    let deepseek_v_cache_len = kv_cache_rows
        .checked_mul(p.deepseek_kv_lora_rank)
        .ok_or_else(|| "overflow while computing deepseek value cache size".to_string())?;
    let kv_cache_len = kv_cache_rows
        .checked_mul(kv_dim)
        .ok_or_else(|| "overflow while computing kv cache size".to_string())?;
    let kv_cache_q4_len = kv_cache_len.div_ceil(2);

    let requested_mode = kv_cache_mode();
    let (kv_cache_format, key_cache_q8, value_cache_q8, key_cache_q4, value_cache_q4) =
        match requested_mode {
            SwitchKvCacheMode::Q8 => {
                let key = alloc_i8(kv_cache_len, "Q8 key cache")?;
                let value = alloc_i8(kv_cache_len, "Q8 value cache")?;
                (KvCacheFormat::Q8, key, value, Vec::new(), Vec::new())
            }
            SwitchKvCacheMode::Q4 => {
                let key = alloc_u8(kv_cache_q4_len, "Q4 key cache")?;
                let value = alloc_u8(kv_cache_q4_len, "Q4 value cache")?;
                (KvCacheFormat::Q4, Vec::new(), Vec::new(), key, value)
            }
            SwitchKvCacheMode::Auto => {
                let q8_try = (|| -> Result<(Vec<i8>, Vec<i8>), String> {
                    let key = alloc_i8(kv_cache_len, "Q8 key cache")?;
                    let value = alloc_i8(kv_cache_len, "Q8 value cache")?;
                    Ok((key, value))
                })();
                match q8_try {
                    Ok((key, value)) => (KvCacheFormat::Q8, key, value, Vec::new(), Vec::new()),
                    Err(q8_err) => {
                        eprintln!("KV cache Q8 allocation failed: {q8_err}");
                        eprintln!("Falling back to KV cache Q4 format.");
                        let key = alloc_u8(kv_cache_q4_len, "Q4 key cache")?;
                        let value = alloc_u8(kv_cache_q4_len, "Q4 value cache")?;
                        (KvCacheFormat::Q4, Vec::new(), Vec::new(), key, value)
                    }
                }
            }
        };

    Ok(RunState {
        x: vec![0.0; p.dim],
        xb: vec![0.0; max_dim],
        xb2: vec![0.0; p.dim],
        hb: vec![0.0; scratch_dim],
        hb2: vec![0.0; scratch_dim],
        moe_tmp: vec![0.0; p.dim],
        moe_logits: vec![0.0; p.n_experts],
        moe_topk_indices: vec![0usize; p.n_experts_used.max(1)],
        moe_topk_weights: vec![0.0f32; p.n_experts_used.max(1)],
        moe_scores: vec![0.0; p.n_experts],
        moe_selected_group: vec![true; p.moe_n_group.max(1)],
        moe_group_scores: vec![0.0; p.moe_n_group.max(1)],
        moe_group_rank: vec![0usize; p.moe_n_group.max(1)],
        q: vec![0.0; q_dim],
        k: vec![0.0; kv_dim],
        v: vec![0.0; kv_dim],
        ssm_qkv: vec![0.0; ssm_conv_dim],
        ssm_conv: vec![0.0; ssm_conv_dim],
        ssm_q: vec![0.0; ssm_inner],
        ssm_k: vec![0.0; ssm_inner],
        ssm_v: vec![0.0; ssm_inner],
        ssm_z: vec![0.0; ssm_inner],
        ssm_ba: vec![0.0; 2 * ssm_v_heads],
        ssm_gate_exp: vec![0.0; ssm_v_heads],
        ssm_beta: vec![0.0; ssm_v_heads],
        ssm_proj: vec![0.0; ssm_inner],
        ssm_kv_mem: vec![0.0; ssm_inner],
        ssm_delta: vec![0.0; ssm_inner],
        ssm_conv_state: vec![0.0; p.n_layers * ssm_conv_stride],
        ssm_state: vec![0.0; p.n_layers * ssm_state_stride],
        att: alloc_f32(att_len, "attention buffer")?,
        logits: vec![0.0; p.vocab_size],
        kv_cache_format,
        key_cache_q8,
        value_cache_q8,
        key_cache_q4,
        value_cache_q4,
        key_cache_scale: alloc_f32(kv_cache_rows, "KV key scale buffer")?,
        value_cache_scale: alloc_f32(kv_cache_rows, "KV value scale buffer")?,
        deepseek_key_cache: alloc_f32(deepseek_k_cache_len, "DeepSeek MLA key cache")?,
        deepseek_value_cache: alloc_f32(deepseek_v_cache_len, "DeepSeek MLA value cache")?,
        rope_freqs,
        rope_freqs_swa,
        rope_cos: vec![0.0; rope_size],
        rope_sin: vec![0.0; rope_size],
        rope_cache_pos: -1,
        rope_cache_is_swa: -1,
        head_size,
        kv_dim,
        q_dim,
        kv_mul: p.n_heads / p.n_kv_heads,
        attn_scale: 1.0 / (head_size as f32).sqrt(),
        embed_scale: (p.dim as f32).sqrt(),
    })
}

pub(crate) fn transformer(
    token: usize,
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
) -> Result<(), String> {
    let dim = p.dim;
    let hidden_dim = p.hidden_dim;
    let head_size = s.head_size;
    let kv_dim = s.kv_dim;
    let q_dim = s.q_dim;
    let kv_mul = s.kv_mul;
    let deepseek_qk_nope = p.deepseek_qk_nope_head_dim;
    let deepseek_qk_rope = p.deepseek_qk_rope_head_dim;
    let deepseek_qk_head = deepseek_qk_nope + deepseek_qk_rope;
    let deepseek_qk_dim = p.n_heads * deepseek_qk_head;
    let deepseek_v_head = p.deepseek_v_head_dim;
    let deepseek_v_dim = p.n_heads * deepseek_v_head;
    let deepseek_q_lora = p.deepseek_q_lora_rank;
    let deepseek_kv_lora = p.deepseek_kv_lora_rank;
    let deepseek_kv_a_rows = deepseek_kv_lora + deepseek_qk_rope;
    let deepseek_q_abs_dim = p.n_heads * deepseek_kv_lora;
    let deepseek_mla_k_dim = deepseek_kv_lora + deepseek_qk_rope;
    let eps = if p.rms_norm_eps > 0.0 {
        p.rms_norm_eps
    } else {
        1e-5
    };
    let do_layer_debug =
        layer_debug_enabled() && layer_debug_pos().map_or(pos == 0, |p0| pos == p0);

    let emb_row = &w.token_embedding_table[token * dim..(token + 1) * dim];
    s.x[..dim].copy_from_slice(emb_row);

    if p.is_gemma3 {
        scale_slice_inplace(&mut s.x[..dim], s.embed_scale);
    }

    for l in 0..p.n_layers {
        if p.is_gemma3 {
            rmsnorm_gemma(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_att_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        } else {
            rmsnorm(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_att_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        let is_qwen3next_ssm_layer = p.is_qwen3next && w.attn_qkv[l].rows > 0;
        if p.is_deepseek2 {
            let attn_prof = prof_start();
            if deepseek_qk_head == 0
                || deepseek_qk_dim == 0
                || deepseek_v_head == 0
                || deepseek_v_dim == 0
                || deepseek_q_lora == 0
                || deepseek_kv_lora == 0
                || deepseek_kv_a_rows == 0
                || deepseek_q_abs_dim == 0
                || deepseek_mla_k_dim == 0
            {
                return Err("invalid deepseek2 MLA configuration".to_string());
            }
            if w.deepseek_wk_b[l].rows != deepseek_q_abs_dim
                || w.deepseek_wk_b[l].cols != deepseek_qk_nope
            {
                return Err(format!(
                    "deepseek2 attn_k_b shape mismatch at layer {l}: got rows={}, cols={}, expected rows={}, cols={}",
                    w.deepseek_wk_b[l].rows,
                    w.deepseek_wk_b[l].cols,
                    deepseek_q_abs_dim,
                    deepseek_qk_nope
                ));
            }
            if w.deepseek_wv_b[l].rows != deepseek_v_dim
                || w.deepseek_wv_b[l].cols != deepseek_kv_lora
            {
                return Err(format!(
                    "deepseek2 attn_v_b shape mismatch at layer {l}: got rows={}, cols={}, expected rows={}, cols={}",
                    w.deepseek_wv_b[l].rows,
                    w.deepseek_wv_b[l].cols,
                    deepseek_v_dim,
                    deepseek_kv_lora
                ));
            }

            matmul_quantized(
                &mut s.hb[..deepseek_q_lora],
                &s.xb[..dim],
                &w.deepseek_wq_a[l],
                mapped,
            )?;
            rmsnorm_inplace(
                &mut s.hb[..deepseek_q_lora],
                &w.deepseek_q_a_norm[l * deepseek_q_lora..(l + 1) * deepseek_q_lora],
                deepseek_q_lora,
                eps,
            );
            matmul_quantized(
                &mut s.q[..deepseek_qk_dim],
                &s.hb[..deepseek_q_lora],
                &w.deepseek_wq_b[l],
                mapped,
            )?;

            matmul_quantized(
                &mut s.hb[..deepseek_kv_a_rows],
                &s.xb[..dim],
                &w.deepseek_wkv_a[l],
                mapped,
            )?;
            rmsnorm_inplace(
                &mut s.hb[..deepseek_kv_lora],
                &w.deepseek_kv_a_norm[l * deepseek_kv_lora..(l + 1) * deepseek_kv_lora],
                deepseek_kv_lora,
                eps,
            );

            let rope_half = deepseek_qk_rope / 2;
            if deepseek_qk_rope == 0 || rope_half == 0 {
                return Err("invalid deepseek2 rope dimension".to_string());
            }
            if s.rope_cache_pos != pos as isize || s.rope_cache_is_swa != 0 {
                for i in 0..rope_half {
                    let val = pos as f32 * s.rope_freqs[i];
                    s.rope_cos[i] = val.cos();
                    s.rope_sin[i] = val.sin();
                }
                s.rope_cache_pos = pos as isize;
                s.rope_cache_is_swa = 0;
            }

            for h in 0..p.n_heads {
                let q_base = h * deepseek_qk_head + deepseek_qk_nope;
                for i in 0..rope_half {
                    let c = s.rope_cos[i];
                    let s1 = s.rope_sin[i];
                    let q0 = s.q[q_base + i];
                    let q1 = s.q[q_base + i + rope_half];
                    s.q[q_base + i] = q0 * c - q1 * s1;
                    s.q[q_base + i + rope_half] = q0 * s1 + q1 * c;
                }
            }

            {
                let k_pe = &mut s.hb[deepseek_kv_lora..deepseek_kv_a_rows];
                for i in 0..rope_half {
                    let c = s.rope_cos[i];
                    let s1 = s.rope_sin[i];
                    let k0 = k_pe[i];
                    let k1 = k_pe[i + rope_half];
                    k_pe[i] = k0 * c - k1 * s1;
                    k_pe[i + rope_half] = k0 * s1 + k1 * c;
                }
            }

            let layer_row_base = l * p.seq_len;
            let row_index = layer_row_base + pos;
            let k_row_off = row_index * deepseek_mla_k_dim;
            let v_row_off = row_index * deepseek_kv_lora;
            {
                let k_row = &mut s.deepseek_key_cache[k_row_off..k_row_off + deepseek_mla_k_dim];
                k_row[..deepseek_kv_lora].copy_from_slice(&s.hb[..deepseek_kv_lora]);
                k_row[deepseek_kv_lora..deepseek_mla_k_dim]
                    .copy_from_slice(&s.hb[deepseek_kv_lora..deepseek_kv_a_rows]);
            }
            s.deepseek_value_cache[v_row_off..v_row_off + deepseek_kv_lora]
                .copy_from_slice(&s.hb[..deepseek_kv_lora]);

            for h in 0..p.n_heads {
                let q_nope = &s.q[h * deepseek_qk_head..h * deepseek_qk_head + deepseek_qk_nope];
                let q_abs = &mut s.hb2[h * deepseek_kv_lora..(h + 1) * deepseek_kv_lora];
                let row_start = h * deepseek_kv_lora;
                matmul_quantized_rows(
                    q_abs,
                    q_nope,
                    &w.deepseek_wk_b[l],
                    row_start,
                    deepseek_kv_lora,
                    mapped,
                )?;
            }
            s.hb[..deepseek_q_abs_dim].fill(0.0);

            let mut attn_scale_score = 1.0 / (deepseek_qk_head as f32).sqrt();
            if p.rope_scaling_yarn
                && p.rope_freq_scale > 0.0
                && p.rope_freq_scale != 1.0
                && p.rope_yarn_log_multiplier != 0.0
            {
                let factor = 1.0 / p.rope_freq_scale;
                if factor.is_finite() && factor > 0.0 {
                    let mscale = 1.0 + 0.1 * p.rope_yarn_log_multiplier * factor.ln();
                    if mscale.is_finite() && mscale > 0.0 {
                        attn_scale_score *= mscale * mscale;
                    }
                }
            }
            let att_all = &mut s.att[..p.n_heads * p.seq_len];

            for h in 0..p.n_heads {
                let att_head_full = &mut att_all[h * p.seq_len..(h + 1) * p.seq_len];
                let att_head = &mut att_head_full[..=pos];
                let q_abs = &s.hb2[h * deepseek_kv_lora..(h + 1) * deepseek_kv_lora];
                let q_pe =
                    &s.q[h * deepseek_qk_head + deepseek_qk_nope..(h + 1) * deepseek_qk_head];

                for (t, slot) in att_head.iter_mut().enumerate() {
                    let t_row = layer_row_base + t;
                    let key_off = t_row * deepseek_mla_k_dim;
                    let key_row = &s.deepseek_key_cache[key_off..key_off + deepseek_mla_k_dim];
                    let score = dot_f32_simd(q_abs, &key_row[..deepseek_kv_lora])
                        + dot_f32_simd(q_pe, &key_row[deepseek_kv_lora..deepseek_mla_k_dim]);
                    *slot = finite_or_zero(score * attn_scale_score);
                }

                softmax(att_head, pos + 1);

                let ctx_head = &mut s.hb[h * deepseek_kv_lora..(h + 1) * deepseek_kv_lora];
                for (t, &a) in att_head.iter().enumerate() {
                    let t_row = layer_row_base + t;
                    let value_off = t_row * deepseek_kv_lora;
                    let value_row =
                        &s.deepseek_value_cache[value_off..value_off + deepseek_kv_lora];
                    for i in 0..deepseek_kv_lora {
                        ctx_head[i] += a * value_row[i];
                    }
                }
            }

            for h in 0..p.n_heads {
                let ctx_head = &s.hb[h * deepseek_kv_lora..(h + 1) * deepseek_kv_lora];
                let out_head = &mut s.xb[h * deepseek_v_head..(h + 1) * deepseek_v_head];
                let row_start = h * deepseek_v_head;
                matmul_quantized_rows(
                    out_head,
                    ctx_head,
                    &w.deepseek_wv_b[l],
                    row_start,
                    deepseek_v_head,
                    mapped,
                )?;
                for v in out_head.iter_mut() {
                    *v = finite_or_zero(*v);
                }
            }

            matmul_quantized(&mut s.xb2[..dim], &s.xb[..deepseek_v_dim], &w.wo[l], mapped)?;
            for v in &mut s.xb2[..dim] {
                *v = finite_or_zero(*v);
            }
            prof_end(&PROF_ATTN_NS, attn_prof);
        } else if is_qwen3next_ssm_layer {
            qwen3next_linear_attention_autoregressive(l, p, s, w, mapped, eps)?;
        } else {
            let attn_prof = prof_start();
            let mut qwen3next_packed_q_gate = false;
            if p.is_qwen3next {
                if w.wq[l].rows >= 2 * q_dim {
                    // Qwen3Next full-attn packs Q and gate interleaved per head:
                    // [q_head0, gate_head0, q_head1, gate_head1, ...]
                    matmul_quantized_rows(
                        &mut s.hb[..2 * q_dim],
                        &s.xb[..dim],
                        &w.wq[l],
                        0,
                        2 * q_dim,
                        mapped,
                    )?;
                    if p.n_heads >= par_attn_min_heads() {
                        let hb_src = &s.hb[..2 * q_dim];
                        s.q[..q_dim].par_chunks_mut(head_size).enumerate().for_each(
                            |(h, q_dst)| {
                                let src_base = h * 2 * head_size;
                                q_dst.copy_from_slice(&hb_src[src_base..src_base + head_size]);
                            },
                        );
                    } else {
                        for h in 0..p.n_heads {
                            let src_base = h * 2 * head_size;
                            let dst_base = h * head_size;
                            s.q[dst_base..dst_base + head_size]
                                .copy_from_slice(&s.hb[src_base..src_base + head_size]);
                        }
                    }
                    qwen3next_packed_q_gate = true;
                } else if w.wq[l].rows == q_dim {
                    matmul_quantized(&mut s.q[..q_dim], &s.xb[..dim], &w.wq[l], mapped)?;
                } else {
                    matmul_quantized_rows(
                        &mut s.q[..q_dim],
                        &s.xb[..dim],
                        &w.wq[l],
                        0,
                        q_dim,
                        mapped,
                    )?;
                }
                if w.wk[l].rows == kv_dim {
                    matmul_quantized(&mut s.k[..kv_dim], &s.xb[..dim], &w.wk[l], mapped)?;
                } else {
                    matmul_quantized_rows(
                        &mut s.k[..kv_dim],
                        &s.xb[..dim],
                        &w.wk[l],
                        0,
                        kv_dim,
                        mapped,
                    )?;
                }
                if w.wv[l].rows == kv_dim {
                    matmul_quantized(&mut s.v[..kv_dim], &s.xb[..dim], &w.wv[l], mapped)?;
                } else {
                    matmul_quantized_rows(
                        &mut s.v[..kv_dim],
                        &s.xb[..dim],
                        &w.wv[l],
                        0,
                        kv_dim,
                        mapped,
                    )?;
                }
            } else {
                matmul_quantized(&mut s.q[..q_dim], &s.xb[..dim], &w.wq[l], mapped)?;
                matmul_quantized(&mut s.k[..kv_dim], &s.xb[..dim], &w.wk[l], mapped)?;
                matmul_quantized(&mut s.v[..kv_dim], &s.xb[..dim], &w.wv[l], mapped)?;
            }

            if p.is_qwen2 && !w.attn_q_bias.is_empty() {
                let qb = &w.attn_q_bias[l * q_dim..(l + 1) * q_dim];
                let kb = &w.attn_k_bias[l * kv_dim..(l + 1) * kv_dim];
                let vb = &w.attn_v_bias[l * kv_dim..(l + 1) * kv_dim];
                for i in 0..q_dim {
                    s.q[i] += qb[i];
                }
                for i in 0..kv_dim {
                    s.k[i] += kb[i];
                    s.v[i] += vb[i];
                }
            }

            if !w.attn_q_norm.is_empty()
                && !w.attn_k_norm.is_empty()
                && !w.attn_qk_norm_present.is_empty()
                && w.attn_qk_norm_present[l]
            {
                let q_norm = &w.attn_q_norm[l * head_size..(l + 1) * head_size];
                let k_norm = &w.attn_k_norm[l * head_size..(l + 1) * head_size];
                rmsnorm_per_head_gemma_inplace(
                    &mut s.q[..q_dim],
                    q_norm,
                    p.n_heads,
                    head_size,
                    eps,
                );
                rmsnorm_per_head_gemma_inplace(
                    &mut s.k[..kv_dim],
                    k_norm,
                    p.n_kv_heads,
                    head_size,
                    eps,
                );
            }

            let is_swa_layer = p.swa_pattern > 0 && (l % p.swa_pattern < p.swa_pattern - 1);
            let rope_freqs = if p.is_gemma3 && is_swa_layer {
                &s.rope_freqs_swa
            } else {
                &s.rope_freqs
            };
            let rope_half = s.rope_cos.len();
            let current_is_swa = if is_swa_layer { 1 } else { 0 };
            if s.rope_cache_pos != pos as isize || s.rope_cache_is_swa != current_is_swa {
                for i in 0..rope_half {
                    let val = pos as f32 * rope_freqs[i];
                    s.rope_cos[i] = val.cos();
                    s.rope_sin[i] = val.sin();
                }
                s.rope_cache_pos = pos as isize;
                s.rope_cache_is_swa = current_is_swa;
            }

            if p.is_gemma3 || p.is_qwen2 || p.is_qwen3moe || p.is_qwen3next {
                let pair_offset = rope_half;
                for h in 0..p.n_heads {
                    let hs = h * head_size;
                    for i in 0..rope_half {
                        let fcr = s.rope_cos[i];
                        let fci = s.rope_sin[i];
                        let v0 = s.q[hs + i];
                        let v1 = s.q[hs + i + pair_offset];
                        s.q[hs + i] = v0 * fcr - v1 * fci;
                        s.q[hs + i + pair_offset] = v0 * fci + v1 * fcr;
                    }
                }
                for h in 0..p.n_kv_heads {
                    let hs = h * head_size;
                    for i in 0..rope_half {
                        let fcr = s.rope_cos[i];
                        let fci = s.rope_sin[i];
                        let v0 = s.k[hs + i];
                        let v1 = s.k[hs + i + pair_offset];
                        s.k[hs + i] = v0 * fcr - v1 * fci;
                        s.k[hs + i + pair_offset] = v0 * fci + v1 * fcr;
                    }
                }
            } else {
                let mut i = 0;
                while i < q_dim {
                    let head_dim_idx = (i % head_size) / 2;
                    let fcr = s.rope_cos[head_dim_idx];
                    let fci = s.rope_sin[head_dim_idx];
                    let rotn = if i < kv_dim { 2 } else { 1 };
                    for v in 0..rotn {
                        let vec = if v == 0 { &mut s.q } else { &mut s.k };
                        let v0 = vec[i];
                        let v1 = vec[i + 1];
                        vec[i] = v0 * fcr - v1 * fci;
                        vec[i + 1] = v0 * fci + v1 * fcr;
                    }
                    i += 2;
                }
            }

            if p.is_gemma3 {
                scale_slice_inplace(&mut s.q[..q_dim], s.attn_scale);
            }

            let layer_row_base = l * p.seq_len;
            let row_index = layer_row_base + pos;
            let row_elem_offset = row_index * kv_dim;

            match s.kv_cache_format {
                KvCacheFormat::Q8 => {
                    quantize_row_q8(
                        &s.k[..kv_dim],
                        &mut s.key_cache_q8[row_elem_offset..row_elem_offset + kv_dim],
                        &mut s.key_cache_scale[row_index],
                    );
                    quantize_row_q8(
                        &s.v[..kv_dim],
                        &mut s.value_cache_q8[row_elem_offset..row_elem_offset + kv_dim],
                        &mut s.value_cache_scale[row_index],
                    );
                }
                KvCacheFormat::Q4 => {
                    quantize_row_q4(
                        &s.k[..kv_dim],
                        &mut s.key_cache_q4,
                        row_elem_offset,
                        &mut s.key_cache_scale[row_index],
                    );
                    quantize_row_q4(
                        &s.v[..kv_dim],
                        &mut s.value_cache_q4,
                        row_elem_offset,
                        &mut s.value_cache_scale[row_index],
                    );
                }
            }

            let attn_scale_score = s.attn_scale;
            let apply_attn_scale = !p.is_gemma3;
            let q_all = &s.q[..q_dim];
            let kv_format = s.kv_cache_format;
            let key_cache_q8 = &s.key_cache_q8;
            let value_cache_q8 = &s.value_cache_q8;
            let key_cache_q4 = &s.key_cache_q4;
            let value_cache_q4 = &s.value_cache_q4;
            let key_scales = &s.key_cache_scale;
            let value_scales = &s.value_cache_scale;
            let (att_all, xb_all) = (&mut s.att[..p.n_heads * p.seq_len], &mut s.xb[..q_dim]);

            if p.n_heads >= par_attn_min_heads() {
                att_all
                    .par_chunks_mut(p.seq_len)
                    .zip(xb_all.par_chunks_mut(head_size))
                    .enumerate()
                    .for_each(|(h, (att_head_full, xb_head))| {
                        let hs = h * head_size;
                        let q_head = &q_all[hs..hs + head_size];
                        let kv_head = h / kv_mul;
                        let kv_head_offset = kv_head * head_size;

                        let att_head = &mut att_head_full[..=pos];
                        for (t, slot) in att_head.iter_mut().enumerate() {
                            let t_row = layer_row_base + t;
                            let row_offset = t_row * kv_dim + kv_head_offset;
                            let mut score = match kv_format {
                                KvCacheFormat::Q8 => {
                                    dot_q8_row(q_head, key_cache_q8, row_offset, key_scales[t_row])
                                }
                                KvCacheFormat::Q4 => {
                                    dot_q4_row(q_head, key_cache_q4, row_offset, key_scales[t_row])
                                }
                            };
                            if apply_attn_scale {
                                score *= attn_scale_score;
                            }
                            *slot = score;
                        }

                        softmax(att_head, pos + 1);

                        xb_head.fill(0.0);
                        for (t, &a) in att_head.iter().enumerate() {
                            let t_row = layer_row_base + t;
                            let row_offset = t_row * kv_dim + kv_head_offset;
                            match kv_format {
                                KvCacheFormat::Q8 => axpy_q8_row(
                                    xb_head,
                                    a,
                                    value_cache_q8,
                                    row_offset,
                                    value_scales[t_row],
                                ),
                                KvCacheFormat::Q4 => axpy_q4_row(
                                    xb_head,
                                    a,
                                    value_cache_q4,
                                    row_offset,
                                    value_scales[t_row],
                                ),
                            }
                        }
                    });
            } else {
                for h in 0..p.n_heads {
                    let hs = h * head_size;
                    let q_head = &q_all[hs..hs + head_size];
                    let kv_head = h / kv_mul;
                    let kv_head_offset = kv_head * head_size;
                    let att_head_full = &mut att_all[h * p.seq_len..(h + 1) * p.seq_len];
                    let att_head = &mut att_head_full[..=pos];

                    for (t, slot) in att_head.iter_mut().enumerate() {
                        let t_row = layer_row_base + t;
                        let row_offset = t_row * kv_dim + kv_head_offset;
                        let mut score = match kv_format {
                            KvCacheFormat::Q8 => {
                                dot_q8_row(q_head, key_cache_q8, row_offset, key_scales[t_row])
                            }
                            KvCacheFormat::Q4 => {
                                dot_q4_row(q_head, key_cache_q4, row_offset, key_scales[t_row])
                            }
                        };
                        if apply_attn_scale {
                            score *= attn_scale_score;
                        }
                        *slot = score;
                    }

                    softmax(att_head, pos + 1);

                    let xb_head = &mut xb_all[hs..hs + head_size];
                    xb_head.fill(0.0);
                    for (t, &a) in att_head.iter().enumerate() {
                        let t_row = layer_row_base + t;
                        let row_offset = t_row * kv_dim + kv_head_offset;
                        match kv_format {
                            KvCacheFormat::Q8 => axpy_q8_row(
                                xb_head,
                                a,
                                value_cache_q8,
                                row_offset,
                                value_scales[t_row],
                            ),
                            KvCacheFormat::Q4 => axpy_q4_row(
                                xb_head,
                                a,
                                value_cache_q4,
                                row_offset,
                                value_scales[t_row],
                            ),
                        }
                    }
                }
            }

            if qwen3next_packed_q_gate {
                for h in 0..p.n_heads {
                    let src_base = h * 2 * head_size + head_size;
                    let dst_base = h * head_size;
                    for i in 0..head_size {
                        s.xb[dst_base + i] =
                            finite_or_zero(s.xb[dst_base + i] * sigmoidf(s.hb[src_base + i]));
                    }
                }
            }
            matmul_quantized(&mut s.xb2[..dim], &s.xb[..q_dim], &w.wo[l], mapped)?;
            for v in &mut s.xb2[..dim] {
                *v = finite_or_zero(*v);
            }
            prof_end(&PROF_ATTN_NS, attn_prof);
        }

        if do_layer_debug {
            eprintln!(
                "[LAYERDBG pos={pos} l={l}] post_attn_norm={:.4} x_norm={:.4}",
                l2_norm(&s.xb2[..dim]),
                l2_norm(&s.x[..dim]),
            );
        }

        if p.is_gemma3 && !w.attn_post_norm.is_empty() {
            rmsnorm_inplace(
                &mut s.xb2[..dim],
                &w.attn_post_norm[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        accum(&mut s.x[..dim], &s.xb2[..dim], dim);

        if p.is_gemma3 {
            rmsnorm_gemma(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_ffn_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        } else {
            rmsnorm(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_ffn_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        let use_moe = if p.is_deepseek2 {
            w.deepseek_layer_is_moe[l]
        } else {
            p.is_qwen3moe || p.is_qwen3next
        };
        if use_moe {
            let moe_prof = prof_start();
            let expert_hidden = p.expert_hidden_dim;
            s.xb2[..dim].copy_from_slice(&s.xb[..dim]);
            matmul_quantized(
                &mut s.moe_logits[..p.n_experts],
                &s.xb2[..dim],
                &w.moe_gate_inp[l],
                mapped,
            )?;
            let n_selected = select_topk_softmax(
                &s.moe_logits[..p.n_experts],
                p.n_experts_used,
                p.moe_n_group,
                p.moe_topk_group,
                p.moe_norm_topk_prob,
                p.moe_routed_scaling_factor,
                &mut s.moe_scores,
                &mut s.moe_selected_group,
                &mut s.moe_group_scores,
                &mut s.moe_group_rank,
                &mut s.moe_topk_indices,
                &mut s.moe_topk_weights,
            );
            s.xb[..dim].fill(0.0);

            for j in 0..n_selected {
                let expert_idx = s.moe_topk_indices[j];
                let route_weight = s.moe_topk_weights[j];
                if route_weight == 0.0 {
                    continue;
                }

                let row_start_ffn = expert_idx * expert_hidden;
                matmul_quantized_rows(
                    &mut s.hb[..expert_hidden],
                    &s.xb2[..dim],
                    &w.moe_gate_exps[l],
                    row_start_ffn,
                    expert_hidden,
                    mapped,
                )?;
                matmul_quantized_rows(
                    &mut s.hb2[..expert_hidden],
                    &s.xb2[..dim],
                    &w.moe_up_exps[l],
                    row_start_ffn,
                    expert_hidden,
                    mapped,
                )?;

                for i in 0..expert_hidden {
                    let v = s.hb[i];
                    s.hb[i] = (v * (1.0 / (1.0 + (-v).exp()))) * s.hb2[i];
                }

                let row_start_down = expert_idx * dim;
                matmul_quantized_rows(
                    &mut s.moe_tmp[..dim],
                    &s.hb[..expert_hidden],
                    &w.moe_down_exps[l],
                    row_start_down,
                    dim,
                    mapped,
                )?;
                for i in 0..dim {
                    s.xb[i] += route_weight * s.moe_tmp[i];
                }
            }

            if p.is_qwen3next && !w.moe_shared_gate_inp.is_empty() {
                let shared_hidden = if p.shared_expert_hidden_dim > 0 {
                    p.shared_expert_hidden_dim
                } else {
                    p.expert_hidden_dim
                };
                let shared_gate = &w.moe_shared_gate_inp[l * dim..(l + 1) * dim];
                let gate_logit = dot_f32_simd(&s.xb2[..dim], shared_gate);
                let gate = 1.0 / (1.0 + (-gate_logit).exp());

                matmul_quantized(&mut s.hb[..shared_hidden], &s.xb2[..dim], &w.w1[l], mapped)?;
                matmul_quantized(&mut s.hb2[..shared_hidden], &s.xb2[..dim], &w.w3[l], mapped)?;
                for i in 0..shared_hidden {
                    let v = s.hb[i];
                    s.hb[i] = (v * (1.0 / (1.0 + (-v).exp()))) * s.hb2[i];
                }
                matmul_quantized(
                    &mut s.moe_tmp[..dim],
                    &s.hb[..shared_hidden],
                    &w.w2[l],
                    mapped,
                )?;
                for i in 0..dim {
                    s.xb[i] += gate * s.moe_tmp[i];
                }
            } else if p.is_deepseek2 {
                let shared_hidden = if p.shared_expert_hidden_dim > 0 {
                    p.shared_expert_hidden_dim
                } else {
                    p.expert_hidden_dim
                };
                let gate = if w.deepseek_shared_gate_present[l] && !w.moe_shared_gate_inp.is_empty()
                {
                    let shared_gate = &w.moe_shared_gate_inp[l * dim..(l + 1) * dim];
                    let gate_logit = dot_f32_simd(&s.xb2[..dim], shared_gate);
                    1.0 / (1.0 + (-gate_logit).exp())
                } else {
                    1.0
                };

                matmul_quantized(&mut s.hb[..shared_hidden], &s.xb2[..dim], &w.w1[l], mapped)?;
                matmul_quantized(&mut s.hb2[..shared_hidden], &s.xb2[..dim], &w.w3[l], mapped)?;
                for i in 0..shared_hidden {
                    let v = s.hb[i];
                    s.hb[i] = (v * (1.0 / (1.0 + (-v).exp()))) * s.hb2[i];
                }
                matmul_quantized(
                    &mut s.moe_tmp[..dim],
                    &s.hb[..shared_hidden],
                    &w.w2[l],
                    mapped,
                )?;
                for i in 0..dim {
                    s.xb[i] += gate * s.moe_tmp[i];
                }
            }
            prof_end(&PROF_MOE_NS, moe_prof);
        } else {
            let ffn_prof = prof_start();
            matmul_quantized(&mut s.hb[..hidden_dim], &s.xb[..dim], &w.w1[l], mapped)?;
            matmul_quantized(&mut s.hb2[..hidden_dim], &s.xb[..dim], &w.w3[l], mapped)?;

            if p.is_gemma3 {
                for i in 0..hidden_dim {
                    let x = s.hb[i];
                    let gelu =
                        0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044_715 * x * x)).tanh());
                    s.hb[i] = gelu * s.hb2[i];
                }
            } else {
                for i in 0..hidden_dim {
                    let v = s.hb[i];
                    s.hb[i] = (v * (1.0 / (1.0 + (-v).exp()))) * s.hb2[i];
                }
            }

            matmul_quantized(&mut s.xb[..dim], &s.hb[..hidden_dim], &w.w2[l], mapped)?;
            prof_end(&PROF_FFN_NS, ffn_prof);
        }

        if p.is_gemma3 && !w.ffn_post_norm.is_empty() {
            rmsnorm_inplace(
                &mut s.xb[..dim],
                &w.ffn_post_norm[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        accum(&mut s.x[..dim], &s.xb[..dim], dim);

        if do_layer_debug {
            eprintln!(
                "[LAYERDBG pos={pos} l={l}] post_ffn_norm={:.4} x_norm={:.4}",
                l2_norm(&s.xb[..dim]),
                l2_norm(&s.x[..dim]),
            );
        }
    }

    if p.is_gemma3 {
        rmsnorm_inplace(&mut s.x[..dim], &w.rms_final_weight[..dim], dim, eps);
    } else {
        rmsnorm_inplace(&mut s.x[..dim], &w.rms_final_weight[..dim], dim, eps);
    }
    for v in &mut s.x[..dim] {
        *v = finite_or_zero(*v);
    }

    if w.wcls_is_embed {
        matmul_f32_embeddings(
            &mut s.logits[..p.vocab_size],
            &s.x[..dim],
            &w.token_embedding_table,
            p.vocab_size,
            dim,
        );
    } else {
        matmul_quantized(&mut s.logits[..p.vocab_size], &s.x[..dim], &w.wcls, mapped)?;
    }
    for v in &mut s.logits[..p.vocab_size] {
        *v = finite_or_zero(*v);
    }

    if p.is_gemma3 && p.final_logit_softcapping > 0.0 {
        let cap = p.final_logit_softcapping;
        for i in 0..p.vocab_size {
            s.logits[i] = cap * (s.logits[i] / cap).tanh();
        }
    }

    Ok(())
}
