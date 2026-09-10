//! Correctness-oriented complete image-block evaluation. No batched matmul is
//! needed: staging all K/V rows at each layer supplies future keys to its queries.

use super::inference::{
    attention_token, finish_logits, layer_after_attention, layer_attention_token_staged,
    layer_input_norm,
};
use crate::engine::kernels::{
    MatmulActivationScratch, matmul_quantized_with_scratch, sanitize_finite_inplace,
};
use crate::engine::profiling::{PROF_ATTN_NS, prof_end, prof_start};
use crate::engine::types::{Config, ImageAttentionMode, RunState, TransformerWeights};

#[allow(clippy::too_many_arguments)]
pub(crate) fn transformer_prefill_image_block(
    embeddings: &[&[f32]],
    base_pos: usize,
    compute_logits: bool,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
) -> Result<(), String> {
    let end = base_pos
        .checked_add(embeddings.len())
        .ok_or_else(|| "image-block prefill range overflow".to_string())?;
    let block = s
        .media_attention_plan
        .block_at(base_pos)
        .ok_or_else(|| "image-block prefill has no matching attention block".to_string())?;
    if p.attention_policy.image_mode != ImageAttentionMode::Bidirectional
        || block.token_start != base_pos
        || block.token_len != embeddings.len()
        || end > p.seq_len
    {
        return Err(
            "image-block prefill must evaluate one complete supported block within context"
                .to_string(),
        );
    }
    if embeddings
        .iter()
        .any(|row| row.len() != p.dim || row.iter().any(|value| !value.is_finite()))
    {
        return Err(
            "image-block prefill requires finite, language-dimension embeddings".to_string(),
        );
    }
    // The staged contract currently supports dense, separate Q/K/V projections.
    // Reject incompatible stateful or packed layouts instead of a causal fallback.
    if p.n_deepstack_layers != 0
        || p.n_experts != 0
        || (0..p.n_layers).any(|layer| {
            w.wq.get(layer)
                .is_none_or(|tensor| tensor.rows != s.q_dim || tensor.cols != p.dim)
                || w.wk
                    .get(layer)
                    .is_none_or(|tensor| tensor.rows != s.kv_dim || tensor.cols != p.dim)
                || w.wv
                    .get(layer)
                    .is_none_or(|tensor| tensor.rows != s.kv_dim || tensor.cols != p.dim)
                || w.attn_qkv.get(layer).is_some_and(|tensor| tensor.rows != 0)
        })
    {
        return Err("image-block prefill requires a dense separate-QKV layout".to_string());
    }
    fn scratch(rows: usize, width: usize) -> Result<Vec<f32>, String> {
        let len = rows
            .checked_mul(width)
            .ok_or_else(|| "image-block scratch size overflow".to_string())?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| "unable to allocate image-block scratch".to_string())?;
        values.resize(len, 0.0);
        Ok(values)
    }
    let mut residuals = scratch(embeddings.len(), p.dim)?;
    let mut queries = scratch(embeddings.len(), s.q_dim)?;
    for (target, embedding) in residuals.chunks_exact_mut(p.dim).zip(embeddings) {
        target.copy_from_slice(embedding);
    }
    let mut matmul = MatmulActivationScratch::new();
    for layer in 0..p.n_layers {
        // All inputs are at the same layer. No query can read another token's
        // K/V from the wrong layer or from an unprocessed future view.
        for (index, row) in residuals.chunks_exact(p.dim).enumerate() {
            s.x[..p.dim].copy_from_slice(row);
            layer_input_norm(p, s, w, layer);
            layer_attention_token_staged(
                p,
                s,
                w,
                mapped,
                &mut matmul,
                layer,
                base_pos + index,
                true,
            )?;
            queries[index * s.q_dim..(index + 1) * s.q_dim].copy_from_slice(&s.q[..s.q_dim]);
        }
        for (index, row) in residuals.chunks_exact_mut(p.dim).enumerate() {
            s.x[..p.dim].copy_from_slice(row);
            s.q[..s.q_dim].copy_from_slice(&queries[index * s.q_dim..(index + 1) * s.q_dim]);
            let attention_prof = prof_start();
            attention_token(p, s, layer, base_pos + index);
            matmul_quantized_with_scratch(
                &mut s.xb2[..p.dim],
                &s.xb[..s.q_dim],
                &w.wo[layer],
                mapped,
                &mut matmul,
            )?;
            sanitize_finite_inplace(&mut s.xb2[..p.dim]);
            prof_end(&PROF_ATTN_NS, attention_prof);
            layer_after_attention(p, s, w, mapped, &mut matmul, layer, base_pos + index, false)?;
            row.copy_from_slice(&s.x[..p.dim]);
        }
    }
    if compute_logits {
        finish_logits(p, s, w, mapped, &mut matmul)?;
    }
    Ok(())
}
