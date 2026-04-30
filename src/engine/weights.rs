use crate::engine::io::{find_gguf_tensor, find_gguf_tensor_names_with_any_prefix};
use crate::engine::kernels::{dequantize_tensor, get_block_size, get_type_size};
use crate::engine::types::{
    Config, GGML_TYPE_F32, GGUFFile, GgmlType, Gguftensor, MultimodalBackend, MultimodalWeights,
    QuantizedTensor, TransformerWeights, WorkerExpertWeights,
};
use std::collections::BTreeMap;

fn tensor_n_elements(tensor: &Gguftensor) -> usize {
    let mut n_elements = 1usize;
    for i in 0..tensor.n_dims as usize {
        n_elements = n_elements.saturating_mul(tensor.ne[i] as usize);
    }
    n_elements
}

fn load_tensor_float(
    gguf: &GGUFFile,
    name: &str,
    expected_elements: Option<usize>,
) -> Result<Vec<f32>, String> {
    let tensor = find_gguf_tensor(gguf, name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let n_elements = tensor_n_elements(tensor);

    if let Some(expected) = expected_elements
        && expected != n_elements
    {
        return Err(format!(
            "tensor {name} has {n_elements} elements, expected {expected}"
        ));
    }

    let block_size = get_block_size(tensor.ttype);
    let type_size = get_type_size(tensor.ttype);
    if type_size == 0 {
        return Err(format!(
            "unsupported tensor type {} for {name}",
            tensor.ttype.0
        ));
    }

    if !n_elements.is_multiple_of(block_size) {
        return Err(format!(
            "tensor {name} element count {n_elements} not divisible by block size {block_size}"
        ));
    }

    let src_size = (n_elements / block_size) * type_size;
    let mapped = gguf.mapped.as_slice();
    if tensor.data_offset + src_size > mapped.len() {
        return Err(format!("tensor {name} exceeds mapped file bounds"));
    }
    gguf.ensure_range(tensor.data_offset, src_size)?;
    let src = &mapped[tensor.data_offset..tensor.data_offset + src_size];

    dequantize_tensor(src, n_elements, tensor.ttype)
}

fn load_tensor_quantized(
    gguf: &GGUFFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<QuantizedTensor, String> {
    let tensor = find_gguf_tensor(gguf, name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let n_elements = tensor_n_elements(tensor);
    if n_elements != rows.saturating_mul(cols) {
        return Err(format!(
            "tensor {name} shape mismatch: got {} elements, expected {} (rows={rows}, cols={cols})",
            n_elements,
            rows.saturating_mul(cols)
        ));
    }

    Ok(QuantizedTensor {
        data_offset: tensor.data_offset,
        ttype: tensor.ttype,
        rows,
        cols,
    })
}

fn load_layer_tensor_float(
    gguf: &GGUFFile,
    layer: usize,
    suffix: &str,
    expected_elements: usize,
) -> Result<Vec<f32>, String> {
    let name = format!("blk.{layer}.{suffix}");
    load_tensor_float(gguf, &name, Some(expected_elements))
}

fn load_layer_tensor_quantized(
    gguf: &GGUFFile,
    layer: usize,
    suffix: &str,
    rows: usize,
    cols: usize,
) -> Result<QuantizedTensor, String> {
    let name = format!("blk.{layer}.{suffix}");
    load_tensor_quantized(gguf, &name, rows, cols)
}

fn load_layer_tensor_quantized_auto_rows(
    gguf: &GGUFFile,
    layer: usize,
    suffix: &str,
    cols: usize,
) -> Result<QuantizedTensor, String> {
    let name = format!("blk.{layer}.{suffix}");
    let tensor =
        find_gguf_tensor(gguf, &name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let n_elements = tensor_n_elements(tensor);
    if cols == 0 || !n_elements.is_multiple_of(cols) {
        return Err(format!(
            "tensor {name} element count {n_elements} is not divisible by cols={cols}"
        ));
    }
    let rows = n_elements / cols;
    load_tensor_quantized(gguf, &name, rows, cols)
}

const VISION_ENCODER_PREFIXES: &[&str] = &[
    "vision.",
    "visual.",
    "vision_tower.",
    "model.vision.",
    "model.visual.",
    "v.",
];
const VISION_PROJECTOR_PREFIXES: &[&str] = &[
    "mm.",
    "mmproj.",
    "multi_modal_projector.",
    "projector.",
    "model.mmproj.",
    "model.projector.",
];
const AUDIO_PREFIXES: &[&str] = &["audio.", "aud.", "speech.", "whisper.", "model.audio."];

fn summarize_tensor_prefixes(gguf: &GGUFFile, max_items: usize) -> String {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &gguf.tensors {
        let key = tensor
            .name
            .split('.')
            .next()
            .unwrap_or("unknown")
            .to_string();
        *counts.entry(key).or_insert(0) += 1;
    }
    let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    entries
        .into_iter()
        .take(max_items)
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn init_multimodal_weights_from_gguf(
    gguf: &GGUFFile,
    p: &Config,
    debug_mode: bool,
) -> Result<Option<MultimodalWeights>, String> {
    if p.capabilities.multimodal_backend == MultimodalBackend::None {
        return Ok(None);
    }

    let vision_tensor_names = find_gguf_tensor_names_with_any_prefix(gguf, VISION_ENCODER_PREFIXES);
    let projector_tensor_names =
        find_gguf_tensor_names_with_any_prefix(gguf, VISION_PROJECTOR_PREFIXES);
    let audio_tensor_names = find_gguf_tensor_names_with_any_prefix(gguf, AUDIO_PREFIXES);

    let needs_vision = p.capabilities.supports_native_image || p.capabilities.supports_native_video;
    if needs_vision {
        let mut missing: Vec<&str> = Vec::new();
        if vision_tensor_names.is_empty() {
            missing.push("vision encoder tensor group");
        }
        if projector_tensor_names.is_empty() {
            missing.push("multimodal projector tensor group");
        }
        if !missing.is_empty() {
            return Err(format!(
                "native multimodal backend '{}' is enabled but required tensors are missing: {}. observed GGUF tensor prefixes: [{}]",
                p.capabilities.multimodal_backend.as_str(),
                missing.join(", "),
                summarize_tensor_prefixes(gguf, 8),
            ));
        }
    }

    if p.capabilities.supports_native_audio && audio_tensor_names.is_empty() {
        return Err(format!(
            "native multimodal backend '{}' reports audio support but GGUF contains no audio tensor group. observed GGUF tensor prefixes: [{}]",
            p.capabilities.multimodal_backend.as_str(),
            summarize_tensor_prefixes(gguf, 8)
        ));
    }

    if debug_mode {
        let preview = |names: &[String]| -> String {
            if names.is_empty() {
                "none".to_string()
            } else {
                names.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
            }
        };
        eprintln!(
            "Multimodal weights probe: backend={}, vision={}, projector={}, audio={}",
            p.capabilities.multimodal_backend.as_str(),
            vision_tensor_names.len(),
            projector_tensor_names.len(),
            audio_tensor_names.len()
        );
        eprintln!(
            "Multimodal weights sample tensors: vision=[{}] projector=[{}] audio=[{}]",
            preview(&vision_tensor_names),
            preview(&projector_tensor_names),
            preview(&audio_tensor_names)
        );
    }

    Ok(Some(MultimodalWeights {
        backend: p.capabilities.multimodal_backend,
        vision_tensor_names,
        projector_tensor_names,
        audio_tensor_names,
    }))
}

pub(crate) fn init_weights_from_gguf(
    gguf: &GGUFFile,
    p: &Config,
    debug_mode: bool,
) -> Result<TransformerWeights, String> {
    let head_size = if p.head_dim > 0 {
        p.head_dim
    } else {
        p.dim / p.n_heads
    };
    let kv_dim = p.n_kv_heads * head_size;
    let q_dim = p.n_heads * head_size;
    let n_layers = p.n_layers;
    let ssm_inner = p.ssm_inner_size;
    let ssm_k_heads = p.ssm_group_count;
    let ssm_v_heads = p.ssm_time_step_rank;
    let ssm_head_dim = p.ssm_state_size;
    let ssm_conv_dim = ssm_inner + 2 * ssm_k_heads * ssm_head_dim;

    let token_embedding_table =
        load_tensor_float(gguf, "token_embd.weight", Some(p.vocab_size * p.dim))?;

    let mut rms_att_weight = vec![0.0f32; n_layers * p.dim];
    let mut rms_ffn_weight = vec![0.0f32; n_layers * p.dim];

    let mut wq = vec![QuantizedTensor::default(); n_layers];
    let mut wk = vec![QuantizedTensor::default(); n_layers];
    let mut wv = vec![QuantizedTensor::default(); n_layers];
    let mut wo = vec![QuantizedTensor::default(); n_layers];
    let mut w1 = vec![QuantizedTensor::default(); n_layers];
    let mut w2 = vec![QuantizedTensor::default(); n_layers];
    let mut w3 = vec![QuantizedTensor::default(); n_layers];
    let mut attn_qkv = if p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut ssm_ba = if p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut ssm_alpha = if p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut ssm_beta = if p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut ssm_conv1d = if p.is_qwen3next {
        vec![Vec::new(); n_layers]
    } else {
        Vec::new()
    };
    let mut ssm_a = if p.is_qwen3next {
        vec![0.0f32; n_layers * ssm_v_heads]
    } else {
        Vec::new()
    };
    let mut ssm_dt_bias = if p.is_qwen3next {
        vec![0.0f32; n_layers * ssm_v_heads]
    } else {
        Vec::new()
    };
    let mut ssm_norm = if p.is_qwen3next {
        vec![0.0f32; n_layers * ssm_head_dim]
    } else {
        Vec::new()
    };
    let mut moe_gate_inp = if p.is_qwen3moe || p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_gate_exps = if p.is_qwen3moe || p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_up_exps = if p.is_qwen3moe || p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_down_exps = if p.is_qwen3moe || p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_shared_gate_inp = if p.is_qwen3next {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };

    let mut attn_q_bias = if p.is_qwen2 {
        vec![0.0f32; n_layers * q_dim]
    } else {
        Vec::new()
    };
    let mut attn_k_bias = if p.is_qwen2 {
        vec![0.0f32; n_layers * kv_dim]
    } else {
        Vec::new()
    };
    let mut attn_v_bias = if p.is_qwen2 {
        vec![0.0f32; n_layers * kv_dim]
    } else {
        Vec::new()
    };

    let mut attn_q_norm =
        if p.is_gemma3 || p.is_qwen2 || p.is_qwen3vl || p.is_qwen3moe || p.is_qwen3next {
            vec![0.0f32; n_layers * head_size]
        } else {
            Vec::new()
        };
    let mut attn_k_norm =
        if p.is_gemma3 || p.is_qwen2 || p.is_qwen3vl || p.is_qwen3moe || p.is_qwen3next {
            vec![0.0f32; n_layers * head_size]
        } else {
            Vec::new()
        };
    let mut attn_qk_norm_present =
        if p.is_gemma3 || p.is_qwen2 || p.is_qwen3vl || p.is_qwen3moe || p.is_qwen3next {
            vec![false; n_layers]
        } else {
            Vec::new()
        };
    let mut attn_post_norm = if p.is_gemma3 || p.is_bert_family {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };
    let mut ffn_post_norm = if p.is_gemma3 || p.is_bert_family {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };
    let mut attn_post_norm_bias: Vec<f32> = if p.is_bert_family {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };
    let mut ffn_post_norm_bias: Vec<f32> = if p.is_bert_family {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };

    for l in 0..n_layers {
        // BERT-family uses post-norm: skip pre-norm tensors (they don't exist in the GGUF).
        // Fill with 1.0 so that rmsnorm(x, weight=1) ≈ identity (pre-norm is bypassed in
        // inference when is_bert_family is true; these slots are never actually read then).
        if p.is_bert_family {
            rms_att_weight[l * p.dim..(l + 1) * p.dim].fill(1.0);
            rms_ffn_weight[l * p.dim..(l + 1) * p.dim].fill(1.0);
        } else {
            let attn_norm = load_layer_tensor_float(gguf, l, "attn_norm.weight", p.dim)?;
            rms_att_weight[l * p.dim..(l + 1) * p.dim].copy_from_slice(&attn_norm);

            let ffn_norm = if p.is_qwen3next {
                load_layer_tensor_float(gguf, l, "post_attention_norm.weight", p.dim)?
            } else {
                load_layer_tensor_float(gguf, l, "ffn_norm.weight", p.dim)?
            };
            rms_ffn_weight[l * p.dim..(l + 1) * p.dim].copy_from_slice(&ffn_norm);
        }

        if p.is_qwen3next {
            if find_gguf_tensor(gguf, &format!("blk.{l}.attn_qkv.weight")).is_some() {
                attn_qkv[l] =
                    load_layer_tensor_quantized_auto_rows(gguf, l, "attn_qkv.weight", p.dim)?;
                if attn_qkv[l].rows < ssm_conv_dim {
                    return Err(format!(
                        "blk.{l}.attn_qkv.weight has {} rows, expected at least {}",
                        attn_qkv[l].rows, ssm_conv_dim
                    ));
                }
                wo[l] = load_layer_tensor_quantized(gguf, l, "attn_gate.weight", ssm_inner, p.dim)?;
                wv[l] = load_layer_tensor_quantized(gguf, l, "ssm_out.weight", p.dim, ssm_inner)?;
                if find_gguf_tensor(gguf, &format!("blk.{l}.ssm_ba.weight")).is_some() {
                    ssm_ba[l] = load_layer_tensor_quantized(
                        gguf,
                        l,
                        "ssm_ba.weight",
                        2 * ssm_v_heads,
                        p.dim,
                    )?;
                } else if find_gguf_tensor(gguf, &format!("blk.{l}.ssm_alpha.weight")).is_some()
                    && find_gguf_tensor(gguf, &format!("blk.{l}.ssm_beta.weight")).is_some()
                {
                    ssm_alpha[l] =
                        load_layer_tensor_quantized_auto_rows(gguf, l, "ssm_alpha.weight", p.dim)?;
                    ssm_beta[l] =
                        load_layer_tensor_quantized_auto_rows(gguf, l, "ssm_beta.weight", p.dim)?;
                    if ssm_alpha[l].rows < ssm_v_heads {
                        return Err(format!(
                            "blk.{l}.ssm_alpha.weight has {} rows, expected at least {}",
                            ssm_alpha[l].rows, ssm_v_heads
                        ));
                    }
                    if ssm_beta[l].rows < ssm_v_heads {
                        return Err(format!(
                            "blk.{l}.ssm_beta.weight has {} rows, expected at least {}",
                            ssm_beta[l].rows, ssm_v_heads
                        ));
                    }
                } else {
                    return Err(format!(
                        "blk.{l} is missing SSM gate tensors: expected either ssm_ba.weight or both ssm_alpha.weight and ssm_beta.weight"
                    ));
                }
                ssm_conv1d[l] = load_tensor_float(
                    gguf,
                    &format!("blk.{l}.ssm_conv1d.weight"),
                    Some(p.ssm_conv_kernel * ssm_conv_dim),
                )?;
                let a = load_layer_tensor_float(gguf, l, "ssm_a", ssm_v_heads)?;
                ssm_a[l * ssm_v_heads..(l + 1) * ssm_v_heads].copy_from_slice(&a);
                let dt = load_layer_tensor_float(gguf, l, "ssm_dt.bias", ssm_v_heads)?;
                ssm_dt_bias[l * ssm_v_heads..(l + 1) * ssm_v_heads].copy_from_slice(&dt);
                let n = load_layer_tensor_float(gguf, l, "ssm_norm.weight", ssm_head_dim)?;
                ssm_norm[l * ssm_head_dim..(l + 1) * ssm_head_dim].copy_from_slice(&n);
                if debug_mode {
                    eprintln!(
                        "qwen3next layer {l}: recurrent qkv_rows={} (t={}) gate_rows={} (t={}) out_rows={} (t={})",
                        attn_qkv[l].rows,
                        attn_qkv[l].ttype.0,
                        wo[l].rows,
                        wo[l].ttype.0,
                        wv[l].rows,
                        wv[l].ttype.0
                    );
                    if ssm_ba[l].rows > 0 {
                        eprintln!(
                            "qwen3next layer {l}: using fused ssm_ba.weight rows={} (t={})",
                            ssm_ba[l].rows, ssm_ba[l].ttype.0
                        );
                    } else if ssm_alpha[l].rows > 0 && ssm_beta[l].rows > 0 {
                        eprintln!(
                            "qwen3next layer {l}: using split ssm_alpha/ssm_beta rows=({},{})",
                            ssm_alpha[l].rows, ssm_beta[l].rows
                        );
                    }
                    if l == 0 {
                        let (mut amin, mut amax) = (f32::INFINITY, f32::NEG_INFINITY);
                        let (mut dtmin, mut dtmax) = (f32::INFINITY, f32::NEG_INFINITY);
                        for &v in &a {
                            amin = amin.min(v);
                            amax = amax.max(v);
                        }
                        for &v in &dt {
                            dtmin = dtmin.min(v);
                            dtmax = dtmax.max(v);
                        }
                        eprintln!(
                            "qwen3next layer 0: ssm_a[min={amin:.6}, max={amax:.6}] ssm_dt.bias[min={dtmin:.6}, max={dtmax:.6}]"
                        );
                    }
                }
            } else {
                wq[l] = load_layer_tensor_quantized_auto_rows(gguf, l, "attn_q.weight", p.dim)?;
                if wq[l].rows < q_dim {
                    return Err(format!(
                        "blk.{l}.attn_q.weight has {} rows, expected at least {}",
                        wq[l].rows, q_dim
                    ));
                }
                wk[l] = load_layer_tensor_quantized_auto_rows(gguf, l, "attn_k.weight", p.dim)?;
                if wk[l].rows < kv_dim {
                    return Err(format!(
                        "blk.{l}.attn_k.weight has {} rows, expected at least {}",
                        wk[l].rows, kv_dim
                    ));
                }
                wv[l] = load_layer_tensor_quantized_auto_rows(gguf, l, "attn_v.weight", p.dim)?;
                if wv[l].rows < kv_dim {
                    return Err(format!(
                        "blk.{l}.attn_v.weight has {} rows, expected at least {}",
                        wv[l].rows, kv_dim
                    ));
                }
                wo[l] = load_layer_tensor_quantized(gguf, l, "attn_output.weight", p.dim, q_dim)?;
                if debug_mode {
                    eprintln!(
                        "qwen3next layer {l}: full q_rows={} (t={}) k_rows={} (t={}) v_rows={} (t={}) o_rows={} (t={})",
                        wq[l].rows,
                        wq[l].ttype.0,
                        wk[l].rows,
                        wk[l].ttype.0,
                        wv[l].rows,
                        wv[l].ttype.0,
                        wo[l].rows,
                        wo[l].ttype.0
                    );
                }
            }
            if p.n_experts > 0 {
                moe_gate_inp[l] = load_layer_tensor_quantized(
                    gguf,
                    l,
                    "ffn_gate_inp.weight",
                    p.n_experts,
                    p.dim,
                )?;
                moe_gate_exps[l] = load_layer_tensor_quantized(
                    gguf,
                    l,
                    "ffn_gate_exps.weight",
                    p.n_experts * p.expert_hidden_dim,
                    p.dim,
                )?;
                moe_up_exps[l] = load_layer_tensor_quantized(
                    gguf,
                    l,
                    "ffn_up_exps.weight",
                    p.n_experts * p.expert_hidden_dim,
                    p.dim,
                )?;
                moe_down_exps[l] = load_layer_tensor_quantized(
                    gguf,
                    l,
                    "ffn_down_exps.weight",
                    p.n_experts * p.dim,
                    p.expert_hidden_dim,
                )?;

                let shared_hidden = if p.shared_expert_hidden_dim > 0 {
                    p.shared_expert_hidden_dim
                } else {
                    p.expert_hidden_dim
                };
                w1[l] = load_layer_tensor_quantized(
                    gguf,
                    l,
                    "ffn_gate_shexp.weight",
                    shared_hidden,
                    p.dim,
                )?;
                w2[l] = load_layer_tensor_quantized(
                    gguf,
                    l,
                    "ffn_down_shexp.weight",
                    p.dim,
                    shared_hidden,
                )?;
                w3[l] = load_layer_tensor_quantized(
                    gguf,
                    l,
                    "ffn_up_shexp.weight",
                    shared_hidden,
                    p.dim,
                )?;
                let shexp_gate =
                    load_layer_tensor_float(gguf, l, "ffn_gate_inp_shexp.weight", p.dim)?;
                moe_shared_gate_inp[l * p.dim..(l + 1) * p.dim].copy_from_slice(&shexp_gate);
            } else {
                w1[l] =
                    load_layer_tensor_quantized(gguf, l, "ffn_gate.weight", p.hidden_dim, p.dim)?;
                w2[l] =
                    load_layer_tensor_quantized(gguf, l, "ffn_down.weight", p.dim, p.hidden_dim)?;
                w3[l] = load_layer_tensor_quantized(gguf, l, "ffn_up.weight", p.hidden_dim, p.dim)?;
            }
        } else if p.is_bert_family
            && find_gguf_tensor(gguf, &format!("blk.{l}.attn_qkv.weight")).is_some()
        {
            // BERT-style fused QKV (nomic-bert, all-minilm …).
            // Pack into wq[l] with rows = q_dim + 2*kv_dim; inference extracts slices via
            // matmul_quantized_rows.
            let total_qkv = q_dim + 2 * kv_dim;
            wq[l] = load_layer_tensor_quantized(gguf, l, "attn_qkv.weight", total_qkv, p.dim)?;
            wo[l] = load_layer_tensor_quantized(gguf, l, "attn_output.weight", p.dim, q_dim)?;
        } else {
            wq[l] = load_layer_tensor_quantized(gguf, l, "attn_q.weight", q_dim, p.dim)?;
            wk[l] = load_layer_tensor_quantized(gguf, l, "attn_k.weight", kv_dim, p.dim)?;
            wv[l] = load_layer_tensor_quantized(gguf, l, "attn_v.weight", kv_dim, p.dim)?;
            wo[l] = load_layer_tensor_quantized(gguf, l, "attn_output.weight", p.dim, q_dim)?;
        }
        if p.is_qwen3moe {
            moe_gate_inp[l] =
                load_layer_tensor_quantized(gguf, l, "ffn_gate_inp.weight", p.n_experts, p.dim)?;
            moe_gate_exps[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_gate_exps.weight",
                p.n_experts * p.expert_hidden_dim,
                p.dim,
            )?;
            moe_up_exps[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_up_exps.weight",
                p.n_experts * p.expert_hidden_dim,
                p.dim,
            )?;
            moe_down_exps[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_down_exps.weight",
                p.n_experts * p.dim,
                p.expert_hidden_dim,
            )?;
        } else if !p.is_qwen3next {
            w1[l] = load_layer_tensor_quantized(gguf, l, "ffn_gate.weight", p.hidden_dim, p.dim)?;
            w2[l] = load_layer_tensor_quantized(gguf, l, "ffn_down.weight", p.dim, p.hidden_dim)?;
            w3[l] = load_layer_tensor_quantized(gguf, l, "ffn_up.weight", p.hidden_dim, p.dim)?;
        }

        if p.is_qwen2 {
            if let Ok(qb) = load_layer_tensor_float(gguf, l, "attn_q.bias", q_dim) {
                attn_q_bias[l * q_dim..(l + 1) * q_dim].copy_from_slice(&qb);
            }
            if let Ok(kb) = load_layer_tensor_float(gguf, l, "attn_k.bias", kv_dim) {
                attn_k_bias[l * kv_dim..(l + 1) * kv_dim].copy_from_slice(&kb);
            }
            if let Ok(vb) = load_layer_tensor_float(gguf, l, "attn_v.bias", kv_dim) {
                attn_v_bias[l * kv_dim..(l + 1) * kv_dim].copy_from_slice(&vb);
            }
        }

        if p.is_gemma3
            || p.is_qwen3moe
            || ((p.is_qwen3next || p.is_qwen2 || p.is_qwen3vl)
                && find_gguf_tensor(gguf, &format!("blk.{l}.attn_q_norm.weight")).is_some()
                && find_gguf_tensor(gguf, &format!("blk.{l}.attn_k_norm.weight")).is_some())
        {
            let q_norm = load_layer_tensor_float(gguf, l, "attn_q_norm.weight", head_size)?;
            let k_norm = load_layer_tensor_float(gguf, l, "attn_k_norm.weight", head_size)?;
            attn_q_norm[l * head_size..(l + 1) * head_size].copy_from_slice(&q_norm);
            attn_k_norm[l * head_size..(l + 1) * head_size].copy_from_slice(&k_norm);
            attn_qk_norm_present[l] = true;
        }

        if p.is_gemma3 {
            let pan = load_layer_tensor_float(gguf, l, "post_attention_norm.weight", p.dim)?;
            attn_post_norm[l * p.dim..(l + 1) * p.dim].copy_from_slice(&pan);

            let pfn = load_layer_tensor_float(gguf, l, "post_ffw_norm.weight", p.dim)?;
            ffn_post_norm[l * p.dim..(l + 1) * p.dim].copy_from_slice(&pfn);
        } else if p.is_bert_family {
            // Post-attention LayerNorm (applied after attn residual).
            let pan = load_layer_tensor_float(gguf, l, "attn_output_norm.weight", p.dim)?;
            attn_post_norm[l * p.dim..(l + 1) * p.dim].copy_from_slice(&pan);
            let pan_b = load_layer_tensor_float(gguf, l, "attn_output_norm.bias", p.dim)?;
            attn_post_norm_bias[l * p.dim..(l + 1) * p.dim].copy_from_slice(&pan_b);

            // Post-FFN LayerNorm (applied after FFN residual).
            let pfn = load_layer_tensor_float(gguf, l, "layer_output_norm.weight", p.dim)?;
            ffn_post_norm[l * p.dim..(l + 1) * p.dim].copy_from_slice(&pfn);
            let pfn_b = load_layer_tensor_float(gguf, l, "layer_output_norm.bias", p.dim)?;
            ffn_post_norm_bias[l * p.dim..(l + 1) * p.dim].copy_from_slice(&pfn_b);
        }
    }

    // BERT models have no output projection norm; the per-block post-norms are the final norms.
    let rms_final_weight =
        if p.is_bert_family && find_gguf_tensor(gguf, "output_norm.weight").is_none() {
            vec![1.0f32; p.dim]
        } else {
            load_tensor_float(gguf, "output_norm.weight", Some(p.dim))?
        };

    let mut wcls_is_embed = false;
    let wcls = if find_gguf_tensor(gguf, "output.weight").is_some() {
        load_tensor_quantized(gguf, "output.weight", p.vocab_size, p.dim)?
    } else {
        if debug_mode {
            eprintln!("Using tied embeddings for output projection");
        }
        wcls_is_embed = true;
        QuantizedTensor {
            data_offset: usize::MAX,
            ttype: GgmlType(GGML_TYPE_F32),
            rows: p.vocab_size,
            cols: p.dim,
        }
    };

    Ok(TransformerWeights {
        token_embedding_table,
        rms_att_weight,
        rms_ffn_weight,
        wq,
        wk,
        wv,
        wo,
        w1,
        w2,
        w3,
        attn_qkv,
        ssm_ba,
        ssm_alpha,
        ssm_beta,
        ssm_conv1d,
        ssm_a,
        ssm_dt_bias,
        ssm_norm,
        moe_gate_inp,
        moe_gate_exps,
        moe_up_exps,
        moe_down_exps,
        moe_shared_gate_inp,
        rms_final_weight,
        wcls,
        wcls_is_embed,
        attn_q_bias,
        attn_k_bias,
        attn_v_bias,
        attn_q_norm,
        attn_k_norm,
        attn_qk_norm_present,
        attn_post_norm,
        ffn_post_norm,
        attn_post_norm_bias,
        ffn_post_norm_bias,
    })
}

pub(crate) fn init_worker_expert_weights_from_gguf(
    gguf: &GGUFFile,
    p: &Config,
) -> Result<WorkerExpertWeights, String> {
    if p.n_experts == 0 || p.expert_hidden_dim == 0 {
        return Err("worker expert loading requires a routed-MoE model".to_string());
    }

    let mut moe_gate_exps = vec![QuantizedTensor::default(); p.n_layers];
    let mut moe_up_exps = vec![QuantizedTensor::default(); p.n_layers];
    let mut moe_down_exps = vec![QuantizedTensor::default(); p.n_layers];

    for l in 0..p.n_layers {
        moe_gate_exps[l] = load_layer_tensor_quantized(
            gguf,
            l,
            "ffn_gate_exps.weight",
            p.n_experts * p.expert_hidden_dim,
            p.dim,
        )?;
        moe_up_exps[l] = load_layer_tensor_quantized(
            gguf,
            l,
            "ffn_up_exps.weight",
            p.n_experts * p.expert_hidden_dim,
            p.dim,
        )?;
        moe_down_exps[l] = load_layer_tensor_quantized(
            gguf,
            l,
            "ffn_down_exps.weight",
            p.n_experts * p.dim,
            p.expert_hidden_dim,
        )?;
    }

    Ok(WorkerExpertWeights {
        moe_gate_exps,
        moe_up_exps,
        moe_down_exps,
    })
}
