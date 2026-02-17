use crate::engine::io::find_gguf_tensor;
use crate::engine::kernels::{dequantize_tensor, get_block_size, get_type_size};
use crate::engine::types::{
    Config, GGUFFile, GgmlType, Gguftensor, QuantizedTensor, TransformerWeights, GGML_TYPE_F32,
};

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

    if let Some(expected) = expected_elements {
        if expected != n_elements {
            return Err(format!(
                "tensor {name} has {n_elements} elements, expected {expected}"
            ));
        }
    }

    let block_size = get_block_size(tensor.ttype);
    let type_size = get_type_size(tensor.ttype);
    if type_size == 0 {
        return Err(format!(
            "unsupported tensor type {} for {name}",
            tensor.ttype.0
        ));
    }

    if n_elements % block_size != 0 {
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
    if cols == 0 || n_elements % cols != 0 {
        return Err(format!(
            "tensor {name} element count {n_elements} is not divisible by cols={cols}"
        ));
    }
    let rows = n_elements / cols;
    load_tensor_quantized(gguf, &name, rows, cols)
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
    let deepseek_qk_head_dim = p.deepseek_qk_nope_head_dim + p.deepseek_qk_rope_head_dim;
    let deepseek_q_dim = p.n_heads * deepseek_qk_head_dim;
    let deepseek_v_dim = p.n_heads * p.deepseek_v_head_dim;
    let deepseek_kv_a_rows = p.deepseek_kv_lora_rank + p.deepseek_qk_rope_head_dim;
    let deepseek_k_absorb_rows = p.n_heads * p.deepseek_kv_lora_rank;

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
    let mut deepseek_wq_a = if p.is_deepseek2 {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut deepseek_wq_b = if p.is_deepseek2 {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut deepseek_wkv_a = if p.is_deepseek2 {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut deepseek_wk_b = if p.is_deepseek2 {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut deepseek_wv_b = if p.is_deepseek2 {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut deepseek_q_a_norm = if p.is_deepseek2 {
        vec![0.0f32; n_layers * p.deepseek_q_lora_rank]
    } else {
        Vec::new()
    };
    let mut deepseek_kv_a_norm = if p.is_deepseek2 {
        vec![0.0f32; n_layers * p.deepseek_kv_lora_rank]
    } else {
        Vec::new()
    };
    let mut deepseek_layer_is_moe = if p.is_deepseek2 {
        vec![false; n_layers]
    } else {
        Vec::new()
    };
    let mut deepseek_shared_gate_present = if p.is_deepseek2 {
        vec![false; n_layers]
    } else {
        Vec::new()
    };
    let mut deepseek_exp_probs_bias = if p.is_deepseek2 {
        vec![0.0f32; n_layers * p.n_experts]
    } else {
        Vec::new()
    };
    let mut ssm_ba = if p.is_qwen3next {
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
    let mut moe_gate_inp = if p.is_qwen3moe || p.is_qwen3next || p.is_deepseek2 {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_gate_exps = if p.is_qwen3moe || p.is_qwen3next || p.is_deepseek2 {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_up_exps = if p.is_qwen3moe || p.is_qwen3next || p.is_deepseek2 {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_down_exps = if p.is_qwen3moe || p.is_qwen3next || p.is_deepseek2 {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_shared_gate_inp = if p.is_qwen3next || p.is_deepseek2 {
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

    let mut attn_q_norm = if p.is_gemma3 || p.is_qwen2 || p.is_qwen3moe || p.is_qwen3next {
        vec![0.0f32; n_layers * head_size]
    } else {
        Vec::new()
    };
    let mut attn_k_norm = if p.is_gemma3 || p.is_qwen2 || p.is_qwen3moe || p.is_qwen3next {
        vec![0.0f32; n_layers * head_size]
    } else {
        Vec::new()
    };
    let mut attn_qk_norm_present = if p.is_gemma3 || p.is_qwen2 || p.is_qwen3moe || p.is_qwen3next {
        vec![false; n_layers]
    } else {
        Vec::new()
    };
    let mut attn_post_norm = if p.is_gemma3 {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };
    let mut ffn_post_norm = if p.is_gemma3 {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };

    for l in 0..n_layers {
        let attn_norm = load_layer_tensor_float(gguf, l, "attn_norm.weight", p.dim)?;
        rms_att_weight[l * p.dim..(l + 1) * p.dim].copy_from_slice(&attn_norm);

        let ffn_norm = if p.is_qwen3next {
            load_layer_tensor_float(gguf, l, "post_attention_norm.weight", p.dim)?
        } else {
            load_layer_tensor_float(gguf, l, "ffn_norm.weight", p.dim)?
        };
        rms_ffn_weight[l * p.dim..(l + 1) * p.dim].copy_from_slice(&ffn_norm);

        if p.is_deepseek2 {
            deepseek_wq_a[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "attn_q_a.weight",
                p.deepseek_q_lora_rank,
                p.dim,
            )?;
            deepseek_wq_b[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "attn_q_b.weight",
                deepseek_q_dim,
                p.deepseek_q_lora_rank,
            )?;
            deepseek_wkv_a[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "attn_kv_a_mqa.weight",
                deepseek_kv_a_rows,
                p.dim,
            )?;
            deepseek_wk_b[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "attn_k_b.weight",
                deepseek_k_absorb_rows,
                p.deepseek_qk_nope_head_dim,
            )?;
            deepseek_wv_b[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "attn_v_b.weight",
                deepseek_v_dim,
                p.deepseek_kv_lora_rank,
            )?;
            wo[l] =
                load_layer_tensor_quantized(gguf, l, "attn_output.weight", p.dim, deepseek_v_dim)?;

            let q_a_norm =
                load_layer_tensor_float(gguf, l, "attn_q_a_norm.weight", p.deepseek_q_lora_rank)?;
            deepseek_q_a_norm[l * p.deepseek_q_lora_rank..(l + 1) * p.deepseek_q_lora_rank]
                .copy_from_slice(&q_a_norm);

            let kv_a_norm =
                load_layer_tensor_float(gguf, l, "attn_kv_a_norm.weight", p.deepseek_kv_lora_rank)?;
            deepseek_kv_a_norm[l * p.deepseek_kv_lora_rank..(l + 1) * p.deepseek_kv_lora_rank]
                .copy_from_slice(&kv_a_norm);

            let has_moe =
                find_gguf_tensor(gguf, &format!("blk.{l}.ffn_gate_exps.weight")).is_some();
            deepseek_layer_is_moe[l] = has_moe;

            if has_moe {
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
                if !deepseek_exp_probs_bias.is_empty()
                    && find_gguf_tensor(gguf, &format!("blk.{l}.exp_probs_b.bias")).is_some()
                {
                    let bias = load_layer_tensor_float(gguf, l, "exp_probs_b.bias", p.n_experts)?;
                    deepseek_exp_probs_bias[l * p.n_experts..(l + 1) * p.n_experts]
                        .copy_from_slice(&bias);
                }

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
                if find_gguf_tensor(gguf, &format!("blk.{l}.ffn_gate_inp_shexp.weight")).is_some() {
                    let shexp_gate =
                        load_layer_tensor_float(gguf, l, "ffn_gate_inp_shexp.weight", p.dim)?;
                    moe_shared_gate_inp[l * p.dim..(l + 1) * p.dim].copy_from_slice(&shexp_gate);
                    deepseek_shared_gate_present[l] = true;
                }
            } else {
                w1[l] =
                    load_layer_tensor_quantized(gguf, l, "ffn_gate.weight", p.hidden_dim, p.dim)?;
                w2[l] =
                    load_layer_tensor_quantized(gguf, l, "ffn_down.weight", p.dim, p.hidden_dim)?;
                w3[l] = load_layer_tensor_quantized(gguf, l, "ffn_up.weight", p.hidden_dim, p.dim)?;
            }
        } else if p.is_qwen3next {
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
                ssm_ba[l] =
                    load_layer_tensor_quantized(gguf, l, "ssm_ba.weight", 2 * ssm_v_heads, p.dim)?;
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
            w3[l] =
                load_layer_tensor_quantized(gguf, l, "ffn_up_shexp.weight", shared_hidden, p.dim)?;
            let shexp_gate = load_layer_tensor_float(gguf, l, "ffn_gate_inp_shexp.weight", p.dim)?;
            moe_shared_gate_inp[l * p.dim..(l + 1) * p.dim].copy_from_slice(&shexp_gate);
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
        } else if !p.is_qwen3next && !p.is_deepseek2 {
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

        if p.is_gemma3 || p.is_qwen3moe {
            let q_norm = load_layer_tensor_float(gguf, l, "attn_q_norm.weight", head_size)?;
            let k_norm = load_layer_tensor_float(gguf, l, "attn_k_norm.weight", head_size)?;
            attn_q_norm[l * head_size..(l + 1) * head_size].copy_from_slice(&q_norm);
            attn_k_norm[l * head_size..(l + 1) * head_size].copy_from_slice(&k_norm);
            attn_qk_norm_present[l] = true;
        } else if (p.is_qwen3next || p.is_qwen2)
            && find_gguf_tensor(gguf, &format!("blk.{l}.attn_q_norm.weight")).is_some()
            && find_gguf_tensor(gguf, &format!("blk.{l}.attn_k_norm.weight")).is_some()
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
        }
    }

    let rms_final_weight = load_tensor_float(gguf, "output_norm.weight", Some(p.dim))?;

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
        deepseek_wq_a,
        deepseek_wq_b,
        deepseek_wkv_a,
        deepseek_wk_b,
        deepseek_wv_b,
        deepseek_q_a_norm,
        deepseek_kv_a_norm,
        deepseek_layer_is_moe,
        deepseek_shared_gate_present,
        deepseek_exp_probs_bias,
        ssm_ba,
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
    })
}
