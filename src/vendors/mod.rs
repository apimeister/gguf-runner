mod deepseek;
mod gemma;
mod llama;
mod qwen;

use crate::engine::io::{get_gguf_float_from_map, get_gguf_int_from_map, get_gguf_string_from_map};
use crate::engine::types::{Config, GGUFFile, Tokenizer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelFamily {
    Llama,
    Gemma,
    Qwen2,
    Qwen3Moe,
    Qwen3Next,
    DeepSeek2,
}

struct ModelIdentity {
    key_prefix: String,
    family: ModelFamily,
}

fn detect_model_identity(gguf: &GGUFFile, debug_mode: bool) -> ModelIdentity {
    let arch = get_gguf_string_from_map(&gguf.kv, "general.architecture").unwrap_or("llama");
    if debug_mode {
        eprintln!("Model architecture: {arch}");
    }

    let mut identity = ModelIdentity {
        key_prefix: "llama".to_string(),
        family: ModelFamily::Llama,
    };

    if arch == "gemma3" || arch == "gemma2" || arch == "gemma" {
        identity.family = ModelFamily::Gemma;
        identity.key_prefix = "gemma3".to_string();
        if get_gguf_int_from_map(&gguf.kv, "gemma3.embedding_length", 0) == 0 {
            if get_gguf_int_from_map(&gguf.kv, "gemma2.embedding_length", 0) != 0 {
                identity.key_prefix = "gemma2".to_string();
            } else if get_gguf_int_from_map(&gguf.kv, "gemma.embedding_length", 0) != 0 {
                identity.key_prefix = "gemma".to_string();
            }
        }
        if debug_mode {
            eprintln!(
                "Detected Gemma architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    } else if arch == "qwen3moe" || arch.starts_with("qwen3moe") {
        identity.family = ModelFamily::Qwen3Moe;
        identity.key_prefix = arch.to_string();
        let probe = format!("{}.embedding_length", identity.key_prefix);
        if get_gguf_int_from_map(&gguf.kv, &probe, 0) == 0
            && get_gguf_int_from_map(&gguf.kv, "qwen3moe.embedding_length", 0) != 0
        {
            identity.key_prefix = "qwen3moe".to_string();
        }
        if debug_mode {
            eprintln!(
                "Detected Qwen3 MoE architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    } else if arch == "qwen3next" || arch.starts_with("qwen3next") {
        identity.family = ModelFamily::Qwen3Next;
        identity.key_prefix = arch.to_string();
        let probe = format!("{}.embedding_length", identity.key_prefix);
        if get_gguf_int_from_map(&gguf.kv, &probe, 0) == 0
            && get_gguf_int_from_map(&gguf.kv, "qwen3next.embedding_length", 0) != 0
        {
            identity.key_prefix = "qwen3next".to_string();
        }
        if debug_mode {
            eprintln!(
                "Detected Qwen3 Next architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    } else if arch.starts_with("qwen") || arch == "qwen2" {
        identity.family = ModelFamily::Qwen2;
        identity.key_prefix = arch.to_string();
        let probe = format!("{}.embedding_length", identity.key_prefix);
        if get_gguf_int_from_map(&gguf.kv, &probe, 0) == 0 {
            if get_gguf_int_from_map(&gguf.kv, "qwen2.embedding_length", 0) != 0 {
                identity.key_prefix = "qwen2".to_string();
            } else if get_gguf_int_from_map(&gguf.kv, "qwen.embedding_length", 0) != 0 {
                identity.key_prefix = "qwen".to_string();
            }
        }
        if debug_mode {
            eprintln!(
                "Detected Qwen architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    } else if arch == "deepseek2" || arch.starts_with("deepseek") {
        identity.family = ModelFamily::DeepSeek2;
        identity.key_prefix = arch.to_string();
        let probe = format!("{}.embedding_length", identity.key_prefix);
        if get_gguf_int_from_map(&gguf.kv, &probe, 0) == 0
            && get_gguf_int_from_map(&gguf.kv, "deepseek2.embedding_length", 0) != 0
        {
            identity.key_prefix = "deepseek2".to_string();
        }
        if debug_mode {
            eprintln!(
                "Detected DeepSeek architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    }

    identity
}

pub(crate) fn build_config_from_gguf(gguf: &GGUFFile, debug_mode: bool) -> Result<Config, String> {
    let identity = detect_model_identity(gguf, debug_mode);
    let key_prefix = identity.key_prefix;

    let key_dim = format!("{key_prefix}.embedding_length");
    let key_hidden = format!("{key_prefix}.feed_forward_length");
    let key_layers = format!("{key_prefix}.block_count");
    let key_heads = format!("{key_prefix}.attention.head_count");
    let key_kv_heads = format!("{key_prefix}.attention.head_count_kv");
    let key_vocab = format!("{key_prefix}.vocab_size");
    let key_ctx = format!("{key_prefix}.context_length");
    let key_rope = format!("{key_prefix}.rope.freq_base");
    let key_rope_scaling_factor = format!("{key_prefix}.rope.scaling.factor");
    let key_rope_scaling_type = format!("{key_prefix}.rope.scaling.type");
    let key_rope_scaling_orig_ctx = format!("{key_prefix}.rope.scaling.original_context_length");
    let key_rope_scaling_yarn_log_mul = format!("{key_prefix}.rope.scaling.yarn_log_multiplier");
    let key_rope_dim = format!("{key_prefix}.rope.dimension_count");
    let key_head_dim = format!("{key_prefix}.attention.key_length");
    let key_rms_eps = format!("{key_prefix}.attention.layer_norm_rms_epsilon");
    let key_softcap = format!("{key_prefix}.final_logit_softcapping");
    let key_rope_swa = format!("{key_prefix}.rope.freq_base_swa");
    let key_expert_count = format!("{key_prefix}.expert_count");
    let key_expert_used_count = format!("{key_prefix}.expert_used_count");
    let key_expert_ffn = format!("{key_prefix}.expert_feed_forward_length");
    let key_expert_shared_ffn = format!("{key_prefix}.expert_shared_feed_forward_length");
    let key_expert_gating_func = format!("{key_prefix}.expert_gating_func");
    let key_ssm_conv_kernel = format!("{key_prefix}.ssm.conv_kernel");
    let key_ssm_inner_size = format!("{key_prefix}.ssm.inner_size");
    let key_ssm_state_size = format!("{key_prefix}.ssm.state_size");
    let key_ssm_time_step_rank = format!("{key_prefix}.ssm.time_step_rank");
    let key_ssm_group_count = format!("{key_prefix}.ssm.group_count");
    let key_q_lora_rank = format!("{key_prefix}.attention.q_lora_rank");
    let key_kv_lora_rank = format!("{key_prefix}.attention.kv_lora_rank");
    let key_key_mla = format!("{key_prefix}.attention.key_length_mla");
    let key_value_mla = format!("{key_prefix}.attention.value_length_mla");
    let key_leading_dense = format!("{key_prefix}.leading_dense_block_count");
    let key_expert_weights_scale = format!("{key_prefix}.expert_weights_scale");

    let mut config = Config {
        dim: get_gguf_int_from_map(&gguf.kv, &key_dim, 4096) as usize,
        hidden_dim: get_gguf_int_from_map(&gguf.kv, &key_hidden, 11008) as usize,
        expert_hidden_dim: get_gguf_int_from_map(&gguf.kv, &key_expert_ffn, 0) as usize,
        shared_expert_hidden_dim: get_gguf_int_from_map(&gguf.kv, &key_expert_shared_ffn, 0)
            as usize,
        n_layers: get_gguf_int_from_map(&gguf.kv, &key_layers, 32) as usize,
        n_heads: get_gguf_int_from_map(&gguf.kv, &key_heads, 32) as usize,
        n_kv_heads: 0,
        n_experts: get_gguf_int_from_map(&gguf.kv, &key_expert_count, 0) as usize,
        n_experts_used: get_gguf_int_from_map(&gguf.kv, &key_expert_used_count, 0) as usize,
        moe_n_group: 1,
        moe_topk_group: 1,
        moe_norm_topk_prob: false,
        moe_routed_scaling_factor: 1.0,
        vocab_size: get_gguf_int_from_map(&gguf.kv, &key_vocab, 32000) as usize,
        seq_len: get_gguf_int_from_map(&gguf.kv, &key_ctx, 2048) as usize,
        rope_theta: 0.0,
        rope_freq_scale: 1.0,
        head_dim: 0,
        rope_dim: 0,
        rope_scaling_yarn: false,
        rope_orig_ctx: 0,
        rope_yarn_log_multiplier: 0.0,
        is_gemma3: identity.family == ModelFamily::Gemma,
        is_qwen2: identity.family == ModelFamily::Qwen2,
        is_qwen3moe: identity.family == ModelFamily::Qwen3Moe,
        is_qwen3next: identity.family == ModelFamily::Qwen3Next,
        is_deepseek2: identity.family == ModelFamily::DeepSeek2,
        final_logit_softcapping: get_gguf_float_from_map(&gguf.kv, &key_softcap, 0.0),
        rms_norm_eps: get_gguf_float_from_map(&gguf.kv, &key_rms_eps, 1e-6),
        rope_theta_swa: get_gguf_float_from_map(&gguf.kv, &key_rope_swa, 10_000.0),
        swa_pattern: 6,
        ssm_conv_kernel: get_gguf_int_from_map(&gguf.kv, &key_ssm_conv_kernel, 0) as usize,
        ssm_inner_size: get_gguf_int_from_map(&gguf.kv, &key_ssm_inner_size, 0) as usize,
        ssm_state_size: get_gguf_int_from_map(&gguf.kv, &key_ssm_state_size, 0) as usize,
        ssm_time_step_rank: get_gguf_int_from_map(&gguf.kv, &key_ssm_time_step_rank, 0) as usize,
        ssm_group_count: get_gguf_int_from_map(&gguf.kv, &key_ssm_group_count, 0) as usize,
        deepseek_leading_dense_block_count: get_gguf_int_from_map(&gguf.kv, &key_leading_dense, 0)
            as usize,
        deepseek_q_lora_rank: get_gguf_int_from_map(&gguf.kv, &key_q_lora_rank, 0) as usize,
        deepseek_kv_lora_rank: get_gguf_int_from_map(&gguf.kv, &key_kv_lora_rank, 0) as usize,
        deepseek_qk_nope_head_dim: 0,
        deepseek_qk_rope_head_dim: 0,
        deepseek_v_head_dim: get_gguf_int_from_map(&gguf.kv, &key_value_mla, 0) as usize,
        deepseek_expert_gating_func: get_gguf_int_from_map(&gguf.kv, &key_expert_gating_func, 0)
            as i32,
    };

    if config.is_qwen3moe || config.is_qwen3next || config.is_deepseek2 {
        qwen::finalize_moe_config(&mut config)?;
        if config.is_qwen3moe {
            qwen::apply_qwen3moe_defaults(&mut config);
        }
        if config.is_qwen3next {
            qwen::validate_qwen3next(&mut config)?;
        }
        if config.is_deepseek2 {
            config.moe_norm_topk_prob = true;
            config.moe_routed_scaling_factor =
                get_gguf_float_from_map(&gguf.kv, &key_expert_weights_scale, 1.0);
            if config.shared_expert_hidden_dim == 0 {
                config.shared_expert_hidden_dim = config.expert_hidden_dim;
            }
        }
    }

    config.n_kv_heads =
        get_gguf_int_from_map(&gguf.kv, &key_kv_heads, config.n_heads as i64) as usize;

    let default_rope_theta = if identity.family == ModelFamily::Gemma {
        gemma::default_rope_theta()
    } else {
        llama::default_rope_theta()
    };
    let rope_scaling_factor = get_gguf_float_from_map(&gguf.kv, &key_rope_scaling_factor, 0.0);
    let rope_scaling_type =
        get_gguf_string_from_map(&gguf.kv, &key_rope_scaling_type).unwrap_or_default();
    config.rope_theta = get_gguf_float_from_map(&gguf.kv, &key_rope, default_rope_theta);
    config.rope_freq_scale = if rope_scaling_factor > 0.0 {
        1.0 / rope_scaling_factor
    } else {
        1.0
    };
    config.rope_scaling_yarn = rope_scaling_type.eq_ignore_ascii_case("yarn");
    config.rope_orig_ctx =
        get_gguf_int_from_map(&gguf.kv, &key_rope_scaling_orig_ctx, config.seq_len as i64) as usize;
    config.rope_yarn_log_multiplier =
        get_gguf_float_from_map(&gguf.kv, &key_rope_scaling_yarn_log_mul, 0.0);
    if config.rope_yarn_log_multiplier != 0.0 {
        // DeepSeek exports frequently store this value scaled by 0.1.
        config.rope_yarn_log_multiplier /= 0.1;
    }
    config.head_dim = get_gguf_int_from_map(
        &gguf.kv,
        &key_head_dim,
        (config.dim / config.n_heads) as i64,
    ) as usize;
    config.rope_dim =
        get_gguf_int_from_map(&gguf.kv, &key_rope_dim, config.head_dim as i64) as usize;
    if config.rope_dim == 0 || config.rope_dim > config.head_dim || (config.rope_dim & 1) != 0 {
        config.rope_dim = config.head_dim;
    }

    if config.is_deepseek2 {
        let mla_key_dim = get_gguf_int_from_map(&gguf.kv, &key_key_mla, 0) as usize;
        if mla_key_dim == 0 {
            return Err(format!(
                "deepseek2 model is missing {key_key_mla}; cannot derive MLA key head dimension"
            ));
        }
        if config.rope_dim == 0 || config.rope_dim >= mla_key_dim {
            return Err(format!(
                "deepseek2 invalid rope dimension {} for MLA key head dimension {}",
                config.rope_dim, mla_key_dim
            ));
        }
        config.head_dim = mla_key_dim;
        config.deepseek_qk_rope_head_dim = config.rope_dim;
        config.deepseek_qk_nope_head_dim = mla_key_dim - config.rope_dim;
        deepseek::validate_deepseek2(&mut config)?;
    }

    if !gguf.vocab_tokens.is_empty() && config.vocab_size != gguf.vocab_tokens.len() {
        if debug_mode {
            eprintln!(
                "Note: Updating vocab_size from {} to {} based on GGUF vocabulary",
                config.vocab_size,
                gguf.vocab_tokens.len()
            );
        }
        config.vocab_size = gguf.vocab_tokens.len();
    }

    if debug_mode {
        eprintln!(
            "Config: dim={}, hidden_dim={}, expert_hidden_dim={}, n_layers={}, n_heads={}, n_kv_heads={}, vocab_size={}, seq_len={}",
            config.dim,
            config.hidden_dim,
            config.expert_hidden_dim,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.vocab_size,
            config.seq_len
        );
        eprintln!(
            "RoPE theta: {}, freq_scale: {}, head_dim: {}, rope_dim: {}, scaling_yarn: {}, rope_orig_ctx: {}, yarn_log_multiplier: {}",
            config.rope_theta,
            config.rope_freq_scale,
            config.head_dim,
            config.rope_dim,
            config.rope_scaling_yarn,
            config.rope_orig_ctx,
            config.rope_yarn_log_multiplier
        );
        if config.is_gemma3 {
            gemma::print_config_debug(&config);
        } else if config.is_qwen3moe {
            qwen::print_qwen3moe_debug(&config);
        } else if config.is_qwen3next {
            qwen::print_qwen3next_debug(&config);
        } else if config.is_deepseek2 {
            deepseek::print_deepseek2_debug(&config);
        }
    }

    Ok(config)
}

pub(crate) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    config: &Config,
    prompt: &str,
    system_prompt: &str,
) -> Vec<i32> {
    if config.is_gemma3 {
        gemma::encode_chat_prompt(tokenizer, prompt, system_prompt)
    } else if config.is_deepseek2 {
        deepseek::encode_deepseek_chat(tokenizer, prompt, system_prompt)
    } else if config.is_qwen3moe || config.is_qwen3next {
        qwen::encode_qwen3_chat(tokenizer, prompt, system_prompt)
    } else if config.is_qwen2 {
        qwen::encode_qwen2_chat(tokenizer, prompt, system_prompt)
    } else {
        llama::encode_chat_prompt(tokenizer, prompt, system_prompt)
    }
}
