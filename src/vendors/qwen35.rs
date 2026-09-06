// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use super::{
    ChatMessage, VendorDecodePolicy, VendorDetailCropPolicy, VendorMultimodalPolicy,
    VendorRuntimeDebugPolicy, VendorTokenizerPolicy, qwen_common,
};
use crate::engine::types::{Config, EncodedPrompt, GenerationRequest, ThinkMode, Tokenizer};

fn qwen35_detail_crop_enabled() -> bool {
    matches!(
        std::env::var("GGUF_QWEN35_DETAIL_CROP"),
        Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
    )
}

fn weak_agent_model(config: &Config) -> bool {
    config.n_experts == 0 && (config.dim <= 1024 || config.n_layers <= 24)
}

pub(super) fn decode_policy(config: &Config) -> VendorDecodePolicy {
    let weak_agent_model = weak_agent_model(config);
    VendorDecodePolicy {
        parse_think_tags: true,
        stop_token_literals: qwen_common::QWEN_STOP_TOKEN_LITERALS,
        stop_text_literals: qwen_common::QWEN_STOP_TEXT_LITERALS,
        deterministic_loop_guard: true,
        deterministic_loop_guard_min_generated_tokens: 96,
        unconditional_loop_guard: true,
        recover_early_endoftext_once: false,
        early_endoftext_recover_max_tokens: 0,
        hidden_think_token_cap_base: 384,
        visible_think_token_cap_base: 192,
        prefer_hidden_think_for_multimodal: true,
        retry_without_think_when_no_post_think_text: true,
        agent_force_deterministic: config.n_experts > 0 || weak_agent_model,
        agent_protocol_max_failures: if weak_agent_model { 1 } else { 3 },
        agent_plain_chat_fallback_after_protocol_failures: weak_agent_model,
    }
}

pub(super) fn tokenizer_policy() -> VendorTokenizerPolicy {
    VendorTokenizerPolicy {
        disable_bos_fallback: true,
        end_turn_token_literals: qwen_common::QWEN_END_TURN_TOKEN_LITERALS,
    }
}

pub(super) fn multimodal_policy() -> VendorMultimodalPolicy {
    VendorMultimodalPolicy {
        image_prompt_suffix: "\nPlease avoid guessing uncertain details. If text is unclear, explicitly say it is unreadable.",
        detail_crop: VendorDetailCropPolicy {
            enabled: qwen35_detail_crop_enabled(),
            max_layers: 24,
            note_text: "\n(Second image: centered close-up crop of the same source.)\n",
            temp_file_prefix: "gguf-runner-qwen35-detail",
        },
        mmproj_filename_score_hints: qwen_common::QWEN_MMPROJ_SCORE_HINTS,
        missing_sidecar_hint: " hint: Qwen3.5 image/video inputs require a compatible Qwen3.5 mmproj sidecar from the same checkpoint family.",
    }
}

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    qwen_common::runtime_debug_policy()
}

pub(super) fn configure_rope(config: &mut Config) -> Result<(), String> {
    qwen_common::configure_interleaved_rope(config, [11, 11, 10, 0])
}

pub(super) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
    image_count: usize,
    think_mode: ThinkMode,
) -> Vec<i32> {
    qwen_common::encode_qwen3_chat(tokenizer, prompt, system_prompt, image_count, think_mode)
}

pub(super) fn encode_chat_messages(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
    think_mode: ThinkMode,
) -> Vec<i32> {
    qwen_common::encode_qwen3_messages(tokenizer, messages, system_prompt, think_mode)
}

pub(super) fn encode_generation_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
    think_mode: ThinkMode,
) -> EncodedPrompt {
    qwen_common::encode_qwen3_request(tokenizer, request, think_mode)
}

#[cfg(test)]
mod tests {
    use super::decode_policy;
    use crate::engine::types::{Config, ModelCapabilities};

    fn base_qwen35_config() -> Config {
        Config {
            dim: 2048,
            input_embedding_dim: 2048,
            n_deepstack_layers: 0,
            hidden_dim: 0,
            expert_hidden_dim: 0,
            shared_expert_hidden_dim: 0,
            n_layers: 40,
            n_heads: 0,
            n_kv_heads: 0,
            n_experts: 0,
            n_experts_used: 0,
            moe_n_group: 0,
            moe_topk_group: 0,
            moe_norm_topk_prob: false,
            moe_routed_scaling_factor: 0.0,
            vocab_size: 0,
            seq_len: 0,
            rope_theta: 0.0,
            head_dim: 0,
            rope_dim: 0,
            rope_sections: [0; 4],
            rope_position_layout: Default::default(),
            is_bert_family: false,
            is_gemma3: false,
            is_qwen2: false,
            is_qwen35: true,
            is_qwen3vl: false,
            is_qwen3moe: false,
            is_qwen3next: false,
            is_qwen3: false,
            is_smolvlm: false,
            uses_chatml_template: false,
            online_attn_fusion: false,
            qwen_chat_template_contains_think: true,
            qwen_chat_template_has_builtin_system: false,
            qwen_chat_template_uses_empty_think: false,
            capabilities: ModelCapabilities::default(),
            final_logit_softcapping: 0.0,
            rms_norm_eps: 0.0,
            rope_theta_swa: 0.0,
            swa_pattern: 0,
            ssm_conv_kernel: 0,
            ssm_inner_size: 0,
            ssm_state_size: 0,
            ssm_time_step_rank: 0,
            ssm_group_count: 0,
        }
    }

    #[test]
    fn interleaved_rope_validates_vendor_defaults_and_explicit_sections() {
        use crate::engine::types::RopePositionLayout;
        let mut config = base_qwen35_config();
        config.head_dim = 256;
        config.rope_dim = 64;
        super::configure_rope(&mut config).unwrap();
        assert_eq!(config.rope_sections, [11, 11, 10, 0]);
        assert_eq!(config.rope_position_layout, RopePositionLayout::Interleaved);
        super::configure_rope(&mut config).unwrap();
        for sections in [
            [10, 11, 10, 0],
            [1, 1, 30, 0],
            [10, 11, 10, 1],
            [usize::MAX; 4],
        ] {
            config.rope_sections = sections;
            assert!(super::configure_rope(&mut config).is_err());
        }
        config.rope_dim = 128;
        config.rope_sections = [0; 4];
        crate::vendors::qwen3vl::configure_rope(&mut config).unwrap();
        assert_eq!(config.rope_sections, [24, 20, 20, 0]);
    }

    #[test]
    fn weak_qwen35_agent_policy_uses_low_retry_budget() {
        let mut config = base_qwen35_config();
        config.dim = 1024;
        config.n_layers = 24;
        let policy = decode_policy(&config);
        assert!(policy.agent_force_deterministic);
        assert_eq!(policy.agent_protocol_max_failures, 1);
        assert!(policy.agent_plain_chat_fallback_after_protocol_failures);
    }

    #[test]
    fn larger_qwen35_agent_policy_keeps_standard_retry_budget() {
        let config = base_qwen35_config();
        let policy = decode_policy(&config);
        assert!(!policy.agent_plain_chat_fallback_after_protocol_failures);
        assert_eq!(policy.agent_protocol_max_failures, 3);
    }
}
