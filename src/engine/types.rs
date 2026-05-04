use std::collections::HashMap;
#[cfg(unix)]
use std::ffi::c_void;
use std::fs::File;
use std::io;
#[cfg(not(unix))]
use std::io::Read;
#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};
#[cfg(unix)]
use std::os::fd::AsRawFd;

pub(crate) const GGUF_MAGIC: u32 = 0x4655_4747;

pub(crate) const GGUF_TYPE_UINT8: u32 = 0;
pub(crate) const GGUF_TYPE_INT8: u32 = 1;
pub(crate) const GGUF_TYPE_UINT16: u32 = 2;
pub(crate) const GGUF_TYPE_INT16: u32 = 3;
pub(crate) const GGUF_TYPE_UINT32: u32 = 4;
pub(crate) const GGUF_TYPE_INT32: u32 = 5;
pub(crate) const GGUF_TYPE_FLOAT32: u32 = 6;
pub(crate) const GGUF_TYPE_BOOL: u32 = 7;
pub(crate) const GGUF_TYPE_STRING: u32 = 8;
pub(crate) const GGUF_TYPE_ARRAY: u32 = 9;
pub(crate) const GGUF_TYPE_UINT64: u32 = 10;
pub(crate) const GGUF_TYPE_INT64: u32 = 11;
pub(crate) const GGUF_TYPE_FLOAT64: u32 = 12;

pub(crate) const QK4_0: usize = 32;
pub(crate) const QK4_1: usize = 32;
pub(crate) const QK5_0: usize = 32;
pub(crate) const QK5_1: usize = 32;
pub(crate) const QK8_0: usize = 32;
pub(crate) const QK_K: usize = 256;
pub(crate) const QK4_NL: usize = 32;

pub(crate) const GGML_TYPE_F32: i32 = 0;
pub(crate) const GGML_TYPE_F16: i32 = 1;
pub(crate) const GGML_TYPE_Q4_0: i32 = 2;
pub(crate) const GGML_TYPE_Q4_1: i32 = 3;
pub(crate) const GGML_TYPE_Q5_0: i32 = 6;
pub(crate) const GGML_TYPE_Q5_1: i32 = 7;
pub(crate) const GGML_TYPE_Q8_0: i32 = 8;
pub(crate) const GGML_TYPE_Q2_K: i32 = 10;
pub(crate) const GGML_TYPE_Q3_K: i32 = 11;
pub(crate) const GGML_TYPE_Q4_K: i32 = 12;
pub(crate) const GGML_TYPE_Q5_K: i32 = 13;
pub(crate) const GGML_TYPE_Q6_K: i32 = 14;
pub(crate) const GGML_TYPE_IQ4_NL: i32 = 20;
pub(crate) const GGML_TYPE_BF16: i32 = 30;

pub(crate) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub(crate) const LLAMA3_BOS_TOKEN: i32 = 128000;
pub(crate) const LLAMA3_EOS_TOKEN: i32 = 128001;
pub(crate) const LLAMA3_START_HEADER: i32 = 128006;
pub(crate) const LLAMA3_END_HEADER: i32 = 128007;
pub(crate) const LLAMA3_EOT: i32 = 128009;

pub(crate) const GEMMA3_BOS_TOKEN: i32 = 2;
pub(crate) const GEMMA3_START_TURN: i32 = 106;
pub(crate) const GEMMA3_END_TURN: i32 = 107;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum ThinkMode {
    #[default]
    Yes,
    No,
    Hidden,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VendorTokenizerPolicy {
    pub(crate) disable_bos_fallback: bool,
    pub(crate) end_turn_token_literals: &'static [&'static str],
}

impl Default for VendorTokenizerPolicy {
    fn default() -> Self {
        Self {
            disable_bos_fallback: false,
            end_turn_token_literals: &["<|eot_id|>"],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MediaRef {
    pub(crate) path: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ContentPart {
    Text(String),
    Image(MediaRef),
    Video(MediaRef),
    Audio(MediaRef),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GenerationRequest {
    pub(crate) system_prompt: String,
    pub(crate) parts: Vec<ContentPart>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaceholderSpan {
    pub(crate) token_start: usize,
    pub(crate) token_len: usize,
    pub(crate) media_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EncodedPrompt {
    pub(crate) token_ids: Vec<i32>,
    pub(crate) image_spans: Vec<PlaceholderSpan>,
    pub(crate) video_spans: Vec<PlaceholderSpan>,
    pub(crate) audio_spans: Vec<PlaceholderSpan>,
}

impl EncodedPrompt {
    pub(crate) fn from_token_ids(token_ids: Vec<i32>) -> Self {
        Self {
            token_ids,
            image_spans: Vec::new(),
            video_spans: Vec::new(),
            audio_spans: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MultimodalBackend {
    None,
    Gemma3,
    Qwen3Vl,
    Qwen35,
}

impl MultimodalBackend {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            MultimodalBackend::None => "none",
            MultimodalBackend::Gemma3 => "gemma3",
            MultimodalBackend::Qwen3Vl => "qwen3vl",
            MultimodalBackend::Qwen35 => "qwen35",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModelCapabilities {
    pub(crate) multimodal_backend: MultimodalBackend,
    pub(crate) supports_native_image: bool,
    pub(crate) supports_native_video: bool,
    pub(crate) supports_native_audio: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            multimodal_backend: MultimodalBackend::None,
            supports_native_image: false,
            supports_native_video: false,
            supports_native_audio: false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MultimodalWeights {
    pub(crate) backend: MultimodalBackend,
    pub(crate) vision_tensor_names: Vec<String>,
    pub(crate) projector_tensor_names: Vec<String>,
    pub(crate) audio_tensor_names: Vec<String>,
}

impl MultimodalWeights {
    #[inline]
    pub(crate) fn total_tensor_count(&self) -> usize {
        self.vision_tensor_names.len()
            + self.projector_tensor_names.len()
            + self.audio_tensor_names.len()
    }
}

#[cfg(unix)]
pub(crate) const PROT_READ: i32 = 0x1;
#[cfg(unix)]
pub(crate) const MAP_SHARED: i32 = 0x0001;
#[cfg(target_os = "linux")]
const MADV_WILLNEED: i32 = 3;
#[cfg(target_os = "linux")]
const MADV_HUGEPAGE: i32 = 14;

#[cfg(unix)]
unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GgmlType(pub(crate) i32);

impl GgmlType {
    pub(crate) fn from_u32(v: u32) -> Self {
        Self(v as i32)
    }
}

impl Default for GgmlType {
    fn default() -> Self {
        Self(GGML_TYPE_F32)
    }
}

pub(crate) struct MappedFile {
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
    #[cfg(not(unix))]
    #[allow(dead_code)]
    backing: Box<[u8]>,
}

impl MappedFile {
    #[cfg(target_os = "linux")]
    #[inline]
    fn apply_linux_mmap_advice(ptr: *mut c_void, len: usize) {
        unsafe {
            let _ = madvise(ptr, len, MADV_WILLNEED);
            let _ = madvise(ptr, len, MADV_HUGEPAGE);
        }
    }

    #[cfg(unix)]
    pub(crate) fn map(file: &File) -> io::Result<Self> {
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot mmap empty file",
            ));
        }
        let fd = file.as_raw_fd();
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_SHARED, fd, 0) };
        if ptr as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        #[cfg(target_os = "linux")]
        Self::apply_linux_mmap_advice(ptr, len);
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    #[cfg(not(unix))]
    pub(crate) fn map(file: &File) -> io::Result<Self> {
        let mut reader = file.try_clone()?;
        reader.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        if bytes.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot map empty file",
            ));
        }

        let mut backing = bytes.into_boxed_slice();
        let ptr = backing.as_mut_ptr();
        let len = backing.len();
        Ok(Self { ptr, len, backing })
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

// SAFETY: the mapped region is only read (never written) during inference;
// `ptr` is valid for the lifetime of the struct and immutable across threads.
unsafe impl Sync for MappedFile {}

impl Drop for MappedFile {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            let _ = munmap(self.ptr as *mut c_void, self.len);
        }
    }
}

pub(crate) fn ensure_model_range(offset: usize, len: usize) -> Result<(), String> {
    let _ = offset;
    let _ = len;
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) enum GgufValue {
    UInt(u64),
    Int(i64),
    F32(f32),
    F64(f64),
    F32Array(Vec<f32>),
    I64Array(Vec<i64>),
    Bool(bool),
    Str(String),
}

#[derive(Clone, Debug)]
pub(crate) struct Gguftensor {
    pub(crate) name: String,
    pub(crate) n_dims: u32,
    pub(crate) ne: [u64; 4],
    pub(crate) ttype: GgmlType,
    pub(crate) offset: u64,
    pub(crate) data_offset: usize,
}

pub(crate) struct GGUFFile {
    pub(crate) version: u32,
    pub(crate) n_tensors: u64,
    pub(crate) n_kv: u64,
    pub(crate) kv: HashMap<String, GgufValue>,
    pub(crate) tensors: Vec<Gguftensor>,
    pub(crate) tensor_lookup: HashMap<String, usize>,
    pub(crate) tensor_data_start: usize,
    pub(crate) vocab_tokens: Vec<String>,
    pub(crate) vocab_scores: Vec<f32>,
    pub(crate) vocab_merges: Vec<String>,
    pub(crate) mapped: MappedFile,
}

impl GGUFFile {
    #[inline]
    pub(crate) fn ensure_range(&self, _offset: usize, _len: usize) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct Config {
    pub(crate) dim: usize,
    pub(crate) input_embedding_dim: usize,
    pub(crate) n_deepstack_layers: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) expert_hidden_dim: usize,
    pub(crate) shared_expert_hidden_dim: usize,
    pub(crate) n_layers: usize,
    pub(crate) n_heads: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) n_experts: usize,
    pub(crate) n_experts_used: usize,
    pub(crate) moe_n_group: usize,
    pub(crate) moe_topk_group: usize,
    pub(crate) moe_norm_topk_prob: bool,
    pub(crate) moe_routed_scaling_factor: f32,
    pub(crate) vocab_size: usize,
    pub(crate) seq_len: usize,
    pub(crate) rope_theta: f32,
    pub(crate) head_dim: usize,
    pub(crate) rope_dim: usize,
    pub(crate) rope_sections: [usize; 4],
    pub(crate) is_bert_family: bool,
    pub(crate) is_gemma3: bool,
    pub(crate) is_qwen2: bool,
    pub(crate) is_qwen35: bool,
    pub(crate) is_qwen3vl: bool,
    pub(crate) is_qwen3moe: bool,
    pub(crate) is_qwen3next: bool,
    pub(crate) online_attn_fusion: bool,
    pub(crate) qwen_chat_template_contains_think: bool,
    pub(crate) qwen_chat_template_has_builtin_system: bool,
    pub(crate) capabilities: ModelCapabilities,
    pub(crate) final_logit_softcapping: f32,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta_swa: f32,
    pub(crate) swa_pattern: usize,
    pub(crate) ssm_conv_kernel: usize,
    pub(crate) ssm_inner_size: usize,
    pub(crate) ssm_state_size: usize,
    pub(crate) ssm_time_step_rank: usize,
    pub(crate) ssm_group_count: usize,
}

#[derive(Clone, Default)]
pub(crate) struct QuantizedTensor {
    pub(crate) data_offset: usize,
    pub(crate) ttype: GgmlType,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

pub(crate) struct TransformerWeights {
    pub(crate) token_embedding_table: Vec<f32>,
    pub(crate) rms_att_weight: Vec<f32>,
    pub(crate) rms_ffn_weight: Vec<f32>,
    pub(crate) wq: Vec<QuantizedTensor>,
    pub(crate) wk: Vec<QuantizedTensor>,
    pub(crate) wv: Vec<QuantizedTensor>,
    pub(crate) wo: Vec<QuantizedTensor>,
    pub(crate) w1: Vec<QuantizedTensor>,
    pub(crate) w2: Vec<QuantizedTensor>,
    pub(crate) w3: Vec<QuantizedTensor>,
    pub(crate) attn_qkv: Vec<QuantizedTensor>,
    pub(crate) ssm_ba: Vec<QuantizedTensor>,
    pub(crate) ssm_alpha: Vec<QuantizedTensor>,
    pub(crate) ssm_beta: Vec<QuantizedTensor>,
    pub(crate) ssm_conv1d: Vec<Vec<f32>>,
    pub(crate) ssm_a: Vec<f32>,
    pub(crate) ssm_dt_bias: Vec<f32>,
    pub(crate) ssm_norm: Vec<f32>,
    pub(crate) moe_gate_inp: Vec<QuantizedTensor>,
    pub(crate) moe_gate_exps: Vec<QuantizedTensor>,
    pub(crate) moe_up_exps: Vec<QuantizedTensor>,
    pub(crate) moe_down_exps: Vec<QuantizedTensor>,
    pub(crate) local_moe_experts: Option<WorkerExpertWeights>,
    pub(crate) moe_shared_gate_inp: Vec<f32>,
    pub(crate) rms_final_weight: Vec<f32>,
    pub(crate) wcls: QuantizedTensor,
    pub(crate) wcls_is_embed: bool,
    pub(crate) attn_q_bias: Vec<f32>,
    pub(crate) attn_k_bias: Vec<f32>,
    pub(crate) attn_v_bias: Vec<f32>,
    pub(crate) attn_q_norm: Vec<f32>,
    pub(crate) attn_k_norm: Vec<f32>,
    pub(crate) attn_qk_norm_present: Vec<bool>,
    pub(crate) attn_post_norm: Vec<f32>,
    pub(crate) ffn_post_norm: Vec<f32>,
    /// Bias vectors for BERT-family post-LayerNorm (empty for non-BERT models).
    pub(crate) attn_post_norm_bias: Vec<f32>,
    pub(crate) ffn_post_norm_bias: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct WorkerExpertTensors {
    pub(crate) gate: QuantizedTensor,
    pub(crate) up: QuantizedTensor,
    pub(crate) down: QuantizedTensor,
}

pub(crate) struct WorkerExpertWeights {
    pub(crate) experts: Vec<Vec<Option<WorkerExpertTensors>>>,
}

pub(crate) struct WorkerSsmLayerWeights {
    pub(crate) qkv_in: QuantizedTensor,
    pub(crate) gate: QuantizedTensor,
    pub(crate) out_proj: QuantizedTensor,
    pub(crate) ba: Option<QuantizedTensor>,
    pub(crate) alpha: Option<QuantizedTensor>,
    pub(crate) beta: Option<QuantizedTensor>,
    pub(crate) conv1d: Vec<f32>,
    pub(crate) a: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) norm: Vec<f32>,
}

pub(crate) struct WorkerSsmSession {
    pub(crate) layers: Vec<Option<WorkerSsmLayerWeights>>,
    pub(crate) conv_state: Vec<f32>,
    pub(crate) ssm_state: Vec<f32>,
    pub(crate) dim: usize,
    pub(crate) d_inner: usize,
    pub(crate) n_k_heads: usize,
    pub(crate) n_v_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) n_layers: usize,
    pub(crate) eps: f32,
    // scratch buffers (pre-allocated):
    pub(crate) ssm_qkv: Vec<f32>,
    pub(crate) ssm_z: Vec<f32>,
    pub(crate) ssm_ba_scratch: Vec<f32>,
    pub(crate) ssm_gate_exp: Vec<f32>,
    pub(crate) ssm_beta_scratch: Vec<f32>,
    pub(crate) ssm_conv: Vec<f32>,
    pub(crate) ssm_q: Vec<f32>,
    pub(crate) ssm_k: Vec<f32>,
    pub(crate) ssm_v: Vec<f32>,
    pub(crate) ssm_proj: Vec<f32>,
    pub(crate) ssm_kv_mem: Vec<f32>,
    pub(crate) ssm_delta: Vec<f32>,
}

pub(crate) struct RunState {
    pub(crate) x: Vec<f32>,
    pub(crate) xb: Vec<f32>,
    pub(crate) xb2: Vec<f32>,
    pub(crate) hb: Vec<f32>,
    pub(crate) hb2: Vec<f32>,
    pub(crate) moe_tmp: Vec<f32>,
    pub(crate) moe_logits: Vec<f32>,
    pub(crate) moe_topk_indices: Vec<usize>,
    pub(crate) moe_topk_weights: Vec<f32>,
    pub(crate) moe_scores: Vec<f32>,
    pub(crate) moe_selected_group: Vec<bool>,
    pub(crate) moe_group_scores: Vec<f32>,
    pub(crate) moe_group_rank: Vec<usize>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) ssm_qkv: Vec<f32>,
    pub(crate) ssm_conv: Vec<f32>,
    pub(crate) ssm_q: Vec<f32>,
    pub(crate) ssm_k: Vec<f32>,
    pub(crate) ssm_v: Vec<f32>,
    pub(crate) ssm_z: Vec<f32>,
    pub(crate) ssm_ba: Vec<f32>,
    pub(crate) ssm_gate_exp: Vec<f32>,
    pub(crate) ssm_beta: Vec<f32>,
    pub(crate) ssm_proj: Vec<f32>,
    pub(crate) ssm_kv_mem: Vec<f32>,
    pub(crate) ssm_delta: Vec<f32>,
    pub(crate) ssm_conv_state: Vec<f32>,
    pub(crate) ssm_state: Vec<f32>,
    pub(crate) att: Vec<f32>,
    pub(crate) logits: Vec<f32>,
    pub(crate) kv_cache_format: KvCacheFormat,
    pub(crate) key_cache_q8: Vec<i8>,
    pub(crate) value_cache_q8: Vec<i8>,
    pub(crate) key_cache_turbo_base: Vec<u8>,
    pub(crate) value_cache_turbo_base: Vec<u8>,
    pub(crate) key_cache_turbo_sign: Vec<u8>,
    pub(crate) value_cache_turbo_sign: Vec<u8>,
    pub(crate) key_cache_scale: Vec<f32>,
    pub(crate) value_cache_scale: Vec<f32>,
    pub(crate) key_cache_residual_norm: Vec<f32>,
    pub(crate) value_cache_residual_norm: Vec<f32>,
    pub(crate) turbo_sign_table: Vec<u8>,
    pub(crate) turbo_scratch0: Vec<f32>,
    pub(crate) turbo_scratch1: Vec<f32>,
    pub(crate) turbo_scratch2: Vec<f32>,
    pub(crate) rope_freqs: Vec<f32>,
    pub(crate) rope_freqs_swa: Vec<f32>,
    pub(crate) rope_cos: Vec<f32>,
    pub(crate) rope_sin: Vec<f32>,
    pub(crate) rope_cache_pos: isize,
    pub(crate) rope_cache_is_swa: isize,
    pub(crate) head_size: usize,
    pub(crate) kv_dim: usize,
    pub(crate) q_dim: usize,
    pub(crate) kv_mul: usize,
    pub(crate) attn_scale: f32,
    pub(crate) embed_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvCacheFormat {
    Q8,
    Turbo,
}

#[derive(Default)]
pub(crate) struct Tokenizer {
    pub(crate) vocab: Vec<String>,
    pub(crate) vocab_scores: Vec<f32>,
    pub(crate) vocab_size: usize,
    pub(crate) max_token_length: usize,
    pub(crate) bos_token: i32,
    pub(crate) eos_token: i32,
    pub(crate) start_header_token: i32,
    pub(crate) end_header_token: i32,
    pub(crate) eot_token: i32,
    pub(crate) pre_tokenizer: TokenizerPreType,
    pub(crate) use_sentencepiece: bool,
    pub(crate) token_to_id: HashMap<String, i32>,
    pub(crate) merges: Vec<String>,
    pub(crate) merge_ranks: HashMap<String, usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum TokenizerPreType {
    #[default]
    Gpt2,
    Qwen2,
    Qwen35,
}

pub(crate) struct XorShiftRng {
    pub(crate) seed: u64,
}

impl XorShiftRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub(crate) fn random_u32(&mut self) -> u32 {
        self.seed ^= self.seed >> 12;
        self.seed ^= self.seed << 25;
        self.seed ^= self.seed >> 27;
        ((self.seed.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 32) as u32
    }

    pub(crate) fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16_777_216.0
    }
}
