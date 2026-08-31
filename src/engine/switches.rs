use std::sync::OnceLock;
use std::sync::atomic::AtomicU8;

#[inline]
fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn par_matmul_min_rows_default() -> usize {
    let n_threads = available_threads();
    if n_threads <= 4 { 192 } else { 384 }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn par_matmul_min_rows_default() -> usize {
    let n_threads = available_threads();
    if n_threads <= 2 {
        128
    } else if n_threads <= 4 {
        192
    } else if n_threads <= 8 {
        256
    } else if n_threads <= 12 {
        320
    } else {
        384
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn par_matmul_chunk_rows_default() -> usize {
    let n_threads = available_threads();
    if n_threads <= 4 { 32 } else { 64 }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn par_matmul_chunk_rows_default() -> usize {
    let n_threads = available_threads();
    if n_threads <= 2 {
        16
    } else if n_threads <= 4 {
        24
    } else if n_threads <= 8 {
        32
    } else if n_threads <= 12 {
        48
    } else {
        64
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn aarch64_matmul_prefetch_rows_default() -> usize {
    let n_threads = available_threads();
    if n_threads <= 4 {
        4
    } else if n_threads <= 8 {
        6
    } else if n_threads <= 12 {
        8
    } else {
        10
    }
}

#[inline]
fn par_attn_min_heads_default() -> usize {
    let n_threads = available_threads();
    if n_threads <= 4 { 4 } else { 8 }
}

#[inline]
fn par_qwen3next_min_heads_default() -> usize {
    let n_threads = available_threads();
    if n_threads <= 4 { 4 } else { 8 }
}

static PAR_MATMUL_MIN_ROWS_CFG: OnceLock<usize> = OnceLock::new();
static PAR_MATMUL_CHUNK_ROWS_CFG: OnceLock<usize> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
static AARCH64_MATMUL_PREFETCH_ROWS_CFG: OnceLock<usize> = OnceLock::new();
static PAR_ATTN_MIN_HEADS_CFG: OnceLock<usize> = OnceLock::new();
static PAR_QWEN3NEXT_MIN_HEADS_CFG: OnceLock<usize> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
static AARCH64_DOTPROD_Q8_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
static AARCH64_QK_MR4_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
static AARCH64_I8MM_Q8_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
static AARCH64_BF16_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_AVX2_FMA_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_F16C_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_QK_MR4_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_AVXVNNI_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_AVX512VNNI_Q8_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q2K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q3K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q4_0_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q4_1_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_IQ4_NL_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q5_0_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q5_1_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q4K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q5K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q6K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q8_0_MR2_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q2K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q3K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q4_0_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q4_1_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q5_0_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q5_1_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_IQ4_NL_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q4K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q5K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q6K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
static LAYER_DEBUG_CFG: OnceLock<bool> = OnceLock::new();
static LAYER_DEBUG_POS_CFG: OnceLock<Option<usize>> = OnceLock::new();
static KV_CACHE_MODE_CFG: OnceLock<KvCacheMode> = OnceLock::new();
static BATCH_PREFILL_CFG: OnceLock<bool> = OnceLock::new();
static BATCH_PREFILL_EXACT_CFG: OnceLock<bool> = OnceLock::new();
static BATCH_PREFILL_CHUNK_CFG: OnceLock<usize> = OnceLock::new();
static MM_FLOAT_BATCH_CFG: OnceLock<bool> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvCacheMode {
    Q8,
    Turbo,
}

pub(crate) struct RuntimeSwitchConfig {
    pub(crate) par_matmul_min_rows: Option<usize>,
    pub(crate) par_matmul_chunk_rows: Option<usize>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_matmul_prefetch_rows: Option<usize>,
    pub(crate) par_attn_min_heads: Option<usize>,
    pub(crate) par_qwen3next_min_heads: Option<usize>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_dotprod_q8: Option<bool>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_qk_mr4: Option<bool>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_i8mm: Option<bool>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_bf16: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avx2: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_f16c: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_qk_mr4: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avxvnni: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avx512vnni_q8: Option<bool>,
    pub(crate) layer_debug: Option<bool>,
    pub(crate) layer_debug_pos: Option<usize>,
    pub(crate) kv_cache_mode: Option<KvCacheMode>,
}

#[inline]
pub(crate) fn layer_debug_enabled() -> bool {
    *LAYER_DEBUG_CFG.get_or_init(|| false)
}

#[inline]
pub(crate) fn layer_debug_pos() -> Option<usize> {
    *LAYER_DEBUG_POS_CFG.get_or_init(|| None)
}

#[inline]
pub(crate) fn par_matmul_min_rows() -> usize {
    *PAR_MATMUL_MIN_ROWS_CFG.get_or_init(par_matmul_min_rows_default)
}

#[inline]
pub(crate) fn par_matmul_chunk_rows() -> usize {
    *PAR_MATMUL_CHUNK_ROWS_CFG.get_or_init(par_matmul_chunk_rows_default)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn aarch64_matmul_prefetch_rows() -> usize {
    *AARCH64_MATMUL_PREFETCH_ROWS_CFG.get_or_init(aarch64_matmul_prefetch_rows_default)
}

#[inline]
pub(crate) fn par_attn_min_heads() -> usize {
    *PAR_ATTN_MIN_HEADS_CFG.get_or_init(par_attn_min_heads_default)
}

#[inline]
pub(crate) fn par_qwen3next_min_heads() -> usize {
    *PAR_QWEN3NEXT_MIN_HEADS_CFG.get_or_init(par_qwen3next_min_heads_default)
}

#[inline]
pub(crate) fn kv_cache_mode() -> KvCacheMode {
    *KV_CACHE_MODE_CFG.get_or_init(|| KvCacheMode::Turbo)
}

/// Batched prefill kill switch: on by default, `GGUF_BATCH_PREFILL=0` disables.
#[inline]
pub(crate) fn use_batch_prefill() -> bool {
    *BATCH_PREFILL_CFG.get_or_init(|| {
        std::env::var("GGUF_BATCH_PREFILL")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// Debug mode: use the bitwise sequential-mirror batched matmul instead of
/// the fast dequantize-once kernel (`GGUF_BATCH_PREFILL_EXACT=1`). Isolates
/// whether a batched-vs-sequential output difference is structural (a bug —
/// exact mode must match bitwise) or f32-rounding from the fast kernel.
#[inline]
pub(crate) fn batch_prefill_exact() -> bool {
    *BATCH_PREFILL_EXACT_CFG.get_or_init(|| {
        std::env::var("GGUF_BATCH_PREFILL_EXACT")
            .map(|v| v.trim() == "1")
            .unwrap_or(false)
    })
}

/// Tokens per batched-prefill chunk (`GGUF_BATCH_PREFILL_CHUNK`, default 64).
#[inline]
pub(crate) fn batch_prefill_chunk() -> usize {
    *BATCH_PREFILL_CHUNK_CFG.get_or_init(|| {
        std::env::var("GGUF_BATCH_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&c| c > 0)
            .unwrap_or(64)
    })
}

/// Multimodal F16/BF16 encoder token batching: on by default,
/// `GGUF_MM_FLOAT_BATCH=0` restores the parallel per-token path.
#[inline]
pub(crate) fn use_mm_float_batch() -> bool {
    *MM_FLOAT_BATCH_CFG.get_or_init(|| {
        std::env::var("GGUF_MM_FLOAT_BATCH")
            .map(|value| value.trim() != "0")
            .unwrap_or(true)
    })
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn use_aarch64_dotprod_q8() -> bool {
    *AARCH64_DOTPROD_Q8_CFG.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn use_aarch64_qk_mr4() -> bool {
    *AARCH64_QK_MR4_CFG.get_or_init(|| true)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn use_aarch64_i8mm_q8() -> bool {
    *AARCH64_I8MM_Q8_CFG.get_or_init(|| std::arch::is_aarch64_feature_detected!("i8mm"))
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn use_aarch64_bf16() -> bool {
    *AARCH64_BF16_CFG.get_or_init(|| std::arch::is_aarch64_feature_detected!("bf16"))
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_avx2_fma() -> bool {
    *X86_AVX2_FMA_CFG.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_f16c() -> bool {
    *X86_F16C_CFG.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_qk_mr4() -> bool {
    *X86_QK_MR4_CFG.get_or_init(|| true)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_avx_vnni() -> bool {
    *X86_AVXVNNI_CFG.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avxvnni")
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_avx512_vnni_q8() -> bool {
    *X86_AVX512VNNI_Q8_CFG.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vl")
    })
}

pub(crate) fn init_runtime_config(config: &RuntimeSwitchConfig) {
    if let Some(v) = config.par_matmul_min_rows {
        let _ = PAR_MATMUL_MIN_ROWS_CFG.set(v);
    }
    if let Some(v) = config.par_matmul_chunk_rows {
        let _ = PAR_MATMUL_CHUNK_ROWS_CFG.set(v);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(v) = config.aarch64_matmul_prefetch_rows {
        let _ = AARCH64_MATMUL_PREFETCH_ROWS_CFG.set(v);
    }
    if let Some(v) = config.par_attn_min_heads {
        let _ = PAR_ATTN_MIN_HEADS_CFG.set(v);
    }
    if let Some(v) = config.par_qwen3next_min_heads {
        let _ = PAR_QWEN3NEXT_MIN_HEADS_CFG.set(v);
    }
    if let Some(v) = config.layer_debug {
        let _ = LAYER_DEBUG_CFG.set(v);
    }
    if let Some(v) = config.layer_debug_pos {
        let _ = LAYER_DEBUG_POS_CFG.set(Some(v));
    }
    if let Some(v) = config.kv_cache_mode {
        let _ = KV_CACHE_MODE_CFG.set(v);
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(v) = config.aarch64_dotprod_q8 {
            let enabled = v && std::arch::is_aarch64_feature_detected!("dotprod");
            let _ = AARCH64_DOTPROD_Q8_CFG.set(enabled);
        }
        if let Some(v) = config.aarch64_qk_mr4 {
            let _ = AARCH64_QK_MR4_CFG.set(v);
        }
        if let Some(v) = config.aarch64_i8mm {
            let enabled = v && std::arch::is_aarch64_feature_detected!("i8mm");
            let _ = AARCH64_I8MM_Q8_CFG.set(enabled);
        }
        if let Some(v) = config.aarch64_bf16 {
            let enabled = v && std::arch::is_aarch64_feature_detected!("bf16");
            let _ = AARCH64_BF16_CFG.set(enabled);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(v) = config.x86_avx2 {
            let enabled = v
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma");
            let _ = X86_AVX2_FMA_CFG.set(enabled);
        }
        if let Some(v) = config.x86_f16c {
            let enabled = v
                && std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("f16c")
                && std::arch::is_x86_feature_detected!("fma");
            let _ = X86_F16C_CFG.set(enabled);
        }
        if let Some(v) = config.x86_qk_mr4 {
            let _ = X86_QK_MR4_CFG.set(v);
        }
        if let Some(v) = config.x86_avxvnni {
            let enabled = v
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("avxvnni");
            let _ = X86_AVXVNNI_CFG.set(enabled);
        }
        if let Some(v) = config.x86_avx512vnni_q8 {
            let enabled = v
                && std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vl");
            let _ = X86_AVX512VNNI_Q8_CFG.set(enabled);
        }
    }
}
