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
#[cfg(target_arch = "x86_64")]
static X86_AVX2_FMA_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_AVX2_BF16ENC_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_F16C_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_QK_MR4_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_AVXVNNI_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_AVX512VNNI_Q8_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_AVX512F_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_AVX512BF16_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_IS_AMD_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q4K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q5K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q6K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q8_0_MR2_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q4K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q5K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q6K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
static LAYER_DEBUG_CFG: OnceLock<bool> = OnceLock::new();
static LAYER_DEBUG_POS_CFG: OnceLock<Option<usize>> = OnceLock::new();
static KV_CACHE_MODE_CFG: OnceLock<KvCacheMode> = OnceLock::new();

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
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avx512: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avx512bf16: Option<bool>,
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

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_avx2_fma() -> bool {
    *X86_AVX2_FMA_CFG.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

/// Returns true if AVX-2 bf16/fp16 encoding SIMD is available on x86_64.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_avx2_bf16_enc() -> bool {
    *X86_AVX2_BF16ENC_CFG.get_or_init(|| use_x86_avx2_fma())
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
#[allow(dead_code)]
pub(crate) fn use_x86_avx2_bf16_enc() -> bool {
    false
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

/// Returns true if AVX-512 Foundation (core vector ops) is available on x86_64.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_avx512f() -> bool {
    *X86_AVX512F_CFG.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
    })
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
#[allow(dead_code)]
pub(crate) fn use_x86_avx512f() -> bool {
    false
}

/// Returns true if AVX-512 BF16 instructions are available on x86_64.
/// Intel Core Ultra supports this natively.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_avx512bf16() -> bool {
    *X86_AVX512BF16_CFG.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512bf16")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    })
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
#[allow(dead_code)]
pub(crate) fn use_x86_avx512bf16() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn is_x86_amd() -> bool {
    *X86_IS_AMD_CFG.get_or_init(|| {
        use std::arch::x86_64::__cpuid;

        // CPUID vendor string is EBX, EDX, ECX for leaf 0.
        let leaf0 = unsafe { __cpuid(0) };
        let mut vendor = [0u8; 12];
        vendor[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
        vendor[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());
        vendor == *b"AuthenticAMD"
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
        if let Some(v) = config.x86_avx512 {
            let enabled = v
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512vl");
            let _ = X86_AVX512F_CFG.set(enabled);
        }
        if let Some(v) = config.x86_avx512bf16 {
            let enabled = v
                && std::arch::is_x86_feature_detected!("avx512bf16")
                && std::arch::is_x86_feature_detected!("avx512vnni");
            let _ = X86_AVX512BF16_CFG.set(enabled);
        }
    }
}
