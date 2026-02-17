use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

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
pub(crate) const GGML_TYPE_BF16: i32 = 29;

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

#[cfg(unix)]
pub(crate) const PROT_READ: i32 = 0x1;
#[cfg(unix)]
pub(crate) const MAP_SHARED: i32 = 0x0001;
#[cfg(target_os = "linux")]
const MADV_WILLNEED: i32 = 3;
#[cfg(target_os = "linux")]
const MADV_HUGEPAGE: i32 = 14;

#[cfg(unix)]
extern "C" {
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
extern "C" {
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

    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            let _ = munmap(self.ptr as *mut c_void, self.len);
        }
    }
}

pub(crate) const LAZY_CHUNK_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const LAZY_BOOTSTRAP_START_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const LAZY_BOOTSTRAP_MAX_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const LAZY_FETCH_RETRIES: usize = 3;

pub(crate) static LAZY_MODEL_LOADER: OnceLock<Arc<LazyModelLoader>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LazyChunkState {
    Missing,
    Fetching,
    Ready,
    Failed,
}

pub(crate) struct LazyModelLoader {
    pub(crate) url: String,
    pub(crate) file: File,
    pub(crate) file_len: usize,
    pub(crate) chunk_bytes: usize,
    pub(crate) chunk_count: usize,
    pub(crate) states: Mutex<Vec<LazyChunkState>>,
    pub(crate) cv: Condvar,
    pub(crate) debug_mode: bool,
    pub(crate) ready_chunks: AtomicUsize,
    pub(crate) fetch_attempts: AtomicUsize,
    pub(crate) fetch_waits: AtomicUsize,
    pub(crate) foreground_fetches: AtomicUsize,
    pub(crate) background_fetches: AtomicUsize,
}

impl LazyModelLoader {
    pub(crate) fn new(url: &str, model_path: &str, debug_mode: bool) -> Result<Self, String> {
        let file_len = Self::probe_remote_len(url)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(model_path)
            .map_err(|e| format!("cannot open local cache file {model_path}: {e}"))?;
        file.set_len(file_len as u64)
            .map_err(|e| format!("cannot size cache file {model_path}: {e}"))?;

        let chunk_bytes = LAZY_CHUNK_BYTES;
        let chunk_count = file_len.div_ceil(chunk_bytes);
        let states = vec![LazyChunkState::Missing; chunk_count];

        Ok(Self {
            url: url.to_string(),
            file,
            file_len,
            chunk_bytes,
            chunk_count,
            states: Mutex::new(states),
            cv: Condvar::new(),
            debug_mode,
            ready_chunks: AtomicUsize::new(0),
            fetch_attempts: AtomicUsize::new(0),
            fetch_waits: AtomicUsize::new(0),
            foreground_fetches: AtomicUsize::new(0),
            background_fetches: AtomicUsize::new(0),
        })
    }

    fn probe_remote_len(url: &str) -> Result<usize, String> {
        if let Ok(resp) = ureq::head(url).call() {
            if let Some(v) = resp.header("Content-Length") {
                if let Ok(n) = v.parse::<u64>() {
                    if n > 0 {
                        return Ok(n as usize);
                    }
                }
            }
        }

        let resp = ureq::get(url)
            .set("Range", "bytes=0-0")
            .call()
            .map_err(|e| format!("cannot query remote size from {url}: {e}"))?;
        if let Some(cr) = resp.header("Content-Range") {
            return Self::parse_content_range_total(cr);
        }
        if let Some(v) = resp.header("Content-Length") {
            if let Ok(n) = v.parse::<u64>() {
                if n > 0 {
                    return Ok(n as usize);
                }
            }
        }
        Err(format!(
            "cannot determine remote size for {url} (missing Content-Length/Content-Range)"
        ))
    }

    fn parse_content_range_total(content_range: &str) -> Result<usize, String> {
        let total = content_range
            .rsplit('/')
            .next()
            .ok_or_else(|| format!("invalid Content-Range header: {content_range}"))?;
        total
            .parse::<u64>()
            .map(|v| v as usize)
            .map_err(|e| format!("invalid Content-Range total in {content_range}: {e}"))
    }

    fn chunk_bounds(&self, chunk_idx: usize) -> (usize, usize) {
        let start = chunk_idx * self.chunk_bytes;
        let end = (start + self.chunk_bytes).min(self.file_len);
        (start, end)
    }

    fn fetch_chunk_into_cache(&self, chunk_idx: usize) -> Result<(), String> {
        let (start, end) = self.chunk_bounds(chunk_idx);
        let mut last_err: Option<String> = None;

        for _ in 0..LAZY_FETCH_RETRIES {
            match self.fetch_chunk_once(start, end) {
                Ok(()) => return Ok(()),
                Err(e) => last_err = Some(e),
            }
        }

        Err(last_err.unwrap_or_else(|| "unknown chunk fetch failure".to_string()))
    }

    fn fetch_chunk_once(&self, start: usize, end: usize) -> Result<(), String> {
        let range = format!("bytes={start}-{}", end - 1);
        let resp = ureq::get(&self.url)
            .set("Range", &range)
            .call()
            .map_err(|e| format!("range request failed ({range}): {e}"))?;

        let status = resp.status();
        if status != 206 && !(status == 200 && start == 0 && end == self.file_len) {
            return Err(format!(
                "unexpected HTTP status {} for range {}",
                status, range
            ));
        }

        let mut body = Vec::with_capacity(end - start);
        resp.into_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("failed reading HTTP body for range {range}: {e}"))?;
        if body.len() != end - start {
            return Err(format!(
                "short read for range {range}: got {} bytes, expected {}",
                body.len(),
                end - start
            ));
        }

        write_all_at(&self.file, &body, start as u64)
            .map_err(|e| format!("failed writing cache bytes [{start}..{end}): {e}"))?;
        Ok(())
    }

    fn ensure_chunk_ready(&self, chunk_idx: usize, is_background: bool) -> Result<(), String> {
        let mut states = self
            .states
            .lock()
            .map_err(|_| "lazy loader state lock poisoned".to_string())?;
        loop {
            match states[chunk_idx] {
                LazyChunkState::Ready => return Ok(()),
                LazyChunkState::Missing | LazyChunkState::Failed => {
                    states[chunk_idx] = LazyChunkState::Fetching;
                    self.fetch_attempts.fetch_add(1, AtomicOrdering::Relaxed);
                    if is_background {
                        self.background_fetches
                            .fetch_add(1, AtomicOrdering::Relaxed);
                    } else {
                        self.foreground_fetches
                            .fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    break;
                }
                LazyChunkState::Fetching => {
                    self.fetch_waits.fetch_add(1, AtomicOrdering::Relaxed);
                    states = self
                        .cv
                        .wait(states)
                        .map_err(|_| "lazy loader state lock poisoned".to_string())?;
                }
            }
        }
        drop(states);

        let result = self.fetch_chunk_into_cache(chunk_idx);
        let mut states = self
            .states
            .lock()
            .map_err(|_| "lazy loader state lock poisoned".to_string())?;
        states[chunk_idx] = if result.is_ok() {
            self.ready_chunks.fetch_add(1, AtomicOrdering::Relaxed);
            LazyChunkState::Ready
        } else {
            LazyChunkState::Failed
        };
        self.cv.notify_all();
        result
    }

    pub(crate) fn debug_stats_line(&self) -> String {
        let ready = self.ready_chunks.load(AtomicOrdering::Relaxed);
        let pct = if self.chunk_count == 0 {
            100.0
        } else {
            100.0 * ready as f64 / self.chunk_count as f64
        };
        let attempts = self.fetch_attempts.load(AtomicOrdering::Relaxed);
        let waits = self.fetch_waits.load(AtomicOrdering::Relaxed);
        let fg = self.foreground_fetches.load(AtomicOrdering::Relaxed);
        let bg = self.background_fetches.load(AtomicOrdering::Relaxed);
        format!(
            "Lazy model: ready={}/{} ({:.1}%), fetch_attempts={}, waits={}, fg_fetches={}, bg_fetches={}",
            ready, self.chunk_count, pct, attempts, waits, fg, bg
        )
    }

    pub(crate) fn ensure_range(&self, offset: usize, len: usize) -> Result<(), String> {
        if len == 0 || self.chunk_count == 0 {
            return Ok(());
        }
        if offset >= self.file_len {
            return Err(format!(
                "lazy ensure_range offset {} outside file size {}",
                offset, self.file_len
            ));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| "lazy ensure_range overflow".to_string())?
            .min(self.file_len);
        let first = offset / self.chunk_bytes;
        let last = (end - 1) / self.chunk_bytes;
        for c in first..=last {
            self.ensure_chunk_ready(c, false)?;
        }
        Ok(())
    }

    pub(crate) fn start_background_download(self: &Arc<Self>) {
        let this = Arc::clone(self);
        std::thread::spawn(move || {
            for chunk_idx in 0..this.chunk_count {
                if let Err(e) = this.ensure_chunk_ready(chunk_idx, true) {
                    if this.debug_mode {
                        eprintln!(
                            "Lazy model background download stopped at chunk {}: {}",
                            chunk_idx, e
                        );
                    }
                    return;
                }
                if this.debug_mode
                    && (chunk_idx + 1 == this.chunk_count || (chunk_idx + 1) % 128 == 0)
                {
                    eprintln!("{}", this.debug_stats_line());
                }
            }
            if this.debug_mode {
                eprintln!("Lazy model background download finished");
                eprintln!("{}", this.debug_stats_line());
            }
        });
    }
}

pub(crate) fn ensure_model_range(offset: usize, len: usize) -> Result<(), String> {
    if let Some(loader) = LAZY_MODEL_LOADER.get() {
        loader.ensure_range(offset, len)?;
    }
    Ok(())
}

fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        while !buf.is_empty() {
            let n = file.write_at(buf, offset)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write model cache",
                ));
            }
            buf = &buf[n..];
            offset += n as u64;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        let _ = buf;
        let _ = offset;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "lazy model cache writes require unix platform",
        ))
    }
}

#[derive(Clone, Debug)]
pub(crate) enum GgufValue {
    UInt(u64),
    Int(i64),
    F32(f32),
    F64(f64),
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
    pub(crate) lazy_loader: Option<Arc<LazyModelLoader>>,
}

impl GGUFFile {
    #[inline]
    pub(crate) fn ensure_range(&self, offset: usize, len: usize) -> Result<(), String> {
        if let Some(loader) = &self.lazy_loader {
            loader.ensure_range(offset, len)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct Config {
    pub(crate) dim: usize,
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
    pub(crate) rope_freq_scale: f32,
    pub(crate) head_dim: usize,
    pub(crate) rope_dim: usize,
    pub(crate) rope_scaling_yarn: bool,
    pub(crate) rope_orig_ctx: usize,
    pub(crate) rope_yarn_log_multiplier: f32,
    pub(crate) is_gemma3: bool,
    pub(crate) is_qwen2: bool,
    pub(crate) is_qwen3moe: bool,
    pub(crate) is_qwen3next: bool,
    pub(crate) is_deepseek2: bool,
    pub(crate) final_logit_softcapping: f32,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta_swa: f32,
    pub(crate) swa_pattern: usize,
    pub(crate) ssm_conv_kernel: usize,
    pub(crate) ssm_inner_size: usize,
    pub(crate) ssm_state_size: usize,
    pub(crate) ssm_time_step_rank: usize,
    pub(crate) ssm_group_count: usize,
    pub(crate) deepseek_leading_dense_block_count: usize,
    pub(crate) deepseek_q_lora_rank: usize,
    pub(crate) deepseek_kv_lora_rank: usize,
    pub(crate) deepseek_qk_nope_head_dim: usize,
    pub(crate) deepseek_qk_rope_head_dim: usize,
    pub(crate) deepseek_v_head_dim: usize,
    pub(crate) deepseek_expert_gating_func: i32,
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
    pub(crate) deepseek_wq_a: Vec<QuantizedTensor>,
    pub(crate) deepseek_wq_b: Vec<QuantizedTensor>,
    pub(crate) deepseek_wkv_a: Vec<QuantizedTensor>,
    pub(crate) deepseek_wk_b: Vec<QuantizedTensor>,
    pub(crate) deepseek_wv_b: Vec<QuantizedTensor>,
    pub(crate) deepseek_q_a_norm: Vec<f32>,
    pub(crate) deepseek_kv_a_norm: Vec<f32>,
    pub(crate) deepseek_layer_is_moe: Vec<bool>,
    pub(crate) deepseek_shared_gate_present: Vec<bool>,
    pub(crate) deepseek_exp_probs_bias: Vec<f32>,
    pub(crate) ssm_ba: Vec<QuantizedTensor>,
    pub(crate) ssm_conv1d: Vec<Vec<f32>>,
    pub(crate) ssm_a: Vec<f32>,
    pub(crate) ssm_dt_bias: Vec<f32>,
    pub(crate) ssm_norm: Vec<f32>,
    pub(crate) moe_gate_inp: Vec<QuantizedTensor>,
    pub(crate) moe_gate_exps: Vec<QuantizedTensor>,
    pub(crate) moe_up_exps: Vec<QuantizedTensor>,
    pub(crate) moe_down_exps: Vec<QuantizedTensor>,
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
    pub(crate) key_cache_q4: Vec<u8>,
    pub(crate) value_cache_q4: Vec<u8>,
    pub(crate) key_cache_scale: Vec<f32>,
    pub(crate) value_cache_scale: Vec<f32>,
    pub(crate) deepseek_key_cache: Vec<f32>,
    pub(crate) deepseek_value_cache: Vec<f32>,
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
    Q4,
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
    pub(crate) use_sentencepiece: bool,
    pub(crate) token_to_id: HashMap<String, i32>,
    pub(crate) merges: Vec<String>,
    pub(crate) merge_ranks: HashMap<String, usize>,
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
