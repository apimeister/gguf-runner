use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

pub(crate) static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);
pub(crate) static PROF_TRANSFORMER_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_MATMUL_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_SSM_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_ATTN_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_MOE_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_FFN_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_FORWARD_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_REMOTE_BATCHES: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_REMOTE_EXPERTS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_LOCAL_EXPERTS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_BYTES_SENT: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_BYTES_RECEIVED: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_REMOTE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_COORD_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_COORD_LOCAL_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_COORD_REMOTE_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_PUSH_FRAMES: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_DMOE_PUSH_EXPERTS: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
pub(crate) fn set_profiling_enabled(enabled: bool) {
    PROFILING_ENABLED.store(enabled, AtomicOrdering::Relaxed);
}

#[inline(always)]
pub(crate) fn profiling_enabled() -> bool {
    PROFILING_ENABLED.load(AtomicOrdering::Relaxed)
}

#[inline(always)]
pub(crate) fn prof_start() -> Option<Instant> {
    if profiling_enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

#[inline(always)]
pub(crate) fn prof_end(counter: &AtomicU64, start: Option<Instant>) {
    if let Some(t0) = start {
        counter.fetch_add(t0.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    }
}

pub(crate) fn record_forward_pass() {
    PROF_FORWARD_PASSES.fetch_add(1, AtomicOrdering::Relaxed);
}

#[inline(always)]
pub(crate) fn record_distributed_local_experts(count: usize) {
    PROF_DMOE_LOCAL_EXPERTS.fetch_add(count as u64, AtomicOrdering::Relaxed);
}

#[inline(always)]
pub(crate) fn record_distributed_remote_request(
    experts: usize,
    bytes_sent: usize,
    bytes_received: usize,
    wait_ns: u64,
) {
    PROF_DMOE_REMOTE_BATCHES.fetch_add(1, AtomicOrdering::Relaxed);
    PROF_DMOE_REMOTE_EXPERTS.fetch_add(experts as u64, AtomicOrdering::Relaxed);
    PROF_DMOE_BYTES_SENT.fetch_add(bytes_sent as u64, AtomicOrdering::Relaxed);
    PROF_DMOE_BYTES_RECEIVED.fetch_add(bytes_received as u64, AtomicOrdering::Relaxed);
    PROF_DMOE_REMOTE_WAIT_NS.fetch_add(wait_ns, AtomicOrdering::Relaxed);
}

#[inline(always)]
pub(crate) fn record_distributed_transport_bytes(bytes_sent: usize, bytes_received: usize) {
    PROF_DMOE_BYTES_SENT.fetch_add(bytes_sent as u64, AtomicOrdering::Relaxed);
    PROF_DMOE_BYTES_RECEIVED.fetch_add(bytes_received as u64, AtomicOrdering::Relaxed);
}

#[inline(always)]
pub(crate) fn record_distributed_coordinator_timing(total_ns: u64, local_ns: u64, remote_ns: u64) {
    PROF_DMOE_COORD_TOTAL_NS.fetch_add(total_ns, AtomicOrdering::Relaxed);
    PROF_DMOE_COORD_LOCAL_NS.fetch_add(local_ns, AtomicOrdering::Relaxed);
    PROF_DMOE_COORD_REMOTE_NS.fetch_add(remote_ns, AtomicOrdering::Relaxed);
}

#[inline(always)]
pub(crate) fn record_distributed_push_frame(experts: usize) {
    PROF_DMOE_PUSH_FRAMES.fetch_add(1, AtomicOrdering::Relaxed);
    PROF_DMOE_PUSH_EXPERTS.fetch_add(experts as u64, AtomicOrdering::Relaxed);
}

pub(crate) fn profiling_reset() {
    PROF_TRANSFORMER_NS.store(0, AtomicOrdering::Relaxed);
    PROF_MATMUL_NS.store(0, AtomicOrdering::Relaxed);
    PROF_SSM_NS.store(0, AtomicOrdering::Relaxed);
    PROF_ATTN_NS.store(0, AtomicOrdering::Relaxed);
    PROF_MOE_NS.store(0, AtomicOrdering::Relaxed);
    PROF_FFN_NS.store(0, AtomicOrdering::Relaxed);
    PROF_FORWARD_PASSES.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_REMOTE_BATCHES.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_REMOTE_EXPERTS.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_LOCAL_EXPERTS.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_BYTES_SENT.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_BYTES_RECEIVED.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_REMOTE_WAIT_NS.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_COORD_TOTAL_NS.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_COORD_LOCAL_NS.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_COORD_REMOTE_NS.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_PUSH_FRAMES.store(0, AtomicOrdering::Relaxed);
    PROF_DMOE_PUSH_EXPERTS.store(0, AtomicOrdering::Relaxed);
}

pub(crate) fn print_profile_report() {
    let total_ns = PROF_TRANSFORMER_NS.load(AtomicOrdering::Relaxed);
    let matmul_ns = PROF_MATMUL_NS.load(AtomicOrdering::Relaxed);
    let ssm_ns = PROF_SSM_NS.load(AtomicOrdering::Relaxed);
    let attn_ns = PROF_ATTN_NS.load(AtomicOrdering::Relaxed);
    let moe_ns = PROF_MOE_NS.load(AtomicOrdering::Relaxed);
    let ffn_ns = PROF_FFN_NS.load(AtomicOrdering::Relaxed);
    let passes = PROF_FORWARD_PASSES.load(AtomicOrdering::Relaxed);
    let distributed_remote_batches = PROF_DMOE_REMOTE_BATCHES.load(AtomicOrdering::Relaxed);
    let distributed_remote_experts = PROF_DMOE_REMOTE_EXPERTS.load(AtomicOrdering::Relaxed);
    let distributed_local_experts = PROF_DMOE_LOCAL_EXPERTS.load(AtomicOrdering::Relaxed);
    let distributed_bytes_sent = PROF_DMOE_BYTES_SENT.load(AtomicOrdering::Relaxed);
    let distributed_bytes_received = PROF_DMOE_BYTES_RECEIVED.load(AtomicOrdering::Relaxed);
    let distributed_remote_wait_ns = PROF_DMOE_REMOTE_WAIT_NS.load(AtomicOrdering::Relaxed);
    let distributed_coord_total_ns = PROF_DMOE_COORD_TOTAL_NS.load(AtomicOrdering::Relaxed);
    let distributed_coord_local_ns = PROF_DMOE_COORD_LOCAL_NS.load(AtomicOrdering::Relaxed);
    let distributed_coord_remote_ns = PROF_DMOE_COORD_REMOTE_NS.load(AtomicOrdering::Relaxed);
    let distributed_push_frames = PROF_DMOE_PUSH_FRAMES.load(AtomicOrdering::Relaxed);
    let distributed_push_experts = PROF_DMOE_PUSH_EXPERTS.load(AtomicOrdering::Relaxed);

    let to_ms = |ns: u64| ns as f64 / 1_000_000.0;
    let to_mb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    let pct = |part: u64| {
        if total_ns == 0 {
            0.0
        } else {
            (part as f64 * 100.0) / total_ns as f64
        }
    };

    eprintln!("\n[PROFILE] forward_passes={passes}");
    eprintln!(
        "[PROFILE] transformer_total={:.3} ms ({:.3} ms/pass)",
        to_ms(total_ns),
        if passes == 0 {
            0.0
        } else {
            to_ms(total_ns) / passes as f64
        }
    );
    eprintln!(
        "[PROFILE] matmul={:.3} ms ({:.1}%)",
        to_ms(matmul_ns),
        pct(matmul_ns)
    );
    eprintln!(
        "[PROFILE] ssm={:.3} ms ({:.1}%)",
        to_ms(ssm_ns),
        pct(ssm_ns)
    );
    eprintln!(
        "[PROFILE] attention={:.3} ms ({:.1}%)",
        to_ms(attn_ns),
        pct(attn_ns)
    );
    eprintln!(
        "[PROFILE] moe={:.3} ms ({:.1}%)",
        to_ms(moe_ns),
        pct(moe_ns)
    );
    eprintln!(
        "[PROFILE] ffn={:.3} ms ({:.1}%)",
        to_ms(ffn_ns),
        pct(ffn_ns)
    );
    eprintln!(
        "[PROFILE] note: counters overlap (e.g. matmul is included in SSM/attention/MoE/FFN)"
    );
    if distributed_remote_batches > 0
        || distributed_remote_experts > 0
        || distributed_bytes_sent > 0
        || distributed_bytes_received > 0
    {
        eprintln!(
            "[PROFILE] distributed_moe remote_batches={} remote_experts={} local_experts={}",
            distributed_remote_batches, distributed_remote_experts, distributed_local_experts
        );
        eprintln!(
            "[PROFILE] distributed_moe transport_sent={:.3} MiB transport_recv={:.3} MiB",
            to_mb(distributed_bytes_sent),
            to_mb(distributed_bytes_received)
        );
        eprintln!(
            "[PROFILE] distributed_moe remote_wait={:.3} ms ({:.3} ms/batch)",
            to_ms(distributed_remote_wait_ns),
            if distributed_remote_batches == 0 {
                0.0
            } else {
                to_ms(distributed_remote_wait_ns) / distributed_remote_batches as f64
            }
        );
        eprintln!(
            "[PROFILE] distributed_moe coordinator_total={:.3} ms local_compute={:.3} ms remote_dispatch={:.3} ms",
            to_ms(distributed_coord_total_ns),
            to_ms(distributed_coord_local_ns),
            to_ms(distributed_coord_remote_ns)
        );
        eprintln!(
            "[PROFILE] distributed_moe push_frames={} push_experts={}",
            distributed_push_frames, distributed_push_experts
        );
    }
}
