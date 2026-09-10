// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.

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

pub(crate) fn profiling_reset() {
    PROF_TRANSFORMER_NS.store(0, AtomicOrdering::Relaxed);
    PROF_MATMUL_NS.store(0, AtomicOrdering::Relaxed);
    PROF_SSM_NS.store(0, AtomicOrdering::Relaxed);
    PROF_ATTN_NS.store(0, AtomicOrdering::Relaxed);
    PROF_MOE_NS.store(0, AtomicOrdering::Relaxed);
    PROF_FFN_NS.store(0, AtomicOrdering::Relaxed);
    PROF_FORWARD_PASSES.store(0, AtomicOrdering::Relaxed);
}

pub(crate) fn print_profile_report() {
    let total_ns = PROF_TRANSFORMER_NS.load(AtomicOrdering::Relaxed);
    let matmul_ns = PROF_MATMUL_NS.load(AtomicOrdering::Relaxed);
    let ssm_ns = PROF_SSM_NS.load(AtomicOrdering::Relaxed);
    let attn_ns = PROF_ATTN_NS.load(AtomicOrdering::Relaxed);
    let moe_ns = PROF_MOE_NS.load(AtomicOrdering::Relaxed);
    let ffn_ns = PROF_FFN_NS.load(AtomicOrdering::Relaxed);
    let passes = PROF_FORWARD_PASSES.load(AtomicOrdering::Relaxed);

    let to_ms = |ns: u64| ns as f64 / 1_000_000.0;
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
}
