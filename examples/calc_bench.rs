//! # calc_bench — calculation engine profiler
//!
//! Benchmarks the raw forward-pass (calculation) performance of the engine without any
//! sampling, tokenization, or I/O overhead. Use this to measure how fast the model
//! computes and where time is spent (attention, FFN, matmul, MoE).
//!
//! ## Quick start
//!
//! ```
//! cargo run --release --example calc_bench -- --model path/to/model.gguf
//! ```
//!
//! Always run with `--release`; the dev profile is many times slower and not useful for
//! real throughput numbers.
//!
//! ## Options
//!
//! | Flag | Short | Default | Description |
//! |------|-------|---------|-------------|
//! | `--model <path>` | `-m` | required | Path to a GGUF model file |
//! | `--tokens <n>` | `-n` | `10` | Number of forward passes to measure |
//! | `--warmup <n>` | `-w` | `3` | Unmeasured passes run first to warm up caches |
//! | `--context <n>` | `-c` | model default | Override the context / sequence length |
//! | `--threads <n>` | `-t` | auto | Number of CPU threads (Rayon pool size) |
//!
//! ## What it measures
//!
//! Each measured pass calls `transformer(token, pos, ...)` directly — no sampling,
//! no tokenizer, no KV-cache management beyond what the engine does internally.
//! The dummy input token (id=1) and content are irrelevant; only the compute matters.
//!
//! Passes run at positions 0 … N-1, so the KV cache grows across them exactly as it
//! would during the decode phase of real inference.
//!
//! ### Output
//!
//! ```text
//! [BENCH] tokens=10 elapsed=1234ms avg=123.4ms/tok throughput=8.11tok/s
//!
//! [PROFILE] forward_passes=10
//! [PROFILE] transformer_total=1234.567 ms (123.457 ms/pass)
//! [PROFILE] matmul=900.123 ms (72.9%)
//! [PROFILE] ssm=0.000 ms (0.0%)
//! [PROFILE] attention=600.456 ms (48.6%)
//! [PROFILE] moe=0.000 ms (0.0%)
//! [PROFILE] ffn=290.234 ms (23.5%)
//! [PROFILE] note: counters overlap (e.g. matmul is included in SSM/attention/MoE/FFN)
//! ```
//!
//! The `[PROFILE]` counters are recorded by the engine's own instrumentation; the
//! `[BENCH]` line is wall-clock time measured around the loop in this example.
//! Because attention and FFN both call matmul internally, `matmul %` will exceed 100 %
//! when summed with the other phases — that is expected and noted in the output.
//!
//! ## Typical usage patterns
//!
//! **Decode throughput at default context:**
//! ```
//! cargo run --release --example calc_bench -- -m model.gguf -n 20
//! ```
//!
//! **Check how a large context affects per-token cost** (attention scales with context):
//! ```
//! cargo run --release --example calc_bench -- -m model.gguf -n 10 -c 32768
//! ```
//!
//! **Thread scaling experiment:**
//! ```
//! cargo run --release --example calc_bench -- -m model.gguf -n 10 -t 4
//! cargo run --release --example calc_bench -- -m model.gguf -n 10 -t 8
//! ```
//!
//! **Quick smoke-test (1 warmup, 1 token):**
//! ```
//! cargo run --release --example calc_bench -- -m model.gguf -w 1 -n 1
//! ```

#![allow(dead_code)]
#![allow(unused_imports)]

#[path = "../src/engine/mod.rs"]
mod engine;
#[path = "../src/vendors/mod.rs"]
mod vendors;

use engine::io::parse_gguf_file;
use engine::profiling::{print_profile_report, profiling_reset, set_profiling_enabled};
use engine::runtime::{apply_context_size_overrides, configure_rayon_threads, malloc_run_state, transformer};
use engine::weights::init_weights_from_gguf;
use std::time::Instant;

struct Options {
    model: String,
    context_size: usize,
    tokens: usize,
    warmup: usize,
    threads: Option<usize>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let opts = parse_args()?;

    eprintln!("Loading model: {}", opts.model);
    let gguf = parse_gguf_file(&opts.model, false)?;

    let mut config = vendors::build_config_from_gguf(&gguf, false)?;
    apply_context_size_overrides(&mut config, opts.context_size, false);

    if let Some(n) = opts.threads {
        configure_rayon_threads(n, false);
    }

    let weights = init_weights_from_gguf(&gguf, &config, false)?;
    let mut state = malloc_run_state(&config)?;
    let mapped = gguf.mapped.as_slice();

    eprintln!(
        "Model: layers={}, dim={}, heads={}, kv_heads={}, seq_len={}, vocab={}",
        config.n_layers,
        config.dim,
        config.n_heads,
        config.n_kv_heads,
        config.seq_len,
        config.vocab_size,
    );
    eprintln!(
        "Benchmark: warmup={}, tokens={}",
        opts.warmup,
        opts.tokens,
    );

    // use a token id that's always valid; content is irrelevant for the calc benchmark
    let token: usize = 1;

    // warmup: run a few passes so branch predictors and caches stabilise, not measured
    set_profiling_enabled(false);
    for pos in 0..opts.warmup {
        transformer(token, pos, &config, &mut state, &weights, mapped)
            .map_err(|e| format!("warmup forward pass failed: {e}"))?;
    }

    // measured run
    profiling_reset();
    set_profiling_enabled(true);
    let t_start = Instant::now();
    for pos in 0..opts.tokens {
        transformer(token, pos, &config, &mut state, &weights, mapped)
            .map_err(|e| format!("forward pass at pos={pos} failed: {e}"))?;
    }
    let elapsed = t_start.elapsed();
    set_profiling_enabled(false);

    let tokens_per_sec = opts.tokens as f64 / elapsed.as_secs_f64();
    let ms_per_token = elapsed.as_secs_f64() * 1000.0 / opts.tokens as f64;

    eprintln!();
    eprintln!(
        "[BENCH] tokens={} elapsed={:.3}ms avg={:.3}ms/tok throughput={:.2}tok/s",
        opts.tokens,
        elapsed.as_millis(),
        ms_per_token,
        tokens_per_sec,
    );
    print_profile_report();

    Ok(())
}

fn parse_args() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut model: Option<String> = None;
    let mut context_size: usize = 0;
    let mut tokens: usize = 10;
    let mut warmup: usize = 3;
    let mut threads: Option<usize> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => {
                model = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --model".to_string())?,
                );
            }
            "--context" | "-c" => {
                let v = args
                    .next()
                    .ok_or_else(|| "missing value for --context".to_string())?;
                context_size = v
                    .parse()
                    .map_err(|_| format!("invalid context size: {v}"))?;
            }
            "--tokens" | "-n" => {
                let v = args
                    .next()
                    .ok_or_else(|| "missing value for --tokens".to_string())?;
                tokens = v.parse().map_err(|_| format!("invalid token count: {v}"))?;
                if tokens == 0 {
                    return Err("--tokens must be > 0".to_string());
                }
            }
            "--warmup" | "-w" => {
                let v = args
                    .next()
                    .ok_or_else(|| "missing value for --warmup".to_string())?;
                warmup = v.parse().map_err(|_| format!("invalid warmup count: {v}"))?;
            }
            "--threads" | "-t" => {
                let v = args
                    .next()
                    .ok_or_else(|| "missing value for --threads".to_string())?;
                threads = Some(v.parse().map_err(|_| format!("invalid thread count: {v}"))?);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let model = model.ok_or_else(|| "missing required --model <path>".to_string())?;
    Ok(Options { model, context_size, tokens, warmup, threads })
}

fn print_help() {
    println!("Usage: cargo run --example calc_bench -- --model <model.gguf> [options]");
    println!();
    println!("Options:");
    println!("  -m, --model <path>       GGUF model file (required)");
    println!("  -c, --context <n>        Override context/sequence length");
    println!("  -n, --tokens <n>         Number of tokens to benchmark (default: 10)");
    println!("  -w, --warmup <n>         Warmup forward passes before measuring (default: 3)");
    println!("  -t, --threads <n>        Number of threads (default: auto)");
    println!();
    println!("Runs the calculation engine without sampling or I/O and reports");
    println!("wall-clock throughput plus per-phase timing (attention, FFN, matmul, MoE).");
}
