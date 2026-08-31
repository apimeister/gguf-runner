// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime/SpeakerRuntime/SpeakerIndexRuntime and does not re-export
// binary-only code.
#![allow(dead_code)]

mod agent;
mod audio_batch;
mod batch;
pub mod embed;
mod events;
mod generation;
mod image_batch;
pub(crate) mod prefill_cache;
mod repl;
pub mod speaker;
mod speaker_index;

use crate::cli::CliOperationMode;
use crate::cli::CliOptions;
use crate::engine::profiling::{print_profile_report, profiling_reset, set_profiling_enabled};
use crate::engine::switches::{
    KvCacheMode, RuntimeSwitchConfig, init_runtime_config, kv_cache_mode, par_attn_min_heads,
    par_matmul_chunk_rows, par_matmul_min_rows, par_qwen3next_min_heads,
};
#[cfg(target_arch = "aarch64")]
use crate::engine::switches::{aarch64_matmul_prefetch_rows, use_aarch64_bf16};
use crate::engine::types::{ContentPart, GenerationRequest, MediaRef};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_IMAGES: usize = 10;
const MAX_VIDEOS: usize = 10;
const MAX_AUDIOS: usize = 10;

const MAX_IMAGE_BYTES: u64 = 50 * 1024 * 1024;
const MAX_VIDEO_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_AUDIO_BYTES: u64 = 1024 * 1024 * 1024;

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];
const VIDEO_EXTENSIONS: &[&str] = &["mp4"];
const REPL_COMMANDS: [&str; 11] = [
    "help",
    "model",
    "image",
    "images",
    "clear-images",
    "clear",
    "doc",
    "docs",
    "clear-docs",
    "exit",
    "quit",
];

#[cfg(test)]
static CWD_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn map_kv_cache_mode(mode: Option<crate::cli::CliKvCacheMode>) -> Option<KvCacheMode> {
    mode.map(|v| match v {
        crate::cli::CliKvCacheMode::Q8 => KvCacheMode::Q8,
        crate::cli::CliKvCacheMode::Turbo => KvCacheMode::Turbo,
    })
}

fn print_cpu_features() {
    fn yn(v: bool) -> &'static str {
        if v { "yes" } else { "no " }
    }

    println!("Architecture: {}", std::env::consts::ARCH);
    println!();

    #[cfg(target_arch = "aarch64")]
    {
        let features: &[(&str, &str, bool)] = &[
            (
                "neon",
                "ARMv8-A (baseline)",
                std::arch::is_aarch64_feature_detected!("neon"),
            ),
            (
                "dotprod",
                "ARMv8.2-A",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            (
                "fp16",
                "ARMv8.2-A",
                std::arch::is_aarch64_feature_detected!("fp16"),
            ),
            (
                "bf16",
                "ARMv8.6-A",
                std::arch::is_aarch64_feature_detected!("bf16"),
            ),
            (
                "i8mm",
                "ARMv8.6-A",
                std::arch::is_aarch64_feature_detected!("i8mm"),
            ),
            (
                "sve",
                "ARMv8.4-A (opt-in)",
                std::arch::is_aarch64_feature_detected!("sve"),
            ),
            (
                "sve2",
                "ARMv9-A",
                std::arch::is_aarch64_feature_detected!("sve2"),
            ),
        ];
        println!("{:<10}  {:<20}  {:>8}", "feature", "ISA", "runtime");
        println!("{}", "-".repeat(44));
        for (name, isa, runtime) in features {
            println!("{:<10}  {:<20}  {:>8}", name, isa, yn(*runtime));
        }
        println!();
        println!("gguf-runner kernels (aarch64):");
        println!("  NEON matmul Q4/Q5/Q6-K MR4:  always enabled");
        println!("  FCVTL fp16 loads:             always enabled (base AArch64)");
        println!("  USHLL/SHL bf16 loads:         always enabled (base AArch64)");
        println!(
            "  BFDOT bf16 batch matmul:       runtime={}",
            yn(std::arch::is_aarch64_feature_detected!("bf16"))
        );
        println!(
            "  dotprod Q8_0:                 runtime={}",
            yn(std::arch::is_aarch64_feature_detected!("dotprod"))
        );
        println!(
            "  i8mm Q8_0 MR2 (SMMLA):       runtime={}",
            yn(std::arch::is_aarch64_feature_detected!("i8mm"))
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        let features: &[(&str, &str, bool)] = &[
            (
                "sse4.1",
                "Intel Penryn 2007",
                std::arch::is_x86_feature_detected!("sse4.1"),
            ),
            (
                "avx",
                "Intel Sandy Br. 2011",
                std::arch::is_x86_feature_detected!("avx"),
            ),
            (
                "avx2",
                "Intel Haswell 2013",
                std::arch::is_x86_feature_detected!("avx2"),
            ),
            (
                "fma",
                "Intel Haswell 2013",
                std::arch::is_x86_feature_detected!("fma"),
            ),
            (
                "f16c",
                "Intel Ivy Br. 2012",
                std::arch::is_x86_feature_detected!("f16c"),
            ),
            (
                "avxvnni",
                "Intel Alder Lk. 2021",
                std::arch::is_x86_feature_detected!("avxvnni"),
            ),
            (
                "avx512f",
                "Intel Skylake-X 2017",
                std::arch::is_x86_feature_detected!("avx512f"),
            ),
            (
                "avx512vnni",
                "Intel Cascade Lk. 2019",
                std::arch::is_x86_feature_detected!("avx512vnni"),
            ),
            (
                "avx512bf16",
                "Intel Cooper Lk. 2020",
                std::arch::is_x86_feature_detected!("avx512bf16"),
            ),
            (
                "avx512vl",
                "Intel Skylake-X 2017",
                std::arch::is_x86_feature_detected!("avx512vl"),
            ),
        ];
        println!("{:<12}  {:<24}  {:>8}", "feature", "ISA", "runtime");
        println!("{}", "-".repeat(50));
        for (name, isa, runtime) in features {
            println!("{:<12}  {:<24}  {:>8}", name, isa, yn(*runtime));
        }
        println!();
        println!("gguf-runner kernels (x86_64):");
        println!(
            "  AVX2+FMA matmul Q4/Q5/Q6-K:  runtime={}",
            yn(std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"))
        );
        println!(
            "  F16C fp16 loads:              runtime={}",
            yn(std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("f16c"))
        );
        println!(
            "  AVX2 bf16 loads:              runtime={}",
            yn(std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"))
        );
        println!(
            "  AVX-VNNI Q8_0:                runtime={}",
            yn(std::arch::is_x86_feature_detected!("avxvnni"))
        );
        println!(
            "  AVX-512VNNI Q8_0:             runtime={}",
            yn(std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vl"))
        );
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    println!("No architecture-specific features detected for this target.");
}

pub(crate) fn run() -> Result<(), String> {
    let cli = CliOptions::parse()?;

    if cli.show_features {
        print_cpu_features();
        return Ok(());
    }

    let runtime_switch_config = RuntimeSwitchConfig {
        par_matmul_min_rows: cli.par_matmul_min_rows,
        par_matmul_chunk_rows: cli.par_matmul_chunk_rows,
        #[cfg(target_arch = "aarch64")]
        aarch64_matmul_prefetch_rows: cli.aarch64_matmul_prefetch_rows,
        par_attn_min_heads: cli.par_attn_min_heads,
        par_qwen3next_min_heads: cli.par_qwen3next_min_heads,
        #[cfg(target_arch = "aarch64")]
        aarch64_dotprod_q8: cli.aarch64_dotprod_q8,
        #[cfg(target_arch = "aarch64")]
        aarch64_qk_mr4: cli.aarch64_qk_mr4,
        #[cfg(target_arch = "aarch64")]
        aarch64_i8mm: cli.aarch64_i8mm,
        #[cfg(target_arch = "aarch64")]
        aarch64_bf16: cli.aarch64_bf16,
        #[cfg(target_arch = "x86_64")]
        x86_avx2: cli.x86_avx2,
        #[cfg(target_arch = "x86_64")]
        x86_f16c: cli.x86_f16c,
        #[cfg(target_arch = "x86_64")]
        x86_qk_mr4: cli.x86_qk_mr4,
        #[cfg(target_arch = "x86_64")]
        x86_avxvnni: cli.x86_avxvnni,
        #[cfg(target_arch = "x86_64")]
        x86_avx512vnni_q8: cli.x86_avx512vnni_q8,
        layer_debug: cli.layer_debug,
        layer_debug_pos: cli.layer_debug_pos,
        kv_cache_mode: map_kv_cache_mode(cli.kv_cache_mode),
    };
    init_runtime_config(&runtime_switch_config);
    let run_started = Instant::now();

    set_profiling_enabled(cli.profiling);
    if cli.profiling {
        profiling_reset();
    }

    if cli.debug && cli.mode == CliOperationMode::Oneshot {
        for line in collect_debug_banner_lines(&cli) {
            eprintln!("{line}");
        }
    }

    if cli.speaker_action.is_some() {
        if let Some(thread_count) = cli.threads {
            crate::engine::runtime::configure_rayon_threads(thread_count, cli.debug);
        }
        speaker::run_cli(&cli)?;
        if cli.profiling {
            print_profile_report();
        }
        if cli.show_timings {
            eprintln!(
                "overall runtime: {:.3}s",
                run_started.elapsed().as_secs_f64()
            );
        }
        return Ok(());
    }

    // --rag-build: build (or rebuild) the RAG index and exit without generating.
    if cli.rag_build {
        return run_rag_build_mode(&cli);
    }

    match cli.mode {
        CliOperationMode::Oneshot => {
            let mut audio_batch_manifest = cli
                .audio_batch
                .as_deref()
                .map(Path::new)
                .map(audio_batch::open_and_validate_manifest)
                .transpose()?;
            let mut image_batch_manifest = cli
                .image_batch
                .as_deref()
                .map(Path::new)
                .map(image_batch::open_and_validate_manifest)
                .transpose()?;
            let mut runtime = generation::ModelRuntime::load(&cli)?;
            if let Some(out) = &cli.render_prefill_cache {
                let blob = runtime.render_prefill_cache_blob(&cli.system_prompt)?;
                fs::write(out, &blob).map_err(|e| format!("write prefill cache '{out}': {e}"))?;
                eprintln!("prefill cache written: {out} ({} bytes)", blob.len());
            } else {
                if let Some(f) = &cli.prefill_cache {
                    let bytes =
                        fs::read(f).map_err(|e| format!("read prefill cache '{f}': {e}"))?;
                    runtime.set_prefill_cache_bytes(&bytes)?;
                }
                if let Some(manifest) = audio_batch_manifest.as_mut() {
                    run_audio_batch_mode(&mut runtime, manifest)?;
                } else if let Some(manifest) = image_batch_manifest.as_mut() {
                    run_image_batch_mode(&mut runtime, manifest, &cli.system_prompt)?;
                } else {
                    run_oneshot_mode(&mut runtime, &cli)?;
                }
            }
        }
        CliOperationMode::Repl => {
            run_repl_mode(&cli)?;
        }
    }

    if cli.profiling {
        print_profile_report();
    }
    if cli.show_timings {
        eprintln!(
            "overall runtime: {:.3}s",
            run_started.elapsed().as_secs_f64()
        );
    }

    Ok(())
}

fn run_image_batch_mode(
    runtime: &mut generation::ModelRuntime,
    manifest: &mut image_batch::ImageBatchManifest,
    default_system_prompt: &str,
) -> Result<(), String> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let summary = image_batch::run_manifest(
        manifest,
        &mut output,
        image_batch::IMAGE_BATCH_CHUNK_SIZE,
        |jobs| {
            let mut results = std::iter::repeat_with(|| None)
                .take(jobs.len())
                .collect::<Vec<Option<Result<String, String>>>>();
            let mut valid_job_indices = Vec::new();
            let mut valid_paths = Vec::new();
            for (index, job) in jobs.iter().enumerate() {
                let validation = job
                    .resolved_path
                    .to_str()
                    .ok_or_else(|| "image batch path contains non-UTF8 characters".to_string())
                    .and_then(|path| {
                        validate_media_paths(
                            &[path.to_string()],
                            "image",
                            1,
                            MAX_IMAGE_BYTES,
                            Some(IMAGE_EXTENSIONS),
                        )
                    });
                match validation {
                    Ok(mut paths) => {
                        valid_job_indices.push(index);
                        valid_paths.push(paths.remove(0));
                    }
                    Err(error) => results[index] = Some(Err(error)),
                }
            }

            let preencoded = runtime.preencode_image_batch(&valid_paths);
            for ((job_index, path), preencode_result) in valid_job_indices
                .into_iter()
                .zip(valid_paths)
                .zip(preencoded)
            {
                let job = &jobs[job_index];
                results[job_index] = Some(match preencode_result {
                    Ok(()) => runtime
                        .generate_text_with_images(
                            &job.prompt,
                            job.system_prompt
                                .as_deref()
                                .unwrap_or(default_system_prompt),
                            std::slice::from_ref(&path),
                            false,
                        )
                        .and_then(require_image_response_text),
                    Err(error) => Err(error),
                });
            }

            results
                .into_iter()
                .map(|result| {
                    result
                        .unwrap_or_else(|| Err("image batch did not produce a result".to_string()))
                })
                .collect()
        },
    )?;
    if summary.failed > 0 {
        return Err(format!(
            "image batch completed with {} succeeded and {} failed item(s)",
            summary.succeeded, summary.failed
        ));
    }
    Ok(())
}

fn run_audio_batch_mode(
    runtime: &mut generation::ModelRuntime,
    manifest: &mut audio_batch::AudioBatchManifest,
) -> Result<(), String> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let summary = audio_batch::run_manifest(manifest, &mut output, |path, context, language| {
        let path = path
            .to_str()
            .ok_or_else(|| "audio batch path contains non-UTF8 characters".to_string())?;
        let validated =
            validate_media_paths(&[path.to_string()], "audio", 1, MAX_AUDIO_BYTES, None)?;
        runtime.transcribe_audio(&validated[0], context, language)
    })?;
    if summary.failed > 0 {
        return Err(format!(
            "audio batch completed with {} succeeded and {} failed item(s)",
            summary.succeeded, summary.failed
        ));
    }
    Ok(())
}

fn run_oneshot_mode(
    runtime: &mut generation::ModelRuntime,
    cli: &CliOptions,
) -> Result<(), String> {
    if cli.tools_enabled {
        if !cli.images.is_empty() || !cli.videos.is_empty() || !cli.audios.is_empty() {
            return Err(
                "`--image/--video/--audio` are not supported together with tools mode yet"
                    .to_string(),
            );
        }
        return agent::run_agent_loop(runtime, cli, &cli.prompt);
    }

    let images = validate_media_paths(
        &cli.images,
        "image",
        MAX_IMAGES,
        MAX_IMAGE_BYTES,
        Some(IMAGE_EXTENSIONS),
    )?;
    let videos = validate_media_paths(
        &cli.videos,
        "video",
        MAX_VIDEOS,
        MAX_VIDEO_BYTES,
        Some(VIDEO_EXTENSIONS),
    )?;
    let audios = validate_media_paths(&cli.audios, "audio", MAX_AUDIOS, MAX_AUDIO_BYTES, None)?;
    if !audios.is_empty() {
        if !images.is_empty() || !videos.is_empty() {
            return Err(
                "offline audio transcription does not support mixed image/video inputs".to_string(),
            );
        }
        if audios.len() != 1 {
            return Err(format!(
                "offline audio transcription accepts exactly one --audio path, got {}; use the embedded API to reuse one loaded runtime across multiple files",
                audios.len()
            ));
        }
        let result =
            runtime.transcribe_audio(&audios[0], &cli.prompt, cli.audio_language.as_deref())?;
        println!("{}", result.transcript);
        return Ok(());
    }
    if !images.is_empty() && videos.is_empty() {
        let output =
            runtime.generate_text_with_images(&cli.prompt, &cli.system_prompt, &images, true)?;
        require_image_response_text(output)?;
        return Ok(());
    }
    let request = build_generation_request(&cli.prompt, &cli.system_prompt, images, videos, audios);
    let _ = runtime.generate_request(&request, true)?;
    Ok(())
}

fn require_image_response_text(output: String) -> Result<String, String> {
    if output.trim().is_empty() {
        Err("image generation completed without response text".to_string())
    } else {
        Ok(output)
    }
}

fn run_rag_build_mode(cli: &CliOptions) -> Result<(), String> {
    let src = cli
        .rag_source
        .as_deref()
        .ok_or("--rag-build requires --rag-source <wiki-dir>".to_string())?;
    let idx_path = cli
        .rag_index
        .as_deref()
        .ok_or("--rag-build requires --rag-index <output.ragidx>".to_string())?;
    let enc_path = cli
        .rag_encoder
        .clone()
        .or_else(|| crate::rag::encoder::discover_embedding_sidecar(&cli.model))
        .ok_or(
            "--rag-build requires --rag-encoder <embed.gguf> or an auto-discoverable embed*.gguf next to the model".to_string(),
        )?;

    eprintln!("RAG build: encoder='{enc_path}', source='{src}', index='{idx_path}'");
    let mut encoder = crate::rag::DocumentEncoder::load(&enc_path, cli.debug)?;
    let src_path = std::path::Path::new(src);
    let index = crate::rag::RagIndex::build_from_dir(
        src_path,
        &mut encoder,
        cli.rag_max_chars_per_chunk,
        cli.rag_max_tokens_per_chunk,
        None,
        cli.debug || cli.profiling,
    )?;
    eprintln!("RAG build: {} chunks embedded, saving…", index.len());
    index.save(std::path::Path::new(idx_path))?;
    eprintln!("RAG build: index saved to '{idx_path}'");
    if cli.profiling {
        print_profile_report();
    }
    Ok(())
}

fn run_repl_mode(cli: &CliOptions) -> Result<(), String> {
    if !cli.images.is_empty() || !cli.videos.is_empty() || !cli.audios.is_empty() {
        return Err("`--image/--video/--audio` are not supported in repl mode yet".to_string());
    }
    repl::run(cli)
}

pub(crate) enum ReplCommandAction {
    Exit,
    Messages(Vec<String>),
    AttachImage(String),
    ListImages,
    ClearImages,
    ClearState,
    LoadDocSource(String),
    ClearDocs,
    DocStatus,
    ModelPrompt(String),
}

pub(crate) fn handle_repl_command(cli: &CliOptions, input: &str) -> ReplCommandAction {
    let Some(raw_cmd) = input.strip_prefix('/') else {
        return ReplCommandAction::ModelPrompt(input.to_string());
    };
    let trimmed = raw_cmd.trim();
    let cmd = trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    let rest = trimmed
        .strip_prefix(cmd.as_str())
        .map(str::trim)
        .unwrap_or("");
    match cmd.as_str() {
        "" => {
            ReplCommandAction::Messages(vec!["Empty command. Type /help for commands.".to_string()])
        }
        "help" => ReplCommandAction::Messages(repl_help_lines()),
        "model" => ReplCommandAction::Messages(vec![cli.model.clone()]),
        "image" => {
            if rest.is_empty() {
                ReplCommandAction::Messages(vec!["Usage: /image <path-to-image>".to_string()])
            } else {
                ReplCommandAction::AttachImage(rest.to_string())
            }
        }
        "images" => ReplCommandAction::ListImages,
        "clear-images" => ReplCommandAction::ClearImages,
        "clear" => ReplCommandAction::ClearState,
        "doc" => {
            if rest.is_empty() {
                ReplCommandAction::Messages(vec![
                    "Usage: /doc <path-to-wiki-directory>".to_string(),
                ])
            } else {
                ReplCommandAction::LoadDocSource(rest.to_string())
            }
        }
        "docs" => ReplCommandAction::DocStatus,
        "clear-docs" => ReplCommandAction::ClearDocs,
        "exit" | "quit" => ReplCommandAction::Exit,
        _ => ReplCommandAction::Messages(vec![format!(
            "Unknown command '/{}'. Type /help for commands.",
            cmd
        )]),
    }
}

fn expand_repl_tab_completion(input: &str) -> String {
    let Some(raw_cmd) = input.strip_prefix('/') else {
        return input.to_string();
    };
    let sanitized = raw_cmd.replace('\t', "");
    let cmd = sanitized.split_whitespace().next().unwrap_or("");
    if cmd.is_empty() {
        return "/".to_string();
    }
    if let Some((typed_cmd, rest)) = split_repl_command_and_rest(&sanitized)
        && matches!(typed_cmd, "image" | "doc")
        && rest.starts_with(char::is_whitespace)
    {
        if let Some(completed) = complete_filesystem_path(rest.trim_start()) {
            return format!("/{typed_cmd} {completed}");
        }
        return format!("/{typed_cmd} {}", rest.trim_start());
    }

    let matches = REPL_COMMANDS
        .iter()
        .copied()
        .filter(|candidate| candidate.starts_with(cmd))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return format!("/{sanitized}");
    }
    let shared = longest_common_prefix(&matches);
    if shared.len() > cmd.len() {
        format!("/{shared}")
    } else if matches.len() == 1 {
        format!("/{}", matches[0])
    } else {
        format!("/{sanitized}")
    }
}

fn split_repl_command_and_rest(input: &str) -> Option<(&str, &str)> {
    let trimmed_start = input.trim_start();
    let cmd_end = trimmed_start.find(char::is_whitespace)?;
    let cmd = &trimmed_start[..cmd_end];
    let rest = &trimmed_start[cmd_end..];
    Some((cmd, rest))
}

fn longest_common_prefix(items: &[&str]) -> String {
    let Some(first) = items.first() else {
        return String::new();
    };
    let mut prefix = (*first).to_string();
    for item in &items[1..] {
        while !item.starts_with(&prefix) {
            if prefix.is_empty() {
                return prefix;
            }
            prefix.pop();
        }
    }
    prefix
}

fn complete_filesystem_path(fragment: &str) -> Option<String> {
    let (dir_part, name_prefix) = match fragment.rsplit_once('/') {
        Some((dir, prefix)) => (format!("{dir}/"), prefix),
        None => (String::new(), fragment),
    };
    let dir_path = if dir_part.is_empty() {
        PathBuf::from(".")
    } else {
        PathBuf::from(&dir_part)
    };
    let mut matches = fs::read_dir(&dir_path)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with(name_prefix) {
                return None;
            }
            let is_dir = entry
                .file_type()
                .ok()
                .map(|ft| ft.is_dir())
                .unwrap_or(false);
            Some((name, is_dir))
        })
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return None;
    }
    matches.sort_by(|a, b| a.0.cmp(&b.0));
    if matches.len() == 1 {
        let (name, is_dir) = &matches[0];
        let suffix = if *is_dir { "/" } else { "" };
        return Some(format!("{dir_part}{name}{suffix}"));
    }
    let names = matches
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    let shared = longest_common_prefix(&names);
    if shared.len() > name_prefix.len() {
        Some(format!("{dir_part}{shared}"))
    } else {
        None
    }
}

pub(crate) fn repl_help_lines() -> Vec<String> {
    vec![
        "REPL commands:".to_string(),
        "  /help          Show this help".to_string(),
        "  /model         Show active model path".to_string(),
        "  /image <path>  Attach an image to the conversation".to_string(),
        "  /images        List active image attachments".to_string(),
        "  /clear-images  Remove all active image attachments".to_string(),
        "  /clear         Reset chat history and active attachments".to_string(),
        "  /doc <dir>     Load knowledge from a wiki directory".to_string(),
        "  /docs          Show active knowledge status".to_string(),
        "  /clear-docs    Drop active knowledge index".to_string(),
        "  /exit          Exit REPL".to_string(),
        "  /quit          Exit REPL".to_string(),
        "  /e<Tab> expands to /exit".to_string(),
    ]
}

pub(crate) fn validate_repl_image_path(path: &str) -> Result<String, String> {
    let validated = validate_media_paths(
        &[path.to_string()],
        "image",
        1,
        MAX_IMAGE_BYTES,
        Some(IMAGE_EXTENSIONS),
    )?;
    let canonical = PathBuf::from(&validated[0])
        .canonicalize()
        .map_err(|e| format!("cannot resolve image path '{}': {e}", validated[0]))?;
    Ok(canonical.to_string_lossy().to_string())
}

pub(crate) fn collect_debug_banner_lines(cli: &CliOptions) -> Vec<String> {
    let mut lines = Vec::new();
    let workspace_root = match cli.tool_root.as_deref() {
        Some(raw) => PathBuf::from(raw),
        None => match std::env::current_dir() {
            Ok(dir) => dir,
            Err(e) => {
                lines.push(format!(
                    "Tool workspace root: <failed to read current directory: {e}>"
                ));
                PathBuf::from(".")
            }
        },
    };
    match workspace_root.canonicalize() {
        Ok(root) => lines.push(format!("Tool workspace root: {}", root.display())),
        Err(e) => lines.push(format!(
            "Tool workspace root: <cannot resolve '{}': {e}>",
            workspace_root.display()
        )),
    }
    let enabled_tools = cli.tool_enablement.enabled_tool_names();
    if enabled_tools.is_empty() {
        lines.push("Allowed tools: none".to_string());
    } else {
        lines.push(format!("Allowed tools: {}", enabled_tools.join(", ")));
    }
    lines.push(format!(
        "Parallel thresholds: matmul_min_rows={}, matmul_chunk_rows={}, attn_min_heads={}, qwen3next_min_heads={}",
        par_matmul_min_rows(),
        par_matmul_chunk_rows(),
        par_attn_min_heads(),
        par_qwen3next_min_heads()
    ));
    #[cfg(target_arch = "aarch64")]
    lines.push(format!(
        "AArch64 kernels: matmul_prefetch_rows={}, bf16_bfdot={}",
        aarch64_matmul_prefetch_rows(),
        use_aarch64_bf16()
    ));
    lines.push(format!("KV cache mode request: {:?}", kv_cache_mode()));
    lines
}

fn validate_media_paths(
    paths: &[String],
    kind: &str,
    max_count: usize,
    max_bytes: u64,
    allowed_extensions: Option<&[&str]>,
) -> Result<Vec<String>, String> {
    if paths.len() > max_count {
        return Err(format!(
            "too many {kind} inputs: got {}, max allowed {max_count}",
            paths.len()
        ));
    }
    let mut validated = Vec::with_capacity(paths.len());
    for path in paths {
        let meta = fs::metadata(path).map_err(|e| format!("cannot read {kind} '{path}': {e}"))?;
        if !meta.is_file() {
            return Err(format!("{kind} path is not a file: {path}"));
        }
        if meta.len() == 0 {
            return Err(format!("{kind} file is empty: {path}"));
        }
        if meta.len() > max_bytes {
            return Err(format!(
                "{kind} file exceeds max size ({} bytes): {path}",
                max_bytes
            ));
        }
        if let Some(extensions) = allowed_extensions {
            let ext = Path::new(path)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !extensions.iter().any(|allowed| *allowed == ext) {
                let allowed_list = extensions.join(", ");
                return Err(format!(
                    "unsupported {kind} extension for '{path}'; allowed: {allowed_list}"
                ));
            }
        }
        validated.push(path.clone());
    }
    Ok(validated)
}

fn build_generation_request(
    prompt: &str,
    system_prompt: &str,
    images: Vec<String>,
    videos: Vec<String>,
    audios: Vec<String>,
) -> GenerationRequest {
    let mut parts = Vec::with_capacity(1 + images.len() + videos.len() + audios.len());
    for path in images {
        parts.push(ContentPart::Image(MediaRef { path }));
    }
    for path in videos {
        parts.push(ContentPart::Video(MediaRef { path }));
    }
    for path in audios {
        parts.push(ContentPart::Audio(MediaRef { path }));
    }
    parts.push(ContentPart::Text(prompt.to_string()));
    GenerationRequest {
        system_prompt: system_prompt.to_string(),
        parts,
        include_empty_system_prompt: false,
        assistant_prefill: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{CWD_TEST_LOCK, expand_repl_tab_completion, require_image_response_text};
    use std::fs;

    #[test]
    fn repl_tab_completion_completes_exit() {
        assert_eq!(expand_repl_tab_completion("/e\t"), "/exit");
    }

    #[test]
    fn repl_tab_completion_completes_model() {
        assert_eq!(expand_repl_tab_completion("/mo\t"), "/model");
    }

    #[test]
    fn repl_tab_completion_extends_shared_prefix_for_ambiguous_commands() {
        assert_eq!(expand_repl_tab_completion("/ima"), "/image");
        assert_eq!(expand_repl_tab_completion("/cl"), "/clear");
    }

    #[test]
    fn image_generation_rejects_silent_success() {
        assert!(require_image_response_text(String::new()).is_err());
        assert!(require_image_response_text(" \n".to_string()).is_err());
        assert_eq!(
            require_image_response_text("chip name".to_string()).unwrap(),
            "chip name"
        );
    }

    #[test]
    fn repl_tab_completion_ambiguous_prefix_keeps_slash_only() {
        assert_eq!(expand_repl_tab_completion("/\t"), "/");
    }

    #[test]
    fn repl_tab_completion_completes_image_path() {
        let _cwd_guard = CWD_TEST_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let test_dir = std::env::temp_dir().join("gguf-runner-repl-tab");
        let _ = fs::remove_dir_all(&test_dir);
        fs::create_dir_all(&test_dir).expect("create temp dir");
        fs::write(test_dir.join("sample-image.jpg"), "x").expect("write image");
        let original_dir = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&test_dir).expect("set cwd");
        let completed = expand_repl_tab_completion("/image samp");
        std::env::set_current_dir(original_dir).expect("restore cwd");
        let _ = fs::remove_dir_all(&test_dir);
        assert_eq!(completed, "/image sample-image.jpg");
    }
}
