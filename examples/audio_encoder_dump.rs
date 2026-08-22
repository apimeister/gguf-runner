#![allow(dead_code)]
#![allow(unused_imports)]

#[path = "../src/engine/mod.rs"]
mod engine;
#[path = "../src/vendors/mod.rs"]
mod vendors;

use engine::audio::{PreparedAudioFeatureWindow, prepare_audios_for_multimodal};
use engine::io::{get_gguf_int_from_map, parse_gguf_file};
use engine::multimodal::{AudioEncoder, build_audio_encoder_from_mmproj};
use engine::types::AudioEncoderBackend;

const DEFAULT_SYNTHETIC_FRAMES: usize = 100;
const DEFAULT_DUMP_VALUES: usize = 3;

#[derive(Clone, Copy, Debug)]
enum SyntheticInput {
    Zero,
    Half,
    One,
    Alternating,
}

struct Options {
    mmproj: String,
    audio: Option<String>,
    synthetic: Option<SyntheticInput>,
    conv_input: Option<String>,
    frames: usize,
    dump_values: usize,
    dump_dir: Option<String>,
    debug: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let mmproj = parse_gguf_file(&options.mmproj, options.debug)?;
    let text_embedding_dim = usize::try_from(get_gguf_int_from_map(
        &mmproj.kv,
        "clip.audio.projection_dim",
        0,
    ))
    .ok()
    .filter(|value| *value > 0)
    .ok_or_else(|| {
        "mmproj is missing required positive metadata 'clip.audio.projection_dim'".to_string()
    })?;
    let encoder =
        build_audio_encoder_from_mmproj(AudioEncoderBackend::Qwen3Asr, text_embedding_dim, mmproj)?;

    println!("backend=qwen3_asr");
    println!("contract={}", encoder.contract_summary());
    println!("text_embedding_dim={text_embedding_dim}");

    if let Some(conv_input) = options.conv_input.as_deref() {
        run_conv_input(
            &encoder,
            conv_input,
            options.dump_values,
            options.dump_dir.as_deref(),
        )
    } else if let Some(audio_path) = options.audio.as_deref() {
        run_audio_file(
            &encoder,
            audio_path,
            options.dump_values,
            options.dump_dir.as_deref(),
        )
    } else {
        let synthetic = options
            .synthetic
            .ok_or_else(|| "internal error: no input selected".to_string())?;
        run_synthetic(
            &encoder,
            synthetic,
            options.frames,
            options.dump_values,
            options.dump_dir.as_deref(),
        )
    }
}

fn run_conv_input(
    encoder: &AudioEncoder,
    input_path: &str,
    dump_values: usize,
    dump_dir: Option<&str>,
) -> Result<(), String> {
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("cannot read convolution fixture '{input_path}': {error}"))?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "convolution fixture '{input_path}' has {} bytes; expected non-empty little-endian f32 data",
            bytes.len()
        ));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    println!(
        "input=conv_fixture path={input_path} values={}",
        values.len()
    );
    let (layer_outputs, transformer, projected) =
        encoder.encode_from_conv_output_with_layers(values)?;
    for (layer, values) in layer_outputs.iter().enumerate() {
        dump_stage_values(dump_dir, &format!("layer_out_{layer}"), 0, values)?;
    }
    dump_stage_values(
        dump_dir,
        "after_transformer",
        0,
        &transformer.data_token_major,
    )?;
    dump_values_summary(
        "after_transformer",
        0,
        transformer.token_count,
        transformer.embedding_dim,
        &transformer.data_token_major,
        dump_values,
    );
    let embedding_dim = projected.tokens.first().map_or(0, Vec::len);
    let token_count = projected.tokens.len();
    let flattened = projected.tokens.into_iter().flatten().collect::<Vec<_>>();
    dump_stage_values(dump_dir, "projected", 0, &flattened)?;
    dump_values_summary(
        "projected",
        0,
        token_count,
        embedding_dim,
        &flattened,
        dump_values,
    );
    Ok(())
}

fn run_audio_file(
    encoder: &AudioEncoder,
    audio_path: &str,
    dump_values: usize,
    dump_dir: Option<&str>,
) -> Result<(), String> {
    let paths = [audio_path.to_string()];
    let prepared = prepare_audios_for_multimodal(
        &paths,
        vendors::audio_preprocess_config(AudioEncoderBackend::Qwen3Asr),
    )?;
    let audio = &prepared[0];
    println!("input=audio path={}", audio.path);
    println!(
        "source_sample_rate={} source_channels={} sample_rate={} samples={} windows={}",
        audio.source_sample_rate,
        audio.source_channels,
        audio.sample_rate,
        audio.total_samples,
        audio.feature_windows.len()
    );
    for (window_index, window) in audio.feature_windows.iter().enumerate() {
        println!(
            "window={} start_frame={} valid_frames={} padded_frames={} mel_bins={}",
            window_index,
            window.start_frame,
            window.valid_frames,
            window.padded_frames,
            window.mel_bins
        );
        encode_and_dump_stages(encoder, window, window_index, dump_values, dump_dir)?;
    }
    Ok(())
}

fn run_synthetic(
    encoder: &AudioEncoder,
    synthetic: SyntheticInput,
    frames: usize,
    dump_values: usize,
    dump_dir: Option<&str>,
) -> Result<(), String> {
    if frames == 0 || !frames.is_multiple_of(DEFAULT_SYNTHETIC_FRAMES) {
        return Err(format!(
            "--frames must be a positive multiple of {DEFAULT_SYNTHETIC_FRAMES}"
        ));
    }
    let mel_bins = 128usize;
    let frame_values = (0..frames)
        .map(|frame| match synthetic {
            SyntheticInput::Zero => 0.0,
            SyntheticInput::Half => 0.5,
            SyntheticInput::One => 1.0,
            SyntheticInput::Alternating => {
                if frame % 2 == 0 {
                    1.0
                } else {
                    0.0
                }
            }
        })
        .collect::<Vec<_>>();
    let mut data_mel_major = Vec::with_capacity(mel_bins * frames);
    for _ in 0..mel_bins {
        data_mel_major.extend_from_slice(&frame_values);
    }
    let window = PreparedAudioFeatureWindow {
        start_frame: 0,
        valid_frames: frames,
        padded_frames: frames,
        mel_bins,
        data_mel_major,
    };
    println!(
        "input=synthetic kind={} frames={} mel_bins={mel_bins}",
        synthetic_name(synthetic),
        frames
    );
    encode_and_dump_stages(encoder, &window, 0, dump_values, dump_dir)
}

fn encode_and_dump_stages(
    encoder: &AudioEncoder,
    window: &PreparedAudioFeatureWindow,
    window_index: usize,
    dump_values: usize,
    dump_dir: Option<&str>,
) -> Result<(), String> {
    let conv = encoder.encode_conv_frontend(window)?;
    dump_stage_values(
        dump_dir,
        "after_conv_out",
        window_index,
        &conv.data_token_major,
    )?;
    dump_values_summary(
        "after_conv_out",
        window_index,
        conv.token_count,
        conv.embedding_dim,
        &conv.data_token_major,
        dump_values,
    );

    let transformer = encoder.encode_transformer_frontend(window)?;
    dump_stage_values(
        dump_dir,
        "after_transformer",
        window_index,
        &transformer.data_token_major,
    )?;
    dump_values_summary(
        "after_transformer",
        window_index,
        transformer.token_count,
        transformer.embedding_dim,
        &transformer.data_token_major,
        dump_values,
    );

    let projected = encoder.encode_feature_window(window)?;
    let embedding_dim = projected.tokens.first().map_or(0, Vec::len);
    let token_count = projected.tokens.len();
    let flattened = projected.tokens.into_iter().flatten().collect::<Vec<_>>();
    dump_stage_values(dump_dir, "projected", window_index, &flattened)?;
    dump_values_summary(
        "projected",
        window_index,
        token_count,
        embedding_dim,
        &flattened,
        dump_values,
    );
    Ok(())
}

fn dump_stage_values(
    dump_dir: Option<&str>,
    stage: &str,
    window_index: usize,
    values: &[f32],
) -> Result<(), String> {
    use std::io::Write;

    let Some(dump_dir) = dump_dir else {
        return Ok(());
    };
    std::fs::create_dir_all(dump_dir)
        .map_err(|error| format!("cannot create dump directory '{dump_dir}': {error}"))?;
    let path = std::path::Path::new(dump_dir).join(format!("window-{window_index}-{stage}.f32"));
    let mut file = std::fs::File::create(&path)
        .map_err(|error| format!("cannot create '{}': {error}", path.display()))?;
    for value in values {
        file.write_all(&value.to_le_bytes())
            .map_err(|error| format!("cannot write '{}': {error}", path.display()))?;
    }
    println!("stage={stage} dump={}", path.display());
    Ok(())
}

fn dump_values_summary(
    stage: &str,
    window_index: usize,
    token_count: usize,
    embedding_dim: usize,
    values: &[f32],
    dump_values: usize,
) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum_f32 = 0.0f32;
    let mut sum_f64 = 0.0f64;
    let mut sum_squares = 0.0f64;
    let mut non_finite = 0usize;
    for value in values {
        if value.is_finite() {
            min = min.min(*value);
            max = max.max(*value);
        } else {
            non_finite += 1;
        }
        sum_f32 += *value;
        let value_f64 = f64::from(*value);
        sum_f64 += value_f64;
        sum_squares += value_f64 * value_f64;
    }
    let mean = if values.is_empty() {
        0.0
    } else {
        sum_f64 / values.len() as f64
    };
    let head = values.iter().take(dump_values).copied().collect::<Vec<_>>();
    let mut tail = values
        .iter()
        .rev()
        .take(dump_values)
        .copied()
        .collect::<Vec<_>>();
    tail.reverse();
    println!(
        "stage={stage} window={window_index} shape={embedding_dim}x{token_count} values={} non_finite={} sum_f32={sum_f32:.9} sum_f64={sum_f64:.12} mean={mean:.12} min={min:.9} max={max:.9} l2={:.12}",
        values.len(),
        non_finite,
        sum_squares.sqrt()
    );
    println!("stage={stage} head={}", format_values(&head));
    println!("stage={stage} tail={}", format_values(&tail));
}

fn format_values(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.9}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn synthetic_name(input: SyntheticInput) -> &'static str {
    match input {
        SyntheticInput::Zero => "zero",
        SyntheticInput::Half => "half",
        SyntheticInput::One => "one",
        SyntheticInput::Alternating => "1010",
    }
}

fn parse_args() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut mmproj = None;
    let mut audio = None;
    let mut synthetic = None;
    let mut conv_input = None;
    let mut frames = DEFAULT_SYNTHETIC_FRAMES;
    let mut dump_values = DEFAULT_DUMP_VALUES;
    let mut dump_dir = None;
    let mut debug = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mmproj" => {
                mmproj = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --mmproj".to_string())?,
                );
            }
            "--audio" => {
                audio = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --audio".to_string())?,
                );
            }
            "--synthetic" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --synthetic".to_string())?;
                synthetic = Some(parse_synthetic(&value)?);
            }
            "--conv-input" => {
                conv_input = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --conv-input".to_string())?,
                );
            }
            "--frames" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --frames".to_string())?;
                frames = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --frames value '{value}'"))?;
            }
            "--dump-values" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --dump-values".to_string())?;
                dump_values = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --dump-values value '{value}'"))?;
            }
            "--dump-dir" => {
                dump_dir = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --dump-dir".to_string())?,
                );
            }
            "--debug" => debug = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let mmproj = mmproj.ok_or_else(|| "missing required --mmproj <path>".to_string())?;
    let input_count = usize::from(audio.is_some())
        + usize::from(synthetic.is_some())
        + usize::from(conv_input.is_some());
    if input_count != 1 {
        return Err(
            "pass exactly one of --audio <path>, --synthetic <kind>, or --conv-input <path>"
                .to_string(),
        );
    }
    Ok(Options {
        mmproj,
        audio,
        synthetic,
        conv_input,
        frames,
        dump_values,
        dump_dir,
        debug,
    })
}

fn parse_synthetic(value: &str) -> Result<SyntheticInput, String> {
    match value {
        "zero" => Ok(SyntheticInput::Zero),
        "half" => Ok(SyntheticInput::Half),
        "one" => Ok(SyntheticInput::One),
        "1010" => Ok(SyntheticInput::Alternating),
        _ => Err(format!(
            "unknown synthetic input '{value}'; expected zero, half, one, or 1010"
        )),
    }
}

fn print_help() {
    println!(
        "Usage: cargo run --release --example audio_encoder_dump -- --mmproj <mmproj.gguf> (--audio <audio.wav> | --synthetic <zero|half|one|1010> | --conv-input <stage.f32>) [--frames <N>] [--dump-values <N>] [--dump-dir <DIR>] [--debug]"
    );
    println!();
    println!("Runs the Qwen3-ASR sidecar and prints stable summaries for parity comparison.");
    println!(
        "Synthetic inputs match llama.cpp's llama-mtmd-debug patterns; --frames defaults to 100."
    );
}
