//! Positional diagnostic arguments: MODEL MMPROJ IMAGE OUTPUT_DIRECTORY [f32].
//! Run in release mode; full 896px attention is intentionally retained.
#![allow(dead_code, unused_imports)]

#[path = "../src/engine/mod.rs"]
mod engine;
#[path = "../src/vendors/mod.rs"]
mod vendors;

use engine::io::parse_gguf_file;
use engine::multimodal::{VisionEncoder, build_vision_encoder_from_mmproj};
use engine::vision::{
    ImageNormalization, ImagePreprocessProfile, ImageResizeMode, prepare_images_for_multimodal,
};
use std::io::Write;
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, mmproj, image, output, precision @ ..] = args.as_slice() else {
        return Err("usage: gemma3_encoder_dump MODEL MMPROJ IMAGE OUTPUT_DIRECTORY [f32]".into());
    };
    let f32_activations = match precision {
        [] => false,
        [value] if value == "f32" => true,
        _ => return Err("optional diagnostic precision must be 'f32'".into()),
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .map_err(|error| error.to_string())?;
    let model = parse_gguf_file(model, false)?;
    let config = vendors::build_config_from_gguf(&model, false)?;
    let policy = vendors::multimodal_policy(&config);
    let encoder = build_vision_encoder_from_mmproj(&config, parse_gguf_file(mmproj, false)?)?
        .ok_or("model has no vision backend")?;
    let size = encoder.recommended_image_size();
    let (mean, std) = encoder.recommended_image_normalization();
    let profile = ImagePreprocessProfile::new_with_mode(
        size,
        size,
        ImageNormalization::MeanStd { mean, std },
        ImageResizeMode::Stretch,
        encoder.recommended_image_alignment(),
    )
    .with_stretch_filter(policy.image_stretch_filter);
    let prepared = prepare_images_for_multimodal(std::slice::from_ref(image), profile)?;
    let VisionEncoder::Gemma3(encoder) = encoder else {
        return Err("this diagnostic requires the Gemma3 vision backend".into());
    };
    // Refuse an existing directory so stale stages cannot masquerade as this run.
    std::fs::create_dir(output).map_err(|error| format!("cannot create {output}: {error}"))?;
    let mut stages = Vec::new();
    let mut observer = |name: &str, values: &[f32]| -> Result<(), String> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("non-finite value in stage {name}"));
        }
        let file = std::fs::File::create(Path::new(output).join(format!("{name}.f32le")))
            .map_err(|error| error.to_string())?;
        let mut writer = std::io::BufWriter::new(file);
        for value in values {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|error| error.to_string())?;
        }
        writer.flush().map_err(|error| error.to_string())?;
        stages.push(serde_json::json!({"name": name, "values": values.len()}));
        println!("{name}: {} values", values.len());
        Ok(())
    };
    observer("input_chw", &prepared[0].data_chw)?;
    // Operation traces locate the first and final mismatches in the pinned
    // 27-layer checkpoint; this diagnostic requires at least 27 layers.
    encoder.encode_image_with_stages(&prepared[0], &[0, 1, 26], f32_activations, &mut observer)?;
    let manifest = serde_json::json!({"schema_version": 1, "image": image, "width": size,
        "height": size, "embedding_dim": config.dim, "stages": stages,
        "arithmetic": "ggml_cpu_v1",
        "encoder_activations": if f32_activations { "f32" } else { "storage_dtype" }});
    std::fs::write(
        Path::new(output).join("stages.json"),
        format!("{manifest:#}\n"),
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}
