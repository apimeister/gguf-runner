//! Detects and embeds the faces of several images and prints every cross-image similarity.
//!
//! ```text
//! cargo run --release --example face_dump -- \
//!     --detector face-yunet-2023mar-f32.gguf --encoder face-sface-2021dec-f32.gguf \
//!     a.jpg b.png
//! ```
//!
//! `--box x,y,w,h` treats that region of the first image as a face known only by its box:
//! landmarks are looked up with the detector, with a plain-crop embedding as the fallback.
//!
//! `--crop <file>` instead embeds an already aligned crop and prints its vector as JSON, which
//! is how the encoder is checked against a reference implementation.

use std::path::{Path, PathBuf};

use gguf_runner::{DetectedFace, FaceBox, FaceDetector, FaceEmbedding, FaceEncoder, FaceImage};

struct Options {
    detector: Option<PathBuf>,
    encoder: PathBuf,
    crops: Vec<PathBuf>,
    manual_box: Option<FaceBox>,
    images: Vec<PathBuf>,
}

fn parse_args() -> Result<Options, String> {
    let (mut detector, mut encoder) = (None, None);
    let (mut crops, mut images) = (Vec::new(), Vec::new());
    let mut manual_box = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("{name} requires a path"))
        };
        match arg.as_str() {
            "--detector" => detector = Some(value("--detector")?),
            "--encoder" => encoder = Some(value("--encoder")?),
            "--crop" => crops.push(value("--crop")?),
            "--box" => {
                let raw = args.next().ok_or("--box requires x,y,w,h")?;
                let parts: Vec<f32> = raw
                    .split(',')
                    .map(|part| part.trim().parse::<f32>().map_err(|e| e.to_string()))
                    .collect::<Result<_, _>>()?;
                let [x, y, width, height] = parts[..] else {
                    return Err("--box requires x,y,w,h".to_string());
                };
                manual_box = Some(FaceBox {
                    x,
                    y,
                    width,
                    height,
                });
            }
            other if other.starts_with("--") => return Err(format!("unknown argument: {other}")),
            other => images.push(PathBuf::from(other)),
        }
    }
    Ok(Options {
        detector,
        encoder: encoder.ok_or("--encoder <encoder.gguf> is required")?,
        crops,
        manual_box,
        images,
    })
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let encoder = FaceEncoder::load_from_file(&options.encoder)?;
    for crop in &options.crops {
        let embedding = encoder.embed_aligned_crop(&FaceImage::open(crop)?)?;
        println!(
            "{}",
            serde_json::json!({"crop": crop.to_string_lossy(), "vector": embedding.vector})
        );
    }
    if options.images.is_empty() {
        return Ok(());
    }
    let detector_path = options
        .detector
        .ok_or("--detector <detector.gguf> is required for images")?;
    let detector = FaceDetector::load_from_file(&detector_path)?;
    let threshold = encoder.published_match_threshold();

    let mut faces: Vec<(usize, usize, FaceEmbedding)> = Vec::new();
    for (image_index, path) in options.images.iter().enumerate() {
        let image = FaceImage::open(Path::new(path))?;
        let detected = detector.detect(&image)?;
        println!(
            "image {image_index}: {} ({}x{}), {} faces",
            path.display(),
            image.width(),
            image.height(),
            detected.len()
        );
        for (face_index, face) in detected.iter().enumerate() {
            println!(
                "  face {face_index}: score {:.3}, box x={:.0} y={:.0} w={:.0} h={:.0}",
                face.score.unwrap_or(0.0),
                face.bbox.x,
                face.bbox.y,
                face.bbox.width,
                face.bbox.height
            );
            faces.push((image_index, face_index, encoder.embed(&image, face)?));
        }
        if let (0, Some(manual)) = (image_index, &options.manual_box) {
            let located = detector.locate(&image, manual)?;
            println!(
                "  box: landmarks {}",
                if located.is_some() {
                    "found"
                } else {
                    "not found"
                }
            );
            let face = located.unwrap_or_else(|| DetectedFace::from_box(*manual));
            let embedding = encoder.embed(&image, &face)?;
            println!("  box: aligned={}", embedding.aligned);
            for (other_image, other_face, other) in faces.iter().filter(|f| f.0 == 0) {
                println!(
                    "  box vs image {other_image} face {other_face}: {:.3}",
                    embedding.cosine_similarity(other)?
                );
            }
        }
    }

    println!("cross-image cosine similarity (published same-person threshold {threshold:.3}):");
    for (left_image, left_face, left) in &faces {
        for (right_image, right_face, right) in &faces {
            if left_image >= right_image {
                continue;
            }
            let similarity = left.cosine_similarity(right)?;
            let verdict = if similarity >= threshold {
                "  <- same person"
            } else {
                ""
            };
            println!(
                "  image {left_image} face {left_face} vs image {right_image} face {right_face}: {similarity:.3}{verdict}"
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
