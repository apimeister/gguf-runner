//! Checks the face path the way an embedding application uses it: both models handed over
//! as static bytes, images handed over as in-memory bytes, and one loaded detector/encoder
//! shared by several threads that each run in their own rayon pool.
//!
//! ```text
//! cargo run --release --example face_embedded_test -- detector.gguf encoder.gguf a.jpg b.png
//! ```
//!
//! Exits nonzero if the embedded results differ from the file-based ones or between threads.

use std::path::Path;

use gguf_runner::{FaceDetector, FaceEmbedding, FaceEncoder, FaceImage, FaceRecord};

const THREADS: usize = 4;

fn leak(path: &str) -> &'static [u8] {
    Box::leak(std::fs::read(path).expect("read model").into_boxed_slice())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: face_embedded_test <detector.gguf> <encoder.gguf> <image>...");
        std::process::exit(2);
    }
    let (detector_path, encoder_path, images) = (&args[0], &args[1], &args[2..]);

    // What an application with linked-in models does.
    let detector = FaceDetector::load_from_bytes(leak(detector_path)).expect("detector bytes");
    let encoder = FaceEncoder::load_from_bytes(leak(encoder_path)).expect("encoder bytes");
    println!(
        "embedded: {} + {} ({} dimensions)",
        detector.model_name(),
        encoder.model_name(),
        encoder.embedding_dimension()
    );

    // File-based loading must describe the same models and give the same results.
    let file_detector = FaceDetector::load_from_file(Path::new(detector_path)).expect("detector");
    let file_encoder = FaceEncoder::load_from_file(Path::new(encoder_path)).expect("encoder");
    assert_eq!(
        detector.model_fingerprint(),
        file_detector.model_fingerprint()
    );
    assert_eq!(
        encoder.model_fingerprint(),
        file_encoder.model_fingerprint()
    );

    let encoded: Vec<Vec<u8>> = images
        .iter()
        .map(|path| std::fs::read(path).expect("read image"))
        .collect();

    let mut serial: Vec<Vec<FaceRecord>> = Vec::new();
    for (path, bytes) in images.iter().zip(&encoded) {
        let image = FaceImage::from_encoded_bytes(bytes).expect("decode image bytes");
        let records = encoder.embed_image(&detector, &image).expect("embed image");
        let from_file = file_encoder
            .embed_all(&file_detector, Path::new(path))
            .expect("embed file");
        assert_eq!(
            records, from_file.faces,
            "embedded and file results differ for {path}"
        );
        println!(
            "{path}: {} faces, identical to file-based loading",
            records.len()
        );
        serial.push(records);
    }

    // One loaded pair of models, several threads, each with its own rayon pool.
    let results: Vec<Vec<Vec<FaceRecord>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let (detector, encoder, encoded) = (&detector, &encoder, &encoded);
                scope.spawn(move || {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(2)
                        .build()
                        .expect("thread pool");
                    pool.install(|| {
                        encoded
                            .iter()
                            .map(|bytes| {
                                let image = FaceImage::from_encoded_bytes(bytes).expect("decode");
                                encoder.embed_image(detector, &image).expect("embed")
                            })
                            .collect()
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect()
    });
    for (thread, result) in results.iter().enumerate() {
        assert_eq!(
            result, &serial,
            "thread {thread} differs from the serial run"
        );
    }
    println!("{THREADS} concurrent threads: identical to the serial run");

    // The stored 128-byte form must rank pairs like the float vectors do.
    let faces: Vec<&FaceRecord> = serial.iter().flatten().collect();
    let mut worst = 0.0f32;
    for (index, left) in faces.iter().enumerate() {
        for right in &faces[index + 1..] {
            let float: f32 = left
                .vector
                .as_ref()
                .unwrap()
                .iter()
                .zip(right.vector.as_ref().unwrap())
                .map(|(a, b)| a * b)
                .sum();
            let stored = FaceEmbedding::quantized_cosine_similarity(
                &FaceEmbedding::decode_quantized_base64(left.vector_q8.as_ref().unwrap()).unwrap(),
                &FaceEmbedding::decode_quantized_base64(right.vector_q8.as_ref().unwrap()).unwrap(),
            )
            .unwrap();
            worst = worst.max((float - stored).abs());
        }
    }
    assert!(worst < 0.02, "stored-form similarity drifts by {worst}");
    println!(
        "{} faces: stored-form similarity within {worst:.4} of float",
        faces.len()
    );
    println!("ok");
}
