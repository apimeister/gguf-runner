//! Development harness for `src/engine/audio`: times the log-Mel front-end and
//! can dump raw feature bits so optimizations can be checked for bit equality.

#![allow(dead_code)]
#![allow(unused_imports)]

#[path = "../src/engine/mod.rs"]
mod engine;
#[path = "../src/vendors/mod.rs"]
mod vendors;

use engine::audio::prepare_audios_for_multimodal;
use engine::types::AudioEncoderBackend;
use std::io::Write;
use std::time::Instant;
use vendors::audio_preprocess_config;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    let dump_dir = std::env::var("DUMP_DIR").ok();
    let config = audio_preprocess_config(AudioEncoderBackend::Qwen3Asr);

    for path in &args {
        let mut times = Vec::with_capacity(reps);
        let mut prepared = None;
        for _ in 0..reps {
            let start = Instant::now();
            let result = prepare_audios_for_multimodal(std::slice::from_ref(path), config)
                .expect("prepare failed");
            times.push(start.elapsed().as_secs_f64());
            prepared = Some(result);
        }
        times.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
        let prepared = prepared.expect("at least one repetition");
        let audio = &prepared[0];
        let frames: usize = audio.feature_windows.iter().map(|w| w.valid_frames).sum();
        let name = path.rsplit('/').next().unwrap_or(path);
        println!(
            "{name}: min={:.5}s p50={:.5}s max={:.5}s n={} frames={frames}",
            times[0],
            times[times.len() / 2],
            times[times.len() - 1],
            times.len()
        );

        if let Some(dir) = &dump_dir {
            let mut out = std::fs::File::create(format!("{dir}/{name}.mel")).expect("dump create");
            for window in &audio.feature_windows {
                for value in &window.data_mel_major {
                    out.write_all(&value.to_bits().to_le_bytes())
                        .expect("dump write");
                }
            }
        }
    }
}
