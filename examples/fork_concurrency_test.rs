//! One loaded model, two handles generating at the same time, each on its own
//! rayon pool, mirroring everlock's interactive and background lanes. Every
//! concurrent answer must match the same request run alone. Usage:
//! `fork_concurrency_test <model.gguf> <mmproj.gguf> <image> [text_rounds] [image_rounds]`
use gguf_runner::{EmbeddedRuntime, Tool};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SYSTEM: &str =
    "You are a helpful assistant. Use the clock tool to answer questions about time.";
const TEXT: &str = "What time is it right now?";
const IMAGE: &str = "Describe this image in one short sentence.";

struct Clock;

impl Tool for Clock {
    fn name(&self) -> &str {
        "clock"
    }
    fn description(&self) -> &str {
        "Returns the current time."
    }
    fn call(&mut self, _args: &serde_json::Value) -> Result<serde_json::Value, String> {
        Ok(serde_json::json!({ "time": "12:30" }))
    }
}

fn text(rt: &mut EmbeddedRuntime, pool: &rayon::ThreadPool) -> String {
    let collected = Arc::new(Mutex::new(String::new()));
    let sink = Arc::clone(&collected);
    pool.install(|| {
        let mut clock = Clock;
        let mut tools: [&mut dyn Tool; 1] = [&mut clock];
        rt.generate_with_tools_streaming(&[], TEXT, SYSTEM, &mut tools, 4, move |tok: &str| {
            sink.lock().unwrap().push_str(tok);
        })
    })
    .expect("text generation");
    collected.lock().unwrap().clone()
}

fn image(rt: &mut EmbeddedRuntime, pool: &rayon::ThreadPool, path: &Path) -> String {
    pool.install(|| rt.generate_with_image(path, IMAGE, ""))
        .expect("image generation")
}

fn pool(name: &'static str, threads: usize) -> Arc<rayon::ThreadPool> {
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |i| format!("{name}-{i}"))
            .build()
            .expect("thread pool"),
    )
}

fn overlap(a: &(Duration, Duration, String), b: &(Duration, Duration, String)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data: &'static [u8] = Box::leak(std::fs::read(&args[0]).expect("model").into_boxed_slice());
    let proj: &'static [u8] =
        Box::leak(std::fs::read(&args[1]).expect("mmproj").into_boxed_slice());
    let path = PathBuf::from(&args[2]);
    let text_rounds: usize = args.get(3).map_or(3, |v| v.parse().expect("text_rounds"));
    let image_rounds: usize = args.get(4).map_or(2, |v| v.parse().expect("image_rounds"));

    let mut base = EmbeddedRuntime::load_from_bytes(data).expect("load");
    base.load_mmproj_from_bytes(proj).expect("mmproj load");
    base.use_hidden_think_mode();

    let mut text_rt = base.fork();
    text_rt.set_context_size(32768);
    text_rt.set_hidden_think_token_cap(0);
    let mut image_rt = base.fork();
    image_rt.set_context_size(16384);
    image_rt.set_hidden_think_token_cap(1024);

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let text_pool = pool("text", threads);
    let image_pool = pool("image", threads);

    let solo_text = text(&mut text_rt, &text_pool);
    println!("solo text: {solo_text:?}");
    let solo_image = image(&mut image_rt, &image_pool, &path);
    println!("solo image: {solo_image:?}");

    let origin = Instant::now();
    let image_thread = {
        let pool = Arc::clone(&image_pool);
        let path = path.clone();
        std::thread::spawn(move || {
            (0..image_rounds)
                .map(|_| {
                    let start = origin.elapsed();
                    let out = image(&mut image_rt, &pool, &path);
                    (start, origin.elapsed(), out)
                })
                .collect::<Vec<_>>()
        })
    };
    let text_thread = {
        let pool = Arc::clone(&text_pool);
        std::thread::spawn(move || {
            (0..text_rounds)
                .map(|_| {
                    let start = origin.elapsed();
                    let out = text(&mut text_rt, &pool);
                    (start, origin.elapsed(), out)
                })
                .collect::<Vec<_>>()
        })
    };
    let image_runs = image_thread.join().expect("image thread");
    let text_runs = text_thread.join().expect("text thread");

    let mismatches = image_runs.iter().filter(|r| r.2 != solo_image).count()
        + text_runs.iter().filter(|r| r.2 != solo_text).count();
    let overlaps: usize = image_runs
        .iter()
        .map(|i| text_runs.iter().filter(|t| overlap(i, t)).count())
        .sum();
    for run in image_runs.iter().filter(|r| r.2 != solo_image) {
        println!("image mismatch: {:?}", run.2);
    }
    for run in text_runs.iter().filter(|r| r.2 != solo_text) {
        println!("text mismatch: {:?}", run.2);
    }
    println!(
        "text runs={} image runs={} overlapping pairs={overlaps} mismatches={mismatches}",
        text_runs.len(),
        image_runs.len()
    );
    assert_eq!(mismatches, 0, "concurrent output differs from solo output");
    assert!(overlaps > 0, "the runs never overlapped");
}
