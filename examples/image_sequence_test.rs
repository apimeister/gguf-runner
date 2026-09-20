//! Mirrors everlock's enhance-metadata pattern: one resident runtime, several
//! image prompts back to back. Usage:
//! `image_sequence_test <model.gguf> <mmproj.gguf> <image> <prompt-file>...`
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (model, mmproj, image, prompts) = (&args[0], &args[1], &args[2], &args[3..]);
    let data: &'static [u8] = Box::leak(fs::read(model).expect("model").into_boxed_slice());
    let proj: &'static [u8] = Box::leak(fs::read(mmproj).expect("mmproj").into_boxed_slice());
    let mut rt = gguf_runner::EmbeddedRuntime::load_from_bytes(data).expect("load");
    rt.load_mmproj_from_bytes(proj).expect("mmproj load");
    rt.use_hidden_think_mode();
    if std::env::var("SEQ_DEBUG").is_ok() {
        rt.set_debug(true);
    }
    for prompt_file in prompts {
        let prompt = fs::read_to_string(prompt_file)
            .expect("prompt")
            .trim_end()
            .to_string();
        rt.set_context_size(16384);
        rt.set_hidden_think_token_cap(1024);
        let out = rt
            .generate_with_image(std::path::Path::new(image), &prompt, "")
            .unwrap_or_else(|e| format!("ERROR: {e}"));
        println!("=== {prompt_file}\n{out}\n");
    }
}
