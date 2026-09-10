fn main() {
    if let Err(e) = gguf_runner::run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
