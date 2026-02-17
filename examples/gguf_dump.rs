#![allow(dead_code)]
#![allow(unused_imports)]

#[path = "../src/engine/mod.rs"]
mod engine;

use engine::io::parse_gguf_file;
use engine::types::{
    GGUFFile, GgmlType, GgufValue, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
};

struct InspectOptions {
    model: String,
    url: Option<String>,
    debug: bool,
    dump_kv: bool,
    dump_tensors: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let opts = parse_args()?;
    let gguf = parse_gguf_file(&opts.model, opts.url.as_deref(), opts.debug)?;

    if opts.dump_kv {
        dump_kv(&gguf);
    }
    if opts.dump_tensors {
        dump_tensors(&gguf);
    }
    Ok(())
}

fn parse_args() -> Result<InspectOptions, String> {
    let mut args = std::env::args().skip(1);
    let mut model: Option<String> = None;
    let mut url: Option<String> = None;
    let mut debug = false;
    let mut dump_kv = false;
    let mut dump_tensors = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => {
                let v = args
                    .next()
                    .ok_or_else(|| "missing value for --model".to_string())?;
                model = Some(v);
            }
            "--url" => {
                let v = args
                    .next()
                    .ok_or_else(|| "missing value for --url".to_string())?;
                url = Some(v);
            }
            "--dump-kv" => dump_kv = true,
            "--dump-tensors" => dump_tensors = true,
            "--debug" => debug = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let model = model.ok_or_else(|| "missing required --model <path>".to_string())?;

    if !dump_kv && !dump_tensors {
        dump_kv = true;
        dump_tensors = true;
    }

    Ok(InspectOptions {
        model,
        url,
        debug,
        dump_kv,
        dump_tensors,
    })
}

fn print_help() {
    println!(
        "Usage: cargo run --example gguf_dump -- --model <model.gguf> [--dump-kv] [--dump-tensors] [--url <model-url>] [--debug]"
    );
    println!();
    println!("Dumps GGUF metadata for model reverse-engineering.");
    println!("If neither --dump-kv nor --dump-tensors is passed, both are dumped.");
}

fn dump_kv(gguf: &GGUFFile) {
    println!("== GGUF KV ==");
    println!(
        "# version={}, n_kv={}, n_tensors={}",
        gguf.version, gguf.n_kv, gguf.n_tensors
    );

    let mut entries: Vec<(&str, &GgufValue)> =
        gguf.kv.iter().map(|(k, v)| (k.as_str(), v)).collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in entries {
        println!("{key}\t{}", format_gguf_value(value));
    }

    println!("tokenizer.ggml.tokens.count\t{}", gguf.vocab_tokens.len());
    println!("tokenizer.ggml.scores.count\t{}", gguf.vocab_scores.len());
    println!("tokenizer.ggml.merges.count\t{}", gguf.vocab_merges.len());
}

fn dump_tensors(gguf: &GGUFFile) {
    println!("== GGUF Tensors ==");
    println!("name\ttype\tdims\telements\toffset\tdata_offset");

    for t in &gguf.tensors {
        let nd = t.n_dims as usize;
        let dims: Vec<u64> = t.ne[..nd].to_vec();
        let elements = dims
            .iter()
            .fold(1usize, |acc, v| acc.saturating_mul(*v as usize));
        let dims_text = dims
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("x");
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            t.name,
            ggml_type_name(t.ttype),
            dims_text,
            elements,
            t.offset,
            t.data_offset
        );
    }
}

fn format_gguf_value(v: &GgufValue) -> String {
    match v {
        GgufValue::UInt(x) => x.to_string(),
        GgufValue::Int(x) => x.to_string(),
        GgufValue::F32(x) => x.to_string(),
        GgufValue::F64(x) => x.to_string(),
        GgufValue::Bool(x) => x.to_string(),
        GgufValue::Str(s) => format!("{s:?}"),
    }
}

fn ggml_type_name(t: GgmlType) -> &'static str {
    match t.0 {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_BF16 => "BF16",
        GGML_TYPE_Q4_0 => "Q4_0",
        GGML_TYPE_Q4_1 => "Q4_1",
        GGML_TYPE_Q5_0 => "Q5_0",
        GGML_TYPE_Q5_1 => "Q5_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        GGML_TYPE_Q2_K => "Q2_K",
        GGML_TYPE_Q3_K => "Q3_K",
        GGML_TYPE_Q4_K => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        GGML_TYPE_Q6_K => "Q6_K",
        GGML_TYPE_IQ4_NL => "IQ4_NL",
        _ => "UNKNOWN",
    }
}
