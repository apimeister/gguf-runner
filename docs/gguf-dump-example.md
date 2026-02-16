# GGUF Dump Example

Use the `gguf_dump` example to inspect model metadata and tensor layout from a GGUF file.

## Run

```bash
cargo run --example gguf_dump -- --model ./model.gguf
```

Optional flags:

- `--dump-kv` only dump GGUF key/value metadata
- `--dump-tensors` only dump tensor table
- `--url <model-url>` load via URL-backed lazy range fetch
- `--debug` print extra parser/debug information

If neither `--dump-kv` nor `--dump-tensors` is provided, both sections are printed.

## Output Sections

- `== GGUF KV ==`
  - architecture, dimensions, vocab metadata, rope settings, and vendor-specific keys
- `== GGUF Tensors ==`
  - one row per tensor with: name, type, dims, element count, and data offsets

This output is useful for:

- verifying model family detection keys
- checking tensor naming conventions for new vendor support
- diagnosing tensor shape mismatches during weight loading
