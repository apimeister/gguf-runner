// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.

/// Zero-dependency binary serialization for the RAG index.
///
/// Format:
///   [8  bytes] magic: b"RAGIDX\x01\x00"
///   [4  bytes] embedding_dim: u32 LE
///   [4  bytes] chunk_count:   u32 LE
///   Per chunk:
///     [2 bytes] source_len: u16 LE
///     [N bytes] source: UTF-8
///     [4 bytes] text_len: u32 LE
///     [N bytes] text: UTF-8
///     [dim * 4] embedding: f32 LE array
use std::io::{Read, Write};
use std::path::Path;

use crate::rag::RagChunk;

const MAGIC: &[u8; 8] = b"RAGIDX\x01\x00";

pub(crate) fn save(path: &Path, dim: usize, chunks: &[RagChunk]) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("cannot create index file '{}': {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(file);

    w.write_all(MAGIC)
        .map_err(|e| format!("write magic: {e}"))?;
    write_u32(&mut w, dim as u32)?;
    write_u32(&mut w, chunks.len() as u32)?;

    for chunk in chunks {
        let src = chunk.source.as_bytes();
        let txt = chunk.text.as_bytes();
        if src.len() > u16::MAX as usize {
            return Err(format!(
                "source path too long ({} bytes): {}",
                src.len(),
                chunk.source
            ));
        }
        if txt.len() > u32::MAX as usize {
            return Err(format!("chunk text too long: {} bytes", txt.len()));
        }
        write_u16(&mut w, src.len() as u16)?;
        w.write_all(src).map_err(|e| format!("write source: {e}"))?;
        write_u32(&mut w, txt.len() as u32)?;
        w.write_all(txt).map_err(|e| format!("write text: {e}"))?;
        for &v in &chunk.embedding {
            w.write_all(&v.to_le_bytes())
                .map_err(|e| format!("write embedding float: {e}"))?;
        }
    }

    w.flush().map_err(|e| format!("flush index: {e}"))?;
    Ok(())
}

pub(crate) fn load(path: &Path) -> Result<(usize, Vec<RagChunk>), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("cannot open index file '{}': {e}", path.display()))?;
    let r = std::io::BufReader::new(file);
    parse(r, &path.display().to_string())
}

/// Parse a serialized index from an in-memory byte slice. Mirrors [`load`] but
/// avoids touching the filesystem — useful for precomputed indices embedded in
/// the binary via `include_bytes!`.
pub(crate) fn load_from_bytes(data: &[u8]) -> Result<(usize, Vec<RagChunk>), String> {
    parse(std::io::Cursor::new(data), "<embedded>")
}

pub(crate) fn save_to_writer<W: Write>(
    mut w: W,
    dim: usize,
    chunks: &[RagChunk],
) -> Result<(), String> {
    w.write_all(MAGIC)
        .map_err(|e| format!("write magic: {e}"))?;
    write_u32(&mut w, dim as u32)?;
    write_u32(&mut w, chunks.len() as u32)?;

    for chunk in chunks {
        let src = chunk.source.as_bytes();
        let txt = chunk.text.as_bytes();
        if src.len() > u16::MAX as usize {
            return Err(format!(
                "source path too long ({} bytes): {}",
                src.len(),
                chunk.source
            ));
        }
        if txt.len() > u32::MAX as usize {
            return Err(format!("chunk text too long: {} bytes", txt.len()));
        }
        write_u16(&mut w, src.len() as u16)?;
        w.write_all(src).map_err(|e| format!("write source: {e}"))?;
        write_u32(&mut w, txt.len() as u32)?;
        w.write_all(txt).map_err(|e| format!("write text: {e}"))?;
        for &v in &chunk.embedding {
            w.write_all(&v.to_le_bytes())
                .map_err(|e| format!("write embedding float: {e}"))?;
        }
    }
    w.flush().map_err(|e| format!("flush index: {e}"))?;
    Ok(())
}

fn parse<R: Read>(mut r: R, source_label: &str) -> Result<(usize, Vec<RagChunk>), String> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)
        .map_err(|e| format!("read magic: {e}"))?;
    if &magic != MAGIC {
        return Err(format!(
            "invalid RAG index magic in '{source_label}' — expected RAGIDX v1"
        ));
    }

    let dim = read_u32(&mut r)? as usize;
    if dim == 0 {
        return Err("invalid RAG index: embedding_dim is 0".to_string());
    }
    let count = read_u32(&mut r)? as usize;

    let mut chunks = Vec::with_capacity(count);
    for i in 0..count {
        let src_len = read_u16(&mut r)? as usize;
        let mut src_bytes = vec![0u8; src_len];
        r.read_exact(&mut src_bytes)
            .map_err(|e| format!("read source[{i}]: {e}"))?;
        let source = String::from_utf8(src_bytes)
            .map_err(|e| format!("source[{i}] is not valid UTF-8: {e}"))?;

        let txt_len = read_u32(&mut r)? as usize;
        let mut txt_bytes = vec![0u8; txt_len];
        r.read_exact(&mut txt_bytes)
            .map_err(|e| format!("read text[{i}]: {e}"))?;
        let text = String::from_utf8(txt_bytes)
            .map_err(|e| format!("text[{i}] is not valid UTF-8: {e}"))?;

        let mut embedding = vec![0f32; dim];
        for slot in &mut embedding {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("read embedding[{i}]: {e}"))?;
            *slot = f32::from_le_bytes(buf);
        }

        chunks.push(RagChunk {
            source,
            text,
            embedding,
        });
    }

    Ok((dim, chunks))
}

// ---------------------------------------------------------------------------
// Little-endian primitives
// ---------------------------------------------------------------------------

fn write_u16<W: Write>(w: &mut W, v: u16) -> Result<(), String> {
    w.write_all(&v.to_le_bytes())
        .map_err(|e| format!("write u16: {e}"))
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<(), String> {
    w.write_all(&v.to_le_bytes())
        .map_err(|e| format!("write u32: {e}"))
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)
        .map_err(|e| format!("read u16: {e}"))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::RagChunk;

    #[test]
    fn roundtrip_empty() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_rag_test_empty.ragidx");
        save(&path, 4, &[]).unwrap();
        let (dim, chunks) = load(&path).unwrap();
        assert_eq!(dim, 4);
        assert!(chunks.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn roundtrip_single_chunk() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_rag_test_single.ragidx");
        let chunk = RagChunk {
            source: "wiki/ops.md".to_string(),
            text: "Deploy with ./deploy.sh".to_string(),
            embedding: vec![0.1, -0.2, 0.3, 0.4],
        };
        save(&path, 4, &[chunk]).unwrap();
        let (dim, chunks) = load(&path).unwrap();
        assert_eq!(dim, 4);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].source, "wiki/ops.md");
        assert_eq!(chunks[0].text, "Deploy with ./deploy.sh");
        assert!((chunks[0].embedding[0] - 0.1f32).abs() < 1e-6);
        assert!((chunks[0].embedding[1] - (-0.2f32)).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bad_magic_rejected() {
        let dir = std::env::temp_dir();
        let path = dir.join("gguf_rag_test_bad_magic.ragidx");
        std::fs::write(&path, b"NOTARAG\x00\x00\x00\x00\x00").unwrap();
        assert!(load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
