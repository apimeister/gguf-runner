// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.

//! Markdown → overlapping chunk splitter.
//!
//! Strategy:
//!  1. Split on heading lines (`#`, `##`, `###`, …) — each section is a base chunk.
//!  2. Carry the heading breadcrumb into the chunk text so retrieved chunks are self-contained.
//!  3. If a section body exceeds `max_chars`, slide a window with ~20 % overlap at paragraph
//!     boundaries.
//!  4. Drop stubs shorter than `MIN_CHUNK_CHARS`.

const MIN_CHUNK_CHARS: usize = 20;

// ---------------------------------------------------------------------------
// Language detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    C,
    Java,
}

fn detect_language(ext: &str) -> Option<Language> {
    match ext {
        "rs" => Some(Language::Rust),
        "py" => Some(Language::Python),
        "ts" | "tsx" => Some(Language::TypeScript),
        "js" | "jsx" => Some(Language::JavaScript),
        "go" => Some(Language::Go),
        "c" | "h" => Some(Language::C),
        "java" => Some(Language::Java),
        _ => None,
    }
}

#[derive(Debug)]
pub(crate) struct RawChunk {
    /// Relative or absolute source path, e.g. `wiki/ops/deploy.md`.
    pub(crate) source: String,
    /// The chunk text that will be embedded and injected into the context.
    pub(crate) text: String,
}

/// Split `content` from `source_path` into overlapping text chunks.
///
/// `max_chars` is the soft upper bound per chunk (characters, not tokens).
/// A value around 1500–2000 works well for most 512-token embedding models.
pub(crate) fn chunk_markdown(source_path: &str, content: &str, max_chars: usize) -> Vec<RawChunk> {
    let overlap = max_chars / 5; // ~20 % overlap
    let sections = split_into_sections(content);
    let mut out = Vec::new();

    // Greedily merge adjacent sections into chunks up to `max_chars`.  Without
    // this, every heading in a heavily-subdivided doc becomes its own tiny
    // chunk — embedding then spends most of its time on per-call overhead and
    // the resulting index bloats.  Merged chunks also retrieve better: small
    // section bodies in isolation rarely have enough signal to score well.
    let mut pending = String::new();
    let flush_pending = |pending: &mut String, out: &mut Vec<RawChunk>| {
        let trimmed = pending.trim();
        if trimmed.len() >= MIN_CHUNK_CHARS {
            out.push(RawChunk {
                source: source_path.to_string(),
                text: trimmed.to_string(),
            });
        }
        pending.clear();
    };

    for (breadcrumb, body) in sections {
        let prefix = if breadcrumb.is_empty() {
            String::new()
        } else {
            format!("{breadcrumb}\n\n")
        };
        let full = format!("{prefix}{body}");

        if full.len() > max_chars {
            // Section larger than max — flush whatever we were merging, then
            // emit sliding windows for this section directly.
            flush_pending(&mut pending, &mut out);
            for window in sliding_window(&full, max_chars, overlap) {
                let trimmed = window.trim();
                if trimmed.len() >= MIN_CHUNK_CHARS {
                    out.push(RawChunk {
                        source: source_path.to_string(),
                        text: trimmed.to_string(),
                    });
                }
            }
            continue;
        }

        // Would appending overflow the current merged chunk? If so, flush
        // and start a fresh one with just this section.
        let joiner = if pending.is_empty() { "" } else { "\n\n" };
        if !pending.is_empty() && pending.len() + joiner.len() + full.len() > max_chars {
            flush_pending(&mut pending, &mut out);
        }
        if !pending.is_empty() {
            pending.push_str(joiner);
        }
        pending.push_str(&full);
    }
    flush_pending(&mut pending, &mut out);

    out
}

/// Walk the file tree under `dir` and chunk every `.md` file.
pub(crate) fn chunk_directory(
    dir: &std::path::Path,
    max_chars: usize,
) -> Result<Vec<RawChunk>, String> {
    let mut chunks = Vec::new();
    chunk_dir_recursive(dir, dir, max_chars, &mut chunks)?;
    Ok(chunks)
}

fn chunk_dir_recursive(
    root: &std::path::Path,
    dir: &std::path::Path,
    max_chars: usize,
    out: &mut Vec<RawChunk>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read directory '{}': {e}", dir.display()))?;
    let mut paths: Vec<std::path::PathBuf> =
        entries.filter_map(|e| e.ok().map(|de| de.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            chunk_dir_recursive(root, &path, max_chars, out)?;
        } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            let is_md = ext == "md";
            let lang = if is_md { None } else { detect_language(ext) };
            if is_md || lang.is_some() {
                let content = std::fs::read_to_string(&path)
                    .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
                // Use a path relative to the root for source attribution.
                let rel = path
                    .strip_prefix(root)
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| path.to_string_lossy().into_owned());
                let chunks = if is_md {
                    chunk_markdown(&rel, &content, max_chars)
                } else {
                    chunk_code(&rel, &content, max_chars, lang.unwrap())
                };
                out.extend(chunks);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Code-aware chunking
// ---------------------------------------------------------------------------

/// Split a code file into chunks, one per top-level definition.
///
/// The definition line (e.g. `fn foo(…)`) is included at the start of each chunk
/// so chunks are self-contained without a separate breadcrumb.
pub(crate) fn chunk_code(
    source_path: &str,
    content: &str,
    max_chars: usize,
    lang: Language,
) -> Vec<RawChunk> {
    let overlap = max_chars / 5;
    let sections = split_code_into_sections(content, lang);
    let mut out = Vec::new();

    for body in sections {
        let text = body.trim();
        if text.len() < MIN_CHUNK_CHARS {
            continue;
        }
        if text.len() <= max_chars {
            out.push(RawChunk {
                source: source_path.to_string(),
                text: text.to_string(),
            });
        } else {
            let windows = sliding_window(text, max_chars, overlap);
            for window in windows {
                if window.trim().len() >= MIN_CHUNK_CHARS {
                    out.push(RawChunk {
                        source: source_path.to_string(),
                        text: window.trim().to_string(),
                    });
                }
            }
        }
    }
    out
}

/// Split `content` at every top-level definition boundary for `lang`.
/// Returns a list of body strings (each starting with the definition line).
fn split_code_into_sections(content: &str, lang: Language) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut split_points: Vec<usize> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        if is_definition_start(line, lang) {
            split_points.push(i);
        }
    }

    if split_points.is_empty() {
        return vec![content.to_string()];
    }

    let mut sections: Vec<String> = Vec::new();

    // Content before the first definition (imports, module-level declarations, etc.)
    if split_points[0] > 0 {
        let preamble = lines[..split_points[0]].join("\n");
        if !preamble.trim().is_empty() {
            sections.push(preamble);
        }
    }

    for (idx, &start) in split_points.iter().enumerate() {
        let end = if idx + 1 < split_points.len() {
            split_points[idx + 1]
        } else {
            lines.len()
        };
        sections.push(lines[start..end].join("\n"));
    }

    sections
}

/// Return `true` if `line` looks like the start of a top-level definition.
///
/// We require no leading whitespace (top-level = column 0) and a language-specific
/// keyword prefix.  The heuristics are intentionally simple — no full parser needed.
fn is_definition_start(line: &str, lang: Language) -> bool {
    // Indented lines are never top-level in any of our supported languages.
    if line.starts_with(|c: char| c.is_whitespace()) || line.is_empty() {
        return false;
    }
    match lang {
        Language::Rust => {
            const KWS: &[&str] = &[
                "fn ",
                "async fn ",
                "pub fn ",
                "pub async fn ",
                "pub(crate) fn ",
                "pub(crate) async fn ",
                "pub(super) fn ",
                "struct ",
                "pub struct ",
                "enum ",
                "pub enum ",
                "impl ",
                "impl<",
                "trait ",
                "pub trait ",
                "type ",
                "pub type ",
                "const ",
                "pub const ",
                "static ",
                "pub static ",
                "mod ",
                "pub mod ",
                "#[",
            ];
            KWS.iter().any(|kw| line.starts_with(kw))
        }
        Language::Python => {
            line.starts_with("def ")
                || line.starts_with("async def ")
                || line.starts_with("class ")
                || line.starts_with("@") // decorator before def/class
        }
        Language::TypeScript | Language::JavaScript => {
            const KWS: &[&str] = &[
                "function ",
                "async function ",
                "export function ",
                "export async function ",
                "export default function ",
                "class ",
                "export class ",
                "export default class ",
                "export const ",
                "export let ",
                "export var ",
                "export type ",
                "export interface ",
                "export enum ",
                "interface ",
                "type ",
                "enum ",
                "const ",
                "let ",
                "var ", // top-level var declarations / arrow fns
                "@",    // decorator
            ];
            KWS.iter().any(|kw| line.starts_with(kw))
        }
        Language::Go => {
            line.starts_with("func ")
                || line.starts_with("type ")
                || line.starts_with("var ")
                || line.starts_with("const ")
        }
        Language::C => {
            // Explicit keywords that begin declarations/definitions at column 0.
            const KWS: &[&str] = &[
                "struct ",
                "enum ",
                "union ",
                "typedef ",
                "void ",
                "int ",
                "char ",
                "float ",
                "double ",
                "long ",
                "short ",
                "unsigned ",
                "signed ",
                "static ",
                "extern ",
                "inline ",
                "const ",
                "__attribute__",
            ];
            if KWS.iter().any(|kw| line.starts_with(kw)) {
                return true;
            }
            // Bare identifier followed by `(` — typical function definition / declaration.
            // Exclude preprocessor lines, comments, and lone `{`.
            !line.starts_with("//")
                && !line.starts_with("/*")
                && !line.starts_with('#')
                && !line.starts_with('*')
                && !line.starts_with('{')
                && line.contains('(')
        }
        Language::Java => {
            const KWS: &[&str] = &[
                "public ",
                "private ",
                "protected ",
                "class ",
                "interface ",
                "enum ",
                "record ",
                "abstract class ",
                "abstract interface ",
                "static ",
                "@",
            ];
            KWS.iter().any(|kw| line.starts_with(kw))
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A section is a `(breadcrumb, body)` pair.
/// breadcrumb: the heading hierarchy joined with " > ", e.g. "# Ops > ## Deploy".
/// body: everything between the opening heading line and the next same-or-higher heading.
fn split_into_sections(content: &str) -> Vec<(String, String)> {
    // Collect (heading_level, heading_text, start_line_index) triples.
    let lines: Vec<&str> = content.lines().collect();
    let mut heading_positions: Vec<(usize, String, usize)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some((level, text)) = parse_heading(line) {
            heading_positions.push((level, text, i));
        }
    }

    if heading_positions.is_empty() {
        // No headings — treat entire file as one section.
        return vec![(String::new(), content.to_string())];
    }

    let mut sections: Vec<(String, String)> = Vec::new();

    // Text before the first heading, if any.
    let first_heading_line = heading_positions[0].2;
    if first_heading_line > 0 {
        let preamble = lines[..first_heading_line].join("\n");
        if !preamble.trim().is_empty() {
            sections.push((String::new(), preamble));
        }
    }

    for (idx, &(level, ref text, start)) in heading_positions.iter().enumerate() {
        let body_start = start + 1;
        let body_end = if idx + 1 < heading_positions.len() {
            heading_positions[idx + 1].2
        } else {
            lines.len()
        };
        let body = lines[body_start..body_end].join("\n");

        // Build breadcrumb: collect ancestor headings at levels < current.
        let breadcrumb = build_breadcrumb(&heading_positions[..idx], level, text);
        sections.push((breadcrumb, body));
    }

    sections
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_end();
    let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = trimmed[hashes..].trim();
    if rest.is_empty() {
        return None;
    }
    // A valid ATX heading must have a space after the hashes.
    if !trimmed[hashes..].starts_with(' ') {
        return None;
    }
    Some((hashes, rest.to_string()))
}

fn build_breadcrumb(
    ancestors: &[(usize, String, usize)],
    current_level: usize,
    current_text: &str,
) -> String {
    // Walk ancestors in reverse; collect the most recent heading at each level < current.
    let mut seen_levels: Vec<(usize, &str)> = Vec::new();
    for &(level, ref text, _) in ancestors.iter().rev() {
        if level < current_level && !seen_levels.iter().any(|(l, _)| *l == level) {
            seen_levels.push((level, text.as_str()));
        }
    }
    seen_levels.sort_by_key(|(l, _)| *l);

    let hashes = "#".repeat(current_level);
    let heading = format!("{hashes} {current_text}");
    if seen_levels.is_empty() {
        heading
    } else {
        let ancestors_str: Vec<String> = seen_levels
            .iter()
            .map(|(l, t)| format!("{} {t}", "#".repeat(*l)))
            .collect();
        format!("{} > {heading}", ancestors_str.join(" > "))
    }
}

/// Split `text` into overlapping windows, splitting preferably at paragraph breaks.
fn sliding_window(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    let mut windows = Vec::new();
    let mut start = 0;
    let len = text.len();

    while start < len {
        // Round end down to a valid char boundary — raw byte arithmetic can land mid-char.
        let end = floor_char_boundary(text, (start + max_chars).min(len));

        let cut = if end < len {
            find_paragraph_break_in(text, start, end)
                .or_else(|| find_word_break_in(text, start, end))
                .unwrap_or(end)
        } else {
            end
        };

        // Guarantee cut is a valid char boundary and makes forward progress.
        let cut = {
            let c = floor_char_boundary(text, cut.min(len));
            if c <= start {
                advance_char_boundary(text, start + 1)
            } else {
                c
            }
        };

        windows.push(text[start..cut].to_string());
        if cut >= len {
            break;
        }
        let step = (cut - start).saturating_sub(overlap).max(1);
        start = advance_char_boundary(text, start + step);
    }

    windows
}

/// Round `pos` down to the nearest char boundary (≤ pos).
fn floor_char_boundary(text: &str, pos: usize) -> usize {
    let mut p = pos.min(text.len());
    while p > 0 && !text.is_char_boundary(p) {
        p -= 1;
    }
    p
}

/// Round `pos` up to the nearest char boundary (≥ pos).
fn advance_char_boundary(text: &str, pos: usize) -> usize {
    let len = text.len();
    let mut p = pos.min(len);
    while p < len && !text.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// Find the last "\n\n" in `text[start..end]`, return absolute offset after it.
fn find_paragraph_break_in(text: &str, start: usize, end: usize) -> Option<usize> {
    let end = floor_char_boundary(text, end.min(text.len()));
    if start >= end {
        return None;
    }
    // "\n\n" is two ASCII bytes — result is always on a char boundary.
    text[start..end].rfind("\n\n").map(|rel| start + rel + 2)
}

/// Find the last whitespace char in `text[start..end]`, return absolute offset after it.
fn find_word_break_in(text: &str, start: usize, end: usize) -> Option<usize> {
    let end = floor_char_boundary(text, end.min(text.len()));
    if start >= end {
        return None;
    }
    let slice = &text[start..end];
    slice.rfind(char::is_whitespace).map(|rel| {
        // Use len_utf8() so multi-byte whitespace is skipped correctly.
        let ch = slice[rel..].chars().next().unwrap();
        start + rel + ch.len_utf8()
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_section_no_headings() {
        let chunks = chunk_markdown("test.md", "hello world\nsome text here", 2000);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("hello world"));
    }

    #[test]
    fn splits_on_headings() {
        // Two small sections fit comfortably under max_chars=2000, so the
        // greedy merger combines them into a single chunk that still carries
        // both heading breadcrumbs.
        let md = "# Alpha\n\nalpha body\n\n## Beta\n\nbeta body\n";
        let chunks = chunk_markdown("test.md", md, 2000);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Alpha"));
        assert!(chunks[0].text.contains("Beta"));
    }

    #[test]
    fn small_sections_merge_until_max_chars() {
        // Three small sections that fit together; one chunk total.
        let md = "# A\n\na body\n\n# B\n\nb body\n\n# C\n\nc body\n";
        let chunks = chunk_markdown("test.md", md, 2000);
        assert_eq!(chunks.len(), 1);

        // Tight max forces a flush between sections (use larger bodies so
        // each individual section still clears MIN_CHUNK_CHARS).
        let md = "# A\n\naaaaaaaaaaaaaaaaaaaaaaaa\n\n# B\n\nbbbbbbbbbbbbbbbbbbbbbbbb\n";
        let chunks = chunk_markdown("test.md", md, 40);
        assert!(
            chunks.len() >= 2,
            "expected multiple chunks, got {chunks:?}"
        );
    }

    #[test]
    fn breadcrumb_carried_into_subsection() {
        let md = "# Ops\n\n## Deploy\n\ndeploy body\n";
        let chunks = chunk_markdown("test.md", md, 2000);
        // "## Deploy" chunk should carry "# Ops" breadcrumb
        let deploy_chunk = chunks.iter().find(|c| c.text.contains("Deploy")).unwrap();
        assert!(deploy_chunk.text.contains("# Ops"), "{}", deploy_chunk.text);
    }

    #[test]
    fn large_section_is_windowed() {
        let body = "word ".repeat(600); // ~3000 chars
        let md = format!("# Big\n\n{body}");
        let chunks = chunk_markdown("test.md", &md, 500);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn multibyte_chars_at_window_boundary_do_not_panic() {
        // '─' is U+2500, 3 bytes (0xE2 0x94 0x80).
        // Pad so that max_chars lands inside this character.
        let pad = "x".repeat(1799);
        let content = format!("# Section\n\n{pad}─────────────────────");
        // Must not panic regardless of where the window boundary falls.
        let chunks = chunk_markdown("test.md", &content, 1800);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            // Ensure each chunk is valid UTF-8 (would panic on bad slicing).
            assert!(std::str::from_utf8(chunk.text.as_bytes()).is_ok());
        }
    }

    #[test]
    fn short_stubs_dropped() {
        let md = "# A\n\n\n# B\n\nsome real content here that is long enough";
        let chunks = chunk_markdown("test.md", md, 2000);
        // Section A has only whitespace body; its heading alone may be < MIN_CHUNK_CHARS
        // Section B has content — should survive.
        assert!(chunks.iter().any(|c| c.text.contains("some real content")));
    }

    // -----------------------------------------------------------------------
    // Code chunker tests
    // -----------------------------------------------------------------------

    #[test]
    fn rust_splits_on_fn_and_struct() {
        let src = r#"use std::collections::HashMap;

struct Foo {
    x: i32,
}

fn bar(a: i32) -> i32 {
    a + 1
}

pub fn baz() {
    println!("hi");
}
"#;
        let chunks = chunk_code("lib.rs", src, 2000, Language::Rust);
        // Should produce: preamble (use), struct Foo, fn bar, pub fn baz
        assert!(chunks.iter().any(|c| c.text.contains("struct Foo")));
        assert!(chunks.iter().any(|c| c.text.contains("fn bar")));
        assert!(chunks.iter().any(|c| c.text.contains("pub fn baz")));
    }

    #[test]
    fn python_splits_on_def_and_class() {
        let src = "import os\n\nclass MyClass:\n    pass\n\ndef my_func():\n    return 42\n";
        let chunks = chunk_code("mod.py", src, 2000, Language::Python);
        assert!(chunks.iter().any(|c| c.text.contains("class MyClass")));
        assert!(chunks.iter().any(|c| c.text.contains("def my_func")));
    }

    #[test]
    fn go_splits_on_func_and_type() {
        let src = "package main\n\ntype Point struct {\n\tX, Y int\n}\n\nfunc New(x, y int) Point {\n\treturn Point{x, y}\n}\n";
        let chunks = chunk_code("main.go", src, 2000, Language::Go);
        assert!(chunks.iter().any(|c| c.text.contains("type Point")));
        assert!(chunks.iter().any(|c| c.text.contains("func New")));
    }

    #[test]
    fn code_large_fn_is_windowed() {
        // A single function body that exceeds max_chars should be split with overlap.
        let body = "    let x = 1;\n".repeat(200); // ~3000 chars
        let src = format!("fn big() {{\n{body}}}\n");
        let chunks = chunk_code("big.rs", &src, 500, Language::Rust);
        assert!(chunks.len() > 1);
        // All chunks must be valid UTF-8 (catches bad slicing).
        for c in &chunks {
            assert!(std::str::from_utf8(c.text.as_bytes()).is_ok());
        }
    }

    #[test]
    fn typescript_splits_on_export_function() {
        let src = "import { foo } from './foo';\n\nexport function greet(name: string): string {\n  return `Hello ${name}`;\n}\n\nexport class Greeter {\n  greet() {}\n}\n";
        let chunks = chunk_code("greet.ts", src, 2000, Language::TypeScript);
        assert!(
            chunks
                .iter()
                .any(|c| c.text.contains("export function greet"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.text.contains("export class Greeter"))
        );
    }
}
