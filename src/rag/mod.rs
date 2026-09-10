// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.

pub(crate) mod chunker;
pub(crate) mod encoder;
pub(crate) mod index_io;

use crate::engine::kernels::matmul_f32_embeddings;
use crate::engine::runtime::malloc_run_state;
use crate::engine::types::RunState;
use encoder::{BertPrefillState, embed_raw};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub(crate) use encoder::DocumentEncoder;
type EmbedBatchWithStatsResult = Result<(Vec<(usize, Vec<f32>)>, RagBuildWorkerStats), String>;

#[derive(Debug, Clone)]
struct RagBuildWorkerStats {
    worker_idx: usize,
    chunks: usize,
    tokens: usize,
    max_tokens: usize,
    est_cost: usize,
    elapsed_secs: f64,
}

#[derive(Debug, Clone)]
struct RagBuildTrace {
    chunks: usize,
    unique_chunks: usize,
    dim: usize,
    is_bert_family: bool,
    total_tokens: usize,
    avg_tokens_per_chunk: f64,
    max_tokens: usize,
    chunking_secs: f64,
    tokenization_secs: f64,
    scheduling_secs: f64,
    state_alloc_secs: f64,
    embedding_secs: f64,
    assembly_secs: f64,
    total_secs: f64,
    worker_stats: Vec<RagBuildWorkerStats>,
}

#[derive(Debug, Clone)]
struct ChunkDedupPlan {
    unique_texts: Vec<String>,
    chunk_to_unique: Vec<usize>,
}

impl RagBuildTrace {
    fn print(&self) {
        let arch = if self.is_bert_family {
            "bert-family"
        } else {
            "decoder-family"
        };
        eprintln!(
            "[RAG-BUILD] chunks={} unique_chunks={} dim={} arch={} tokens={} avg_tokens/chunk={:.1} max_tokens={}",
            self.chunks,
            self.unique_chunks,
            self.dim,
            arch,
            self.total_tokens,
            self.avg_tokens_per_chunk,
            self.max_tokens
        );
        eprintln!(
            "[RAG-BUILD] chunking={:.3}s tokenization={:.3}s scheduling={:.3}s state_alloc={:.3}s embedding={:.3}s assembly={:.3}s total={:.3}s",
            self.chunking_secs,
            self.tokenization_secs,
            self.scheduling_secs,
            self.state_alloc_secs,
            self.embedding_secs,
            self.assembly_secs,
            self.total_secs
        );
        for worker in &self.worker_stats {
            let tok_per_sec = if worker.elapsed_secs > 0.0 {
                worker.tokens as f64 / worker.elapsed_secs
            } else {
                0.0
            };
            eprintln!(
                "[RAG-BUILD] worker={} chunks={} tokens={} max_tokens={} est_cost={} elapsed={:.3}s tok/s={:.0}",
                worker.worker_idx,
                worker.chunks,
                worker.tokens,
                worker.max_tokens,
                worker.est_cost,
                worker.elapsed_secs,
                tok_per_sec
            );
        }
    }
}

fn build_chunk_dedup_plan(
    raw_chunks: &[chunker::RawChunk],
    document_prefix: &str,
) -> ChunkDedupPlan {
    let total = raw_chunks.len();
    let mut unique_lookup: std::collections::HashMap<String, usize> =
        std::collections::HashMap::with_capacity(total);
    let mut unique_texts: Vec<String> = Vec::with_capacity(total);
    let mut chunk_to_unique: Vec<usize> = Vec::with_capacity(total);
    for raw in raw_chunks {
        let mut text = String::with_capacity(document_prefix.len() + raw.text.len());
        if !document_prefix.is_empty() {
            text.push_str(document_prefix);
        }
        text.push_str(&raw.text);
        if let Some(&idx) = unique_lookup.get(&text) {
            chunk_to_unique.push(idx);
        } else {
            let idx = unique_texts.len();
            unique_lookup.insert(text.clone(), idx);
            unique_texts.push(text);
            chunk_to_unique.push(idx);
        }
    }
    ChunkDedupPlan {
        unique_texts,
        chunk_to_unique,
    }
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct RagChunk {
    /// Relative path of the source file within the wiki, e.g. `ops/deploy.md`.
    pub(crate) source: String,
    /// The chunk text injected verbatim into the system prompt.
    pub(crate) text: String,
    /// L2-normalised embedding vector (length = index `dim`).
    pub(crate) embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// RagIndex
// ---------------------------------------------------------------------------

/// In-memory RAG index: a flat list of chunks with a row-major embedding matrix
/// for fast batched cosine similarity search.
pub(crate) struct RagIndex {
    chunks: Vec<RagChunk>,
    dim: usize,
    /// Row-major [n_chunks × dim] matrix — ready for `matmul_f32_embeddings`.
    flat_embeddings: Vec<f32>,
}

impl RagIndex {
    /// Build an index from `wiki_dir`, embedding each chunk with `encoder`.
    /// Prints progress to stderr.
    pub(crate) fn build_from_dir(
        wiki_dir: &std::path::Path,
        encoder: &mut DocumentEncoder,
        max_chars_per_chunk: usize,
        max_tokens_per_chunk: usize,
        progress: Option<std::sync::Arc<dyn Fn(String) + Send + Sync>>,
        trace_build: bool,
    ) -> Result<Self, String> {
        let chunk_t0 = Instant::now();
        let raw_chunks = chunker::chunk_directory(wiki_dir, max_chars_per_chunk)?;
        let chunking_secs = chunk_t0.elapsed().as_secs_f64();
        if raw_chunks.is_empty() {
            return Err(format!(
                "no markdown files found under '{}'",
                wiki_dir.display()
            ));
        }
        Self::build_from_raw_chunks(
            raw_chunks,
            encoder,
            max_tokens_per_chunk,
            progress,
            trace_build,
            chunking_secs,
        )
    }

    /// Shared parallel pipeline used by both `build_from_dir` and
    /// `build_from_text_slices`.  `chunking_secs` is purely cosmetic — it just
    /// labels the chunking phase in the optional trace output.
    fn build_from_raw_chunks(
        raw_chunks: Vec<chunker::RawChunk>,
        encoder: &mut DocumentEncoder,
        max_tokens_per_chunk: usize,
        progress: Option<std::sync::Arc<dyn Fn(String) + Send + Sync>>,
        trace_build: bool,
        chunking_secs: f64,
    ) -> Result<Self, String> {
        let build_started = Instant::now();
        let total = raw_chunks.len();
        if total == 0 {
            return Err("no chunks to index".to_string());
        }
        let dim = encoder.dim();

        // ── Phase 0: deduplicate identical chunk texts ─────────────────────
        // Large doc trees often repeat boilerplate blocks or generated content.
        // Tokenize/embed each exact chunk body once, then fan the shared result
        // back out when assembling the final index.
        let dedup_plan = build_chunk_dedup_plan(&raw_chunks, encoder.document_prefix());
        let unique_texts = dedup_plan.unique_texts;
        let chunk_to_unique = dedup_plan.chunk_to_unique;
        let unique_total = unique_texts.len();

        // ── Phase 1: tokenise all chunks in parallel ───────────────────────
        // The tokenizer lazily builds lookup tables, so prime those once on
        // the main thread, then share the read-only tokenizer across rayon
        // workers for the steady-state encode path.
        let token_t0 = Instant::now();
        encoder.prepare_tokenizer();
        let all_token_ids: Vec<Vec<i32>> = {
            let tokenizer = encoder.prepared_tokenizer();
            unique_texts
                .par_iter()
                .map_init(Vec::new, |ids, text| {
                    tokenizer.encode_prepared(text, ids);
                    if max_tokens_per_chunk > 0 && ids.len() > max_tokens_per_chunk {
                        ids.truncate(max_tokens_per_chunk);
                    }
                    ids.clone()
                })
                .collect()
        };
        let tokenization_secs = token_t0.elapsed().as_secs_f64();
        let total_tokens: usize = chunk_to_unique
            .iter()
            .map(|&idx| all_token_ids[idx].len())
            .sum();
        let max_tokens = all_token_ids.iter().map(Vec::len).max().unwrap_or(0);
        let avg_tokens_per_chunk = total_tokens as f64 / total as f64;

        let ctx = encoder.embed_context();
        let done = AtomicUsize::new(0);
        let start = Instant::now();
        let num_threads = rayon::current_num_threads().min(unique_total).max(1);
        let scheduling_secs = 0.0;
        let work_grain = if ctx.config.is_bert_family {
            1usize
        } else {
            4usize
        };
        let next_chunk = AtomicUsize::new(0);

        // ── Phase 2: embed chunks in parallel ──────────────────────────────
        // Each rayon worker gets its own RunState and BertPrefillState so there
        // is no sharing or locking during inference.  The BertPrefillState buffers
        // grow to the largest chunk seen by that worker and are reused for every
        // subsequent chunk, eliminating per-call heap allocation.
        //
        // The serial parts of embed_prefill_bert (attention, RoPE, LayerNorm)
        // are not rayon-parallelised internally, so keeping the outer parallel
        // loop is essential — it is what puts those serial sections on different
        // cores concurrently.
        let state_alloc_t0 = Instant::now();
        let workers: Vec<(RunState, BertPrefillState)> = (0..num_threads)
            .map(|_| malloc_run_state(ctx.config).map(|rs| (rs, BertPrefillState::new())))
            .collect::<Result<Vec<_>, _>>()?;
        let state_alloc_secs = state_alloc_t0.elapsed().as_secs_f64();

        let embed_t0 = Instant::now();
        let batches: Vec<EmbedBatchWithStatsResult> = workers
            .into_par_iter()
            .enumerate()
            .map(|(worker_idx, (mut rs, mut bs))| {
                let worker_started = Instant::now();
                let mut results = Vec::new();
                let mut batch_chunks = 0usize;
                let mut batch_tokens = 0usize;
                let mut batch_max_tokens = 0usize;
                let mut batch_est_cost = 0usize;

                loop {
                    let lo = next_chunk.fetch_add(work_grain, Ordering::Relaxed);
                    if lo >= unique_total {
                        break;
                    }
                    let hi = (lo + work_grain).min(unique_total);
                    for (offset, ids) in all_token_ids[lo..hi].iter().enumerate() {
                        let idx = lo + offset;
                        batch_chunks += 1;
                        batch_tokens += ids.len();
                        batch_max_tokens = batch_max_tokens.max(ids.len());
                        batch_est_cost += ids.len().saturating_mul(ids.len());
                        let emb = embed_raw(ids, &ctx, &mut rs, &mut bs)?;
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_multiple_of(10) || n == unique_total {
                            let elapsed = start.elapsed().as_secs_f64();
                            let rate = n as f64 / elapsed;
                            let eta = ((unique_total - n) as f64 / rate) as u64;
                            let msg = format!(
                                "{n}/{unique_total} unique chunks  {rate:.0} chunks/s  ETA {eta}s"
                            );
                            if let Some(cb) = &progress {
                                cb(msg);
                            } else {
                                eprint!("\r\x1b[2K  {msg}");
                            }
                        }
                        results.push((idx, emb));
                    }
                }

                Ok((
                    results,
                    RagBuildWorkerStats {
                        worker_idx,
                        chunks: batch_chunks,
                        tokens: batch_tokens,
                        max_tokens: batch_max_tokens,
                        est_cost: batch_est_cost,
                        elapsed_secs: worker_started.elapsed().as_secs_f64(),
                    },
                ))
            })
            .collect();
        let embedding_secs = embed_t0.elapsed().as_secs_f64();

        if progress.is_none() {
            eprint!("\r\x1b[2K");
        }

        let assembly_t0 = Instant::now();
        let mut embeddings: Vec<Option<Vec<f32>>> = (0..unique_total).map(|_| None).collect();
        let mut worker_stats = Vec::with_capacity(num_threads);
        for (batch, stats) in batches.into_iter().collect::<Result<Vec<_>, _>>()? {
            for (idx, embedding) in batch {
                embeddings[idx] = Some(embedding);
            }
            worker_stats.push(stats);
        }

        let mut chunks: Vec<RagChunk> = Vec::with_capacity(total);
        let mut flat_embeddings: Vec<f32> = Vec::with_capacity(total * dim);
        for (raw, unique_idx) in raw_chunks.into_iter().zip(chunk_to_unique) {
            let embedding = embeddings[unique_idx]
                .as_ref()
                .ok_or_else(|| "missing RAG embedding result".to_string())?
                .clone();
            flat_embeddings.extend_from_slice(&embedding);
            chunks.push(RagChunk {
                source: raw.source,
                text: raw.text,
                embedding,
            });
        }
        let assembly_secs = assembly_t0.elapsed().as_secs_f64();

        if trace_build {
            worker_stats.sort_unstable_by(|a, b| {
                b.elapsed_secs
                    .partial_cmp(&a.elapsed_secs)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.worker_idx.cmp(&b.worker_idx))
            });
            RagBuildTrace {
                chunks: total,
                unique_chunks: unique_total,
                dim,
                is_bert_family: ctx.config.is_bert_family,
                total_tokens,
                avg_tokens_per_chunk,
                max_tokens,
                chunking_secs,
                tokenization_secs,
                scheduling_secs,
                state_alloc_secs,
                embedding_secs,
                assembly_secs,
                total_secs: build_started.elapsed().as_secs_f64(),
                worker_stats,
            }
            .print();
        }

        Ok(Self {
            chunks,
            dim,
            flat_embeddings,
        })
    }

    /// Build an index from in-memory `(source_name, markdown_content)` pairs.
    ///
    /// Each pair is chunked with [`chunker::chunk_markdown`] and embedded.
    /// Useful when the source documents are embedded in the binary at compile time.
    pub(crate) fn build_from_text_slices(
        docs: &[(&str, &str)],
        encoder: &mut encoder::DocumentEncoder,
        max_chars_per_chunk: usize,
        max_tokens_per_chunk: usize,
    ) -> Result<Self, String> {
        let chunk_t0 = Instant::now();
        let raw_chunks: Vec<chunker::RawChunk> = docs
            .iter()
            .flat_map(|(source, content)| {
                chunker::chunk_markdown(source, content, max_chars_per_chunk)
            })
            .collect();
        let chunking_secs = chunk_t0.elapsed().as_secs_f64();
        Self::build_from_raw_chunks(
            raw_chunks,
            encoder,
            max_tokens_per_chunk,
            None,
            false,
            chunking_secs,
        )
    }

    /// Load a pre-built index from a `.ragidx` file.
    pub(crate) fn load(path: &std::path::Path) -> Result<Self, String> {
        let (dim, chunks) = index_io::load(path)?;
        let flat_embeddings = chunks
            .iter()
            .flat_map(|c| c.embedding.iter().copied())
            .collect();
        Ok(Self {
            chunks,
            dim,
            flat_embeddings,
        })
    }

    /// Persist the index to a `.ragidx` file.
    pub(crate) fn save(&self, path: &std::path::Path) -> Result<(), String> {
        index_io::save(path, self.dim, &self.chunks)
    }

    /// Load a pre-built index from an in-memory byte slice (e.g. `include_bytes!`).
    pub(crate) fn load_from_bytes(data: &[u8]) -> Result<Self, String> {
        let (dim, chunks) = index_io::load_from_bytes(data)?;
        let flat_embeddings = chunks
            .iter()
            .flat_map(|c| c.embedding.iter().copied())
            .collect();
        Ok(Self {
            chunks,
            dim,
            flat_embeddings,
        })
    }

    /// Serialize the index into a `Vec<u8>` in the `.ragidx` wire format.
    /// Suitable for embedding into a binary at build time.
    pub(crate) fn save_to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut buf: Vec<u8> = Vec::new();
        index_io::save_to_writer(&mut buf, self.dim, &self.chunks)?;
        Ok(buf)
    }

    /// Number of chunks in the index.
    pub(crate) fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Retrieve the `top_k` most similar chunks for `query_embedding`, augmented with
    /// keyword rescue for any significant query words not captured by semantic search.
    ///
    /// `query_embedding` must be L2-normalised (as produced by `DocumentEncoder::embed`).
    /// `query_text` is the raw query string used for keyword matching.
    /// Returns chunks sorted by descending cosine similarity (semantic results first,
    /// then any keyword-rescued chunks that didn't make the semantic top-k).
    pub(crate) fn query(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> Vec<&RagChunk> {
        if self.chunks.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let n = self.chunks.len();
        let k = top_k.min(n);
        let mut scores = vec![0f32; n];
        // All embeddings are L2-normalised → dot product == cosine similarity.
        matmul_f32_embeddings(
            &mut scores,
            query_embedding,
            &self.flat_embeddings,
            n,
            self.dim,
        );

        // Semantic top-k.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected: Vec<usize> = indices[..k].to_vec();
        let selected_set: std::collections::HashSet<usize> = selected.iter().copied().collect();

        // Keyword rescue: find chunks containing significant query words that semantic
        // search missed.  Limits to at most `k` additional chunks.
        let keywords = extract_keywords(query_text);
        if !keywords.is_empty() {
            let mut kw_scored: Vec<(usize, usize)> = self
                .chunks
                .iter()
                .enumerate()
                .filter(|(i, _)| !selected_set.contains(i))
                .filter_map(|(i, chunk)| {
                    let lower = chunk.text.to_ascii_lowercase();
                    let hits = keywords
                        .iter()
                        .filter(|kw| lower.contains(kw.as_str()))
                        .count();
                    if hits > 0 { Some((i, hits)) } else { None }
                })
                .collect();
            kw_scored.sort_unstable_by_key(|b| std::cmp::Reverse(b.1));
            for (i, _) in kw_scored.into_iter().take(k) {
                selected.push(i);
            }
        }

        selected.iter().map(|&i| &self.chunks[i]).collect()
    }
}

/// Extract significant keywords from a query string.
/// Strips punctuation, lowercases, and removes common English stop words.
fn extract_keywords(query: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "need", "dare", "ought", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "their", "this", "that", "these", "those", "what", "who", "which", "where",
        "when", "why", "how", "and", "or", "but", "not", "no", "nor", "so", "yet", "for", "in",
        "on", "at", "to", "of", "by", "as", "with", "from", "about", "into", "than", "then",
        "there",
    ];
    query
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| w.len() >= 3 && !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

// ---------------------------------------------------------------------------
// Context injection helpers
// ---------------------------------------------------------------------------

/// Prepend retrieved chunks to `system_prompt` as a `<knowledge>` block.
pub(crate) fn prepend_rag_context(chunks: &[&RagChunk], system_prompt: &str) -> String {
    if chunks.is_empty() {
        return system_prompt.to_string();
    }

    let mut block = String::from("<knowledge>\n");
    for chunk in chunks {
        block.push('[');
        block.push_str(&chunk.source);
        block.push_str("]\n");
        block.push_str(&chunk.text);
        block.push_str("\n\n");
    }
    // Trim the trailing blank line inside the tag.
    let block = block.trim_end_matches('\n');
    let block = format!("{block}\n</knowledge>");

    if system_prompt.is_empty() {
        block
    } else {
        format!("{block}\n\n{system_prompt}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    fn make_chunk(source: &str, text: &str, emb: Vec<f32>) -> RagChunk {
        RagChunk {
            source: source.to_string(),
            text: text.to_string(),
            embedding: emb,
        }
    }

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn query_returns_top_k() {
        let dim = 4;
        let chunks = vec![
            make_chunk("a.md", "alpha", unit(vec![1.0, 0.0, 0.0, 0.0])),
            make_chunk("b.md", "beta", unit(vec![0.0, 1.0, 0.0, 0.0])),
            make_chunk("c.md", "gamma", unit(vec![0.0, 0.0, 1.0, 0.0])),
        ];
        let flat: Vec<f32> = chunks.iter().flat_map(|c| c.embedding.clone()).collect();
        let index = RagIndex {
            chunks,
            dim,
            flat_embeddings: flat,
        };
        let query = unit(vec![1.0, 0.1, 0.0, 0.0]);
        let results = index.query(&query, "alpha test", 2);
        assert_eq!(results.len(), 2);
        // "alpha" should rank first.
        assert_eq!(results[0].source, "a.md");
    }

    #[test]
    fn prepend_context_format() {
        let chunk = RagChunk {
            source: "ops/deploy.md".to_string(),
            text: "Deploy with ./deploy.sh".to_string(),
            embedding: vec![],
        };
        let out = prepend_rag_context(&[&chunk], "Be helpful.");
        assert!(out.starts_with("<knowledge>"));
        assert!(out.contains("[ops/deploy.md]"));
        assert!(out.contains("Deploy with ./deploy.sh"));
        assert!(out.contains("</knowledge>"));
        assert!(out.contains("Be helpful."));
    }

    #[test]
    fn prepend_context_no_chunks() {
        let out = prepend_rag_context(&[], "original");
        assert_eq!(out, "original");
    }

    #[test]
    fn dedup_plan_reuses_exact_prefixed_texts() {
        let raw_chunks = vec![
            chunker::RawChunk {
                source: "a.md".to_string(),
                text: "same".to_string(),
            },
            chunker::RawChunk {
                source: "b.md".to_string(),
                text: "same".to_string(),
            },
            chunker::RawChunk {
                source: "c.md".to_string(),
                text: "different".to_string(),
            },
        ];

        let plan = build_chunk_dedup_plan(&raw_chunks, "search_document: ");

        assert_eq!(plan.unique_texts.len(), 2);
        assert_eq!(plan.chunk_to_unique, vec![0, 0, 1]);
        assert_eq!(plan.unique_texts[0], "search_document: same");
        assert_eq!(plan.unique_texts[1], "search_document: different");
    }

    fn count_source_files(dir: &Path) -> Result<usize, String> {
        let mut total = 0usize;
        let mut stack = vec![dir.to_path_buf()];
        while let Some(path) = stack.pop() {
            let entries = std::fs::read_dir(&path)
                .map_err(|e| format!("cannot read directory '{}': {e}", path.display()))?;
            for entry in entries {
                let entry = entry.map_err(|e| {
                    format!(
                        "cannot read directory entry under '{}': {e}",
                        path.display()
                    )
                })?;
                let child = entry.path();
                if child.is_dir() {
                    stack.push(child);
                } else if let Some(ext) = child.extension().and_then(|s| s.to_str())
                    && (ext == "md"
                        || matches!(
                            ext,
                            "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "go" | "c" | "h" | "java"
                        ))
                {
                    total += 1;
                }
            }
        }
        Ok(total)
    }

    fn dedup_report_for_dir(
        dir: &Path,
        max_chars: usize,
        document_prefix: &str,
    ) -> Result<String, String> {
        let raw_chunks = chunker::chunk_directory(dir, max_chars)?;
        let plan = build_chunk_dedup_plan(&raw_chunks, document_prefix);
        let total_chunks = raw_chunks.len();
        let unique_chunks = plan.unique_texts.len();
        let duplicate_chunks = total_chunks.saturating_sub(unique_chunks);
        let duplicate_ratio = if total_chunks > 0 {
            duplicate_chunks as f64 / total_chunks as f64
        } else {
            0.0
        };
        let mut reuse_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for &idx in &plan.chunk_to_unique {
            *reuse_counts.entry(idx).or_insert(0) += 1;
        }
        let reused_texts = reuse_counts.values().filter(|&&count| count > 1).count();
        let max_reuse = reuse_counts.values().copied().max().unwrap_or(0);
        let source_files = count_source_files(dir)?;
        Ok(format!(
            "RAG_DEDUP source_dir={} source_files={} max_chars={} chunks={} unique_chunks={} duplicate_chunks={} duplicate_ratio={:.4} reused_texts={} max_reuse={}",
            dir.display(),
            source_files,
            max_chars,
            total_chunks,
            unique_chunks,
            duplicate_chunks,
            duplicate_ratio,
            reused_texts,
            max_reuse
        ))
    }

    fn dedup_source_dir_from_env() -> Result<PathBuf, String> {
        std::env::var_os("RAG_DEDUP_SOURCE_DIR")
            .map(PathBuf::from)
            .ok_or_else(|| "RAG_DEDUP_SOURCE_DIR is not set".to_string())
    }

    #[test]
    #[ignore]
    fn report_source_chunk_dedup_1200() {
        let dir = dedup_source_dir_from_env().expect("source dir env");
        let line = dedup_report_for_dir(&dir, 1200, "").expect("dedup report");
        eprintln!("{line}");
    }

    #[test]
    #[ignore]
    fn report_source_chunk_dedup_1800() {
        let dir = dedup_source_dir_from_env().expect("source dir env");
        let line = dedup_report_for_dir(&dir, 1800, "").expect("dedup report");
        eprintln!("{line}");
    }
}
