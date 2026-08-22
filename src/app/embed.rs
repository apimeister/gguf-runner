// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

use crate::app::events::{RuntimeEvent, RuntimeEventCallback, RuntimeProgress};
use crate::app::generation::ModelRuntime;
pub use crate::engine::types::AudioTranscriptionResult;
use crate::vendors::{ChatMessage, ChatRole};

/// Shared, mutable token sink forwarded to by a [`TurnStream`]. Boxed so any
/// `FnMut(&str)` closure can be stored, `Mutex` for interior mutability across
/// the runtime callback, and `Arc` so it can be reused across tool-call turns.
type TokenSink = Arc<Mutex<Box<dyn FnMut(&str) + Send>>>;

/// Timing / throughput stats for one [`EmbeddedRuntime::generate_with_tools_streaming`]
/// call, aggregated across any tool-call turns. Surfaced so an interactive caller can
/// print a status line (prefill / decode token counts and the decode rate).
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationStats {
    /// Prompt tokens processed (prefill), summed over all turns.
    pub prefill_tokens: usize,
    /// Tokens generated (decode), summed over all turns.
    pub decode_tokens: usize,
    /// Hidden-think tokens (included in `decode_tokens`), summed over all turns.
    pub hidden_think_tokens: usize,
    /// Decode throughput of the final turn, tokens/second (`None` if unmeasured).
    pub tokens_per_second: Option<f64>,
    /// KV-cache context used at the end of the final turn.
    pub context_used: usize,
    /// KV-cache context limit.
    pub context_limit: usize,
}

/// A tool that can be called by the model during generation.
///
/// Implement this trait and pass a slice of `&mut dyn Tool` to
/// [`EmbeddedRuntime::generate_with_tools`] to give the model access to
/// custom functionality (web search, database lookups, calculations, etc.).
///
/// The model communicates via a plain JSON protocol:
/// - To call a tool: `{"type":"tool_call","tool":"<name>","args":{...}}`
/// - To signal completion: `{"type":"final"}`
///
/// After the model signals completion a final natural-language response is
/// generated and streamed back to the caller.
pub trait Tool {
    /// Unique name the model uses to invoke this tool.
    fn name(&self) -> &str;

    /// Human-readable description shown to the model in the system prompt.
    /// Be specific: what the tool does, what arguments it expects.
    fn description(&self) -> &str;

    /// Execute the tool with the arguments the model provided.
    ///
    /// Return a JSON value that will be fed back to the model as the tool
    /// result. On error, return `Err` — the error message is also fed back
    /// so the model can adapt.
    fn call(&mut self, args: &serde_json::Value) -> Result<serde_json::Value, String>;
}

/// A loaded model ready for stateless multi-turn generation.
///
/// Callers manage conversation history as plain `(user, assistant)` string
/// pairs. `EmbeddedRuntime` handles chat-template encoding, context trimming,
/// and token streaming internally.
pub struct EmbeddedRuntime {
    inner: ModelRuntime,
}

impl EmbeddedRuntime {
    /// Load a model from bytes compiled into the binary (e.g. `include_bytes!`).
    ///
    /// Blocks until the model is fully loaded — weights are mapped and
    /// verified. Call once and reuse the returned instance.
    ///
    /// Defaults to **greedy decoding** (`temperature = 0.0`) for predictable,
    /// reproducible output. Use [`set_temperature`](Self::set_temperature) and
    /// the other setters below to opt into stochastic sampling.
    pub fn load_from_bytes(data: &'static [u8]) -> Result<Self, String> {
        Ok(Self {
            inner: ModelRuntime::load_from_bytes(data)?,
        })
    }

    /// Load a model from a filesystem path at runtime.
    ///
    /// Use this instead of [`load_from_bytes`](Self::load_from_bytes) when the
    /// model is too large to embed at compile time (e.g. multimodal vision
    /// models of 2 GB+).  The real path is preserved internally so that mmproj
    /// sidecar files are discovered automatically the first time an image is
    /// passed to [`generate_with_image`](Self::generate_with_image).
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, String> {
        Ok(Self {
            inner: ModelRuntime::load_from_file(path, false)?,
        })
    }

    /// Load a multimodal projector from bytes embedded in the binary.
    ///
    /// Call this once after [`load_from_bytes`] to load a vision encoder or a
    /// Qwen3-ASR audio encoder without a sidecar file on disk. The bytes must be
    /// a valid GGUF mmproj file passed via `include_bytes!`.
    pub fn load_mmproj_from_bytes(&mut self, data: &'static [u8]) -> Result<(), String> {
        self.inner.load_mmproj_from_bytes(data)
    }

    /// Generate text for an image + prompt pair, returning the complete output.
    ///
    /// The image must be a file on the local filesystem — pass the path as-is;
    /// gguf-runner reads and preprocesses the image internally.
    ///
    /// On the first call the mmproj vision projector sidecar is located next to
    /// the model file and loaded; subsequent calls reuse it.
    pub fn generate_with_image(
        &mut self,
        image_path: &std::path::Path,
        prompt: &str,
        system_prompt: &str,
    ) -> Result<String, String> {
        let image_str = image_path
            .to_str()
            .ok_or_else(|| "image path contains non-UTF8 characters".to_string())?;
        self.inner
            .generate_text_with_images(prompt, system_prompt, &[image_str.to_string()], false)
    }

    /// Transcribe one finite audio file with an optional context hint and
    /// forced language.
    ///
    /// The first supported backend is Qwen3-ASR. The model and matching audio
    /// mmproj sidecar stay loaded, so callers can invoke this method repeatedly
    /// for offline batch processing without reloading weights.
    pub fn transcribe_audio(
        &mut self,
        audio_path: &std::path::Path,
        context: &str,
        language: Option<&str>,
    ) -> Result<AudioTranscriptionResult, String> {
        let audio_str = audio_path
            .to_str()
            .ok_or_else(|| "audio path contains non-UTF8 characters".to_string())?;
        self.inner.transcribe_audio(audio_str, context, language)
    }

    /// Set the sampling temperature.
    ///
    /// `0.0` (default) → greedy decoding: always picks the highest-probability
    /// token. Reproducible regardless of seed.
    ///
    /// `>0.0` → stochastic sampling. Higher values produce more diverse output
    /// but can degrade quality on heavily quantized models.
    pub fn set_temperature(&mut self, temperature: f32) -> &mut Self {
        self.inner.set_temperature(temperature);
        self
    }

    /// Restrict sampling to the top `k` highest-probability tokens.
    /// `0` disables the filter. Only takes effect when `temperature > 0`.
    pub fn set_top_k(&mut self, top_k: usize) -> &mut Self {
        self.inner.set_top_k(top_k);
        self
    }

    /// Nucleus sampling cutoff. Only takes effect when `top_k > 0`.
    pub fn set_top_p(&mut self, top_p: f32) -> &mut Self {
        self.inner.set_top_p(top_p);
        self
    }

    /// Penalty applied to tokens that appeared in the recent output.
    /// `1.0` (default) disables the penalty. Values like `1.1`–`1.2`
    /// discourage repetition.
    pub fn set_repeat_penalty(&mut self, repeat_penalty: f32) -> &mut Self {
        self.inner.set_repeat_penalty(repeat_penalty);
        self
    }

    /// Fixed RNG seed for reproducible stochastic sampling. `None` uses a
    /// time-based seed. Ignored when `temperature == 0.0`.
    pub fn set_sampling_seed(&mut self, seed: Option<u64>) -> &mut Self {
        self.inner.set_sampling_seed(seed);
        self
    }

    /// Build a RAG knowledge index from in-memory `(source_name, markdown_content)` pairs.
    ///
    /// `encoder_bytes` is a GGUF embedding model (e.g. `include_bytes!("nomic-embed.gguf")`).
    /// `docs` is a slice of `(source_label, markdown_text)` pairs; the source labels appear
    /// in the context block shown to the model so they should be human-readable paths.
    ///
    /// Every subsequent [`generate`](Self::generate) call will embed the user query,
    /// retrieve the most relevant chunks, and prepend them to the system prompt automatically.
    pub fn load_rag_from_embedded_docs(
        &mut self,
        encoder_bytes: &'static [u8],
        docs: &[(&str, &str)],
    ) -> Result<String, String> {
        self.inner.load_rag_from_embedded_docs(encoder_bytes, docs)
    }

    /// Load a precomputed RAG index that was serialized at build time.
    ///
    /// `encoder_bytes` is the GGUF embedding model used to embed the runtime
    /// query.  `index_bytes` is the output of [`build_serialized_rag_index`] —
    /// a flat byte buffer containing every chunk plus its precomputed embedding.
    ///
    /// Unlike [`load_rag_from_embedded_docs`](Self::load_rag_from_embedded_docs),
    /// this path does **no** per-chunk embedding work at startup, so the model
    /// is ready to serve immediately even for large doc sets.
    pub fn load_rag_from_serialized_bytes(
        &mut self,
        encoder_bytes: &'static [u8],
        index_bytes: &[u8],
    ) -> Result<String, String> {
        self.inner
            .load_rag_from_serialized_bytes(encoder_bytes, index_bytes)
    }

    /// Build a serialized RAG index from `(source, markdown)` pairs and return
    /// its bytes.  Intended to be called from a `build.rs` script so the
    /// resulting bytes can be embedded via `include_bytes!`.
    ///
    /// `encoder_bytes` is the GGUF embedding model used at build time; it must
    /// match (or share a vector space with) the encoder loaded at runtime via
    /// [`load_rag_from_serialized_bytes`].
    pub fn build_serialized_rag_index(
        encoder_bytes: &'static [u8],
        docs: &[(&str, &str)],
        max_chars_per_chunk: usize,
        max_tokens_per_chunk: usize,
    ) -> Result<Vec<u8>, String> {
        let mut encoder = crate::rag::encoder::DocumentEncoder::load_from_bytes(encoder_bytes)?;
        let index = crate::rag::RagIndex::build_from_text_slices(
            docs,
            &mut encoder,
            max_chars_per_chunk,
            max_tokens_per_chunk,
        )?;
        index.save_to_bytes()
    }

    /// Force hidden think mode regardless of what the model's chat template specifies.
    ///
    /// Some models (e.g. Qwen3.5) default to `ThinkMode::No` based on their chat template,
    /// which pre-fills an empty `<think></think>` block. When the model produces no visible
    /// output after that block, there is no retry. Calling this method switches to
    /// `ThinkMode::Hidden` instead, which enables an automatic retry with thinking suppressed
    /// whenever the first pass yields empty output.
    /// Load a prefill-cache blob (see `render_prefill_cache`). When a later
    /// prompt starts with the cached token prefix, its KV/SSM state is
    /// restored instead of prefilled; on any mismatch generation falls back
    /// to a cold prefill.
    pub fn load_prefill_cache(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.inner.set_prefill_cache_bytes(bytes)
    }

    /// Prefill the static prefix implied by `system_prompt` once and return a
    /// serialized snapshot for [`load_prefill_cache`](Self::load_prefill_cache).
    /// Intended for build-time rendering of static prompts.
    pub fn render_prefill_cache(&mut self, system_prompt: &str) -> Result<Vec<u8>, String> {
        self.inner.render_prefill_cache_blob(system_prompt)
    }

    /// Disable thinking entirely (`<think></think>` pre-closed). Recommended
    /// for tool-calling with small models: with thinking enabled they tend to
    /// narrate the intended action inside/after the think block and end the
    /// turn without ever emitting the `<tool_call>`, and long think blocks
    /// starve the visible-answer token budget.
    pub fn use_no_think_mode(&mut self) -> &mut Self {
        self.inner.set_think_mode_no();
        self
    }

    pub fn use_hidden_think_mode(&mut self) -> &mut Self {
        self.inner.set_think_mode_hidden();
        self
    }

    /// Cap the effective context length used for generation and KV cache
    /// allocation.
    ///
    /// Models advertise large native context windows in their GGUF metadata
    /// (e.g. 32 K for Qwen3.5), but most embedded use cases only need a
    /// fraction of that. Shrinking the context reduces the per-generation
    /// KV cache allocation roughly linearly and is the cheapest knob for
    /// trimming memory and prefill time when long conversations are not
    /// expected.
    ///
    /// `0` is a no-op (keeps whatever the GGUF or chat template requested).
    /// Values larger than the model's native context are clamped on a
    /// best-effort basis by the runtime — they will not extend it.
    pub fn set_context_size(&mut self, context_size: usize) -> &mut Self {
        self.inner.set_context_size(context_size);
        self
    }

    /// Cap the number of hidden-mode reasoning tokens per generation.
    ///
    /// Hidden think mode bounds how long the model may reason before its
    /// visible answer; the default derives from the vendor policy, layer
    /// count, and context size (roughly 384–1536 tokens). Interactive callers
    /// keep it tight for latency; background captioning/description workloads
    /// raise it so long multilingual reasoning is not cut short. `0` restores
    /// the vendor-derived default.
    pub fn set_hidden_think_token_cap(&mut self, cap: usize) -> &mut Self {
        self.inner.set_hidden_think_token_cap(cap);
        self
    }

    /// Enable verbose debug logging (prompt tokens, top-k logits per step,
    /// stop-token rank). Prints to stderr. Off by default.
    pub fn set_debug(&mut self, enabled: bool) -> &mut Self {
        self.inner.set_debug_mode(enabled);
        self
    }

    /// Generate a response and stream tokens back through an `mpsc` channel.
    ///
    /// Generation runs synchronously on the calling thread and blocks until
    /// complete. Each text fragment is sent to the channel as it is produced,
    /// and the sender is dropped when generation finishes — so iterating the
    /// returned `Receiver` to completion is all that is needed:
    ///
    /// ```no_run
    /// # let mut rt: gguf_runner::EmbeddedRuntime = todo!();
    /// let rx = rt.generate(&[], "What is 2+2?", "You are helpful")?;
    /// for token in rx {
    ///     print!("{token}");
    /// }
    /// # Ok::<_, String>(())
    /// ```
    ///
    /// `history` is a slice of prior `(user, assistant)` turns. Pass an empty
    /// slice for a single-turn exchange.
    pub fn generate(
        &mut self,
        history: &[(String, String)],
        input: &str,
        system_prompt: &str,
    ) -> Result<Receiver<String>, String> {
        let messages = build_messages(history, input);
        let (tx, rx) = channel::<String>();

        let cb: RuntimeEventCallback = Arc::new(move |event: RuntimeEvent| match event {
            RuntimeEvent::Output(text) => {
                let _ = tx.send(text);
            }
            RuntimeEvent::Log(log) => {
                eprintln!("[debug] {}", log.message);
            }
            _ => {}
        });

        self.inner.set_runtime_event_callback(Some(cb));
        let result = self
            .inner
            .generate_chat_messages_for_repl(&messages, system_prompt);
        // Dropping the stored Arc closes the Sender, so `rx` iteration will end.
        self.inner.set_runtime_event_callback(None);

        result?;
        Ok(rx)
    }

    /// Generate a response and return the complete text as a single `String`.
    ///
    /// Convenience wrapper around [`generate`](Self::generate) for callers
    /// that do not need per-token streaming.
    pub fn generate_collect(
        &mut self,
        history: &[(String, String)],
        input: &str,
        system_prompt: &str,
    ) -> Result<String, String> {
        let rx = self.generate(history, input, system_prompt)?;
        Ok(rx.into_iter().collect())
    }

    /// Generate a response with tool support.
    ///
    /// The model can call any of the provided `tools` zero or more times before
    /// producing a final answer. The call loop runs synchronously; the final
    /// natural-language response is streamed back through the returned
    /// `Receiver<String>` just like [`generate`](Self::generate).
    ///
    /// `max_tool_calls` caps the number of tool invocations per request to
    /// prevent runaway loops.
    ///
    /// # Example
    /// ```no_run
    /// use gguf_runner::Tool;
    /// use serde_json::Value;
    ///
    /// struct WeatherTool;
    /// impl Tool for WeatherTool {
    ///     fn name(&self) -> &str { "get_weather" }
    ///     fn description(&self) -> &str { "Get current weather. Args: {\"city\": string}" }
    ///     fn call(&mut self, args: &Value) -> Result<Value, String> {
    ///         let city = args["city"].as_str().unwrap_or("unknown");
    ///         Ok(serde_json::json!({"temperature": "22°C", "condition": "sunny", "city": city}))
    ///     }
    /// }
    ///
    /// # let mut rt: gguf_runner::EmbeddedRuntime = todo!();
    /// let rx = rt.generate_with_tools(
    ///     &[],
    ///     "What's the weather in Paris?",
    ///     "You are a helpful assistant.",
    ///     &mut [&mut WeatherTool],
    ///     5,
    /// )?;
    /// for token in rx { print!("{token}"); }
    /// # Ok::<_, String>(())
    /// ```
    pub fn generate_with_tools(
        &mut self,
        history: &[(String, String)],
        input: &str,
        system_prompt: &str,
        tools: &mut [&mut dyn Tool],
        max_tool_calls: usize,
    ) -> Result<Receiver<String>, String> {
        let tool_system_prompt = build_tool_system_prompt(system_prompt, tools);

        // Accumulate tool call/result turns as additional history.
        let mut extended_history: Vec<(String, String)> = history.to_vec();
        let mut current_input = input.to_string();
        let mut call_count = 0usize;

        loop {
            let response =
                self.generate_collect(&extended_history, &current_input, &tool_system_prompt)?;

            match extract_tool_call(&response) {
                Some(call) => {
                    if call_count >= max_tool_calls {
                        return Err(format!(
                            "max_tool_calls ({max_tool_calls}) reached before final response"
                        ));
                    }
                    call_count += 1;

                    let result = tools
                        .iter_mut()
                        .find(|t| t.name() == call.name)
                        .map(|t| t.call(&call.arguments))
                        .unwrap_or_else(|| Err(format!("unknown tool: {}", call.name)));

                    let result_json = match result {
                        Ok(v) => v,
                        Err(e) => serde_json::json!({"error": e}),
                    };

                    // Feed the result back to the model wrapped in <tool_response> tags
                    // (the Qwen3 native convention). The model then decides whether to
                    // call another tool or produce the final answer.
                    let tool_result_msg = format!(
                        "<tool_response>\n{}\n</tool_response>",
                        serde_json::to_string(&result_json).unwrap_or_else(|_| "{}".to_string())
                    );
                    extended_history.push((current_input.clone(), response.clone()));
                    current_input = tool_result_msg;
                }

                None => {
                    // No tool call — the response IS the final answer. Strip any
                    // protocol artifacts and stream it back through a channel.
                    let final_text = strip_tool_call_blocks(&response);
                    let (tx, rx) = channel::<String>();
                    if !final_text.is_empty() {
                        let _ = tx.send(final_text);
                    }
                    return Ok(rx);
                }
            }
        }
    }

    /// Like [`generate_with_tools`](Self::generate_with_tools), but streams the
    /// final natural-language answer to `on_token` **token-by-token as it is
    /// produced**, and returns [`GenerationStats`] rather than a channel.
    ///
    /// Tool-call turns run silently: only the model's user-facing prose reaches
    /// `on_token`. A `<tool_call>` block (and everything after it within a turn)
    /// is suppressed, and a partial tag split across streamed fragments is never
    /// emitted. This lets an interactive caller show output immediately instead
    /// of blocking on the whole (possibly multi-turn) tool loop.
    pub fn generate_with_tools_streaming(
        &mut self,
        history: &[(String, String)],
        input: &str,
        system_prompt: &str,
        tools: &mut [&mut dyn Tool],
        max_tool_calls: usize,
        on_token: impl FnMut(&str) + Send + 'static,
    ) -> Result<GenerationStats, String> {
        let tool_system_prompt = build_tool_system_prompt(system_prompt, tools);

        let mut extended_history: Vec<(String, String)> = history.to_vec();
        let mut current_input = input.to_string();
        let mut call_count = 0usize;
        let mut stats = GenerationStats::default();

        // One sink, reused across turns; only the final-answer turn (plus any
        // leading prose before a tool call) actually forwards to it.
        let sink: TokenSink = Arc::new(Mutex::new(Box::new(on_token)));

        loop {
            let messages = build_messages(&extended_history, &current_input);
            let turn = Arc::new(Mutex::new(TurnStream::new(Arc::clone(&sink))));

            let cb_turn = Arc::clone(&turn);
            let cb: RuntimeEventCallback = Arc::new(move |event: RuntimeEvent| {
                if let Ok(mut t) = cb_turn.lock() {
                    match event {
                        RuntimeEvent::Output(text) => t.on_output(&text),
                        RuntimeEvent::Progress(p) => t.last_progress = Some(p),
                        _ => {}
                    }
                }
            });

            self.inner.set_runtime_event_callback(Some(cb));
            let result = self
                .inner
                .generate_chat_messages_for_repl(&messages, &tool_system_prompt);
            self.inner.set_runtime_event_callback(None);
            result?;

            let (raw, last_progress) = {
                let mut t = turn.lock().unwrap();
                t.finish();
                (std::mem::take(&mut t.raw), t.last_progress.take())
            };

            if let Some(p) = last_progress {
                stats.prefill_tokens += p.prefill_tokens;
                stats.decode_tokens += p.decode_tokens;
                stats.hidden_think_tokens += p.hidden_think_tokens;
                stats.tokens_per_second = p.tokens_per_second;
                stats.context_used = p.context_used;
                stats.context_limit = p.context_limit;
            }

            match extract_tool_call(&raw) {
                Some(call) => {
                    if call_count >= max_tool_calls {
                        return Err(format!(
                            "max_tool_calls ({max_tool_calls}) reached before final response"
                        ));
                    }
                    call_count += 1;

                    let result = tools
                        .iter_mut()
                        .find(|t| t.name() == call.name)
                        .map(|t| t.call(&call.arguments))
                        .unwrap_or_else(|| Err(format!("unknown tool: {}", call.name)));
                    let result_json = match result {
                        Ok(v) => v,
                        Err(e) => serde_json::json!({ "error": e }),
                    };
                    let tool_result_msg = format!(
                        "<tool_response>\n{}\n</tool_response>",
                        serde_json::to_string(&result_json).unwrap_or_else(|_| "{}".to_string())
                    );
                    extended_history.push((current_input.clone(), raw));
                    current_input = tool_result_msg;
                }
                None => return Ok(stats),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tool protocol — Qwen3 native format
// ---------------------------------------------------------------------------
//
// Tools are described in the system prompt under a <tools>...</tools> block.
// The model calls a tool by emitting:
//   <tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>
// and may interleave prose. Tool results are fed back wrapped in:
//   <tool_response>...</tool_response>
// If the model produces no <tool_call> tag, the response is treated as the
// final natural-language answer.

const TOOL_CALL_OPEN: &str = "<tool_call>";

/// Per-turn streaming filter for [`EmbeddedRuntime::generate_with_tools_streaming`].
///
/// Forwards the model's user-facing prose to the caller sink as it streams, while
/// withholding a small trailing window so a `<tool_call>` tag straddling two
/// fragments is never partially emitted. Once a complete `<tool_call>` open tag is
/// seen, the remainder of the turn is suppressed (it is protocol, not an answer).
struct TurnStream {
    /// Full raw turn text, including any tool-call block (for `extract_tool_call`).
    raw: String,
    /// Not-yet-forwarded output (may hold a partial `<tool_call>` prefix).
    buf: String,
    /// Set once a `<tool_call>` open tag is seen: suppress the remainder.
    suppressing: bool,
    sink: TokenSink,
    last_progress: Option<RuntimeProgress>,
}

impl TurnStream {
    fn new(sink: TokenSink) -> Self {
        Self {
            raw: String::new(),
            buf: String::new(),
            suppressing: false,
            sink,
            last_progress: None,
        }
    }

    fn on_output(&mut self, text: &str) {
        self.raw.push_str(text);
        if self.suppressing {
            return;
        }
        self.buf.push_str(text);

        // A complete tool-call tag: forward any prose before it, suppress the rest.
        if let Some(pos) = self.buf.find(TOOL_CALL_OPEN) {
            let prose = self.buf[..pos].to_string();
            if !prose.trim().is_empty() {
                self.emit(&prose);
            }
            self.suppressing = true;
            self.buf.clear();
            return;
        }

        // Forward everything except a trailing window that could still grow into
        // a `<tool_call>` tag.
        let keep = partial_tag_suffix_len(&self.buf);
        if self.buf.len() > keep {
            let cut = self.buf.len() - keep;
            let flush: String = self.buf.drain(..cut).collect();
            self.emit(&flush);
        }
    }

    /// Flush any held-back prose at end of turn (no tool call materialized).
    fn finish(&mut self) {
        if !self.suppressing && !self.buf.is_empty() {
            let rest = std::mem::take(&mut self.buf);
            self.emit(&rest);
        }
    }

    fn emit(&self, text: &str) {
        if let Ok(mut guard) = self.sink.lock() {
            let f: &mut (dyn FnMut(&str) + Send) = &mut **guard;
            f(text);
        }
    }
}

/// Largest `k` in `1..TOOL_CALL_OPEN.len()` such that `buf` ends with the first
/// `k` bytes of `<tool_call>` — how many trailing (ASCII) bytes might be the
/// start of a tool-call tag and must be withheld. `0` if none.
fn partial_tag_suffix_len(buf: &str) -> usize {
    let max = (TOOL_CALL_OPEN.len() - 1).min(buf.len());
    (1..=max)
        .rev()
        .find(|&k| buf.as_bytes().ends_with(&TOOL_CALL_OPEN.as_bytes()[..k]))
        .unwrap_or(0)
}

struct ToolCallRequest {
    name: String,
    arguments: serde_json::Value,
}

/// Find a `<tool_call>...</tool_call>` block in `text` and parse its body.
/// Returns `None` if no valid tool call is present.
///
/// Accepts three body formats (in order of preference):
///   1. Qwen3 / generic JSON:    `{"name": "...", "arguments": {...}}`
///   2. Qwen3.5 native XML:      `<function=name><parameter=k>v</parameter>...</function>`
///   3. Loose fallback: any `"name": "..."` substring with optional arguments object.
///
/// Tolerates the model emitting extra `<tool_call>` tags before the body
/// (some small models duplicate the opening tag).
fn extract_tool_call(text: &str) -> Option<ToolCallRequest> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    // Locate the body between the LAST `<tool_call>` open and the FIRST
    // `</tool_call>` close after it. This handles `<tool_call><tool_call>...`.
    let close_idx = text.find(CLOSE)?;
    let before_close = &text[..close_idx];
    let open_idx = before_close.rfind(OPEN)?;
    let body = before_close[open_idx + OPEN.len()..].trim();

    // 1. Strict JSON
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body)
        && let Some(name) = value.get("name").and_then(|v| v.as_str())
    {
        let arguments = value
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        return Some(ToolCallRequest {
            name: name.to_string(),
            arguments,
        });
    }

    // 2. Qwen3.5 native XML
    if let Some(call) = parse_xml_function_call(body) {
        return Some(call);
    }

    // 3. Loose: find `"name": "X"` and treat arguments as missing.
    extract_loose_name(body).map(|name| ToolCallRequest {
        name,
        arguments: serde_json::Value::Object(serde_json::Map::new()),
    })
}

/// Find `"name"\s*:\s*"VALUE"` in `text` and return VALUE. Tolerates malformed
/// JSON around it so we can recover when a small model emits a broken object.
fn extract_loose_name(text: &str) -> Option<String> {
    let key_idx = text.find("\"name\"")?;
    let after_key = &text[key_idx + "\"name\"".len()..];
    let colon_idx = after_key.find(':')?;
    let after_colon = after_key[colon_idx + 1..].trim_start();
    let after_colon = after_colon.strip_prefix('"')?;
    let end_idx = after_colon.find('"')?;
    Some(after_colon[..end_idx].to_string())
}

/// Parse the Qwen3.5 native function call XML:
///   `<function=NAME><parameter=K1>V1</parameter><parameter=K2>V2</parameter></function>`
fn parse_xml_function_call(body: &str) -> Option<ToolCallRequest> {
    let fn_open_marker = "<function=";
    let fn_open_idx = body.find(fn_open_marker)?;
    let after_fn = &body[fn_open_idx + fn_open_marker.len()..];
    let name_end = after_fn.find('>')?;
    let name = after_fn[..name_end].trim().to_string();
    let fn_body = &after_fn[name_end + 1..];
    let fn_body_end = fn_body.find("</function>").unwrap_or(fn_body.len());
    let fn_body = &fn_body[..fn_body_end];

    let mut args = serde_json::Map::new();
    let param_open = "<parameter=";
    let param_close = "</parameter>";
    let mut remaining = fn_body;
    while let Some(p_start) = remaining.find(param_open) {
        let after = &remaining[p_start + param_open.len()..];
        let Some(key_end) = after.find('>') else {
            break;
        };
        let key = after[..key_end].trim().to_string();
        let value_part = &after[key_end + 1..];
        let Some(value_end) = value_part.find(param_close) else {
            break;
        };
        let raw_value = value_part[..value_end].trim();
        // Try to parse as JSON (number, bool, object, array); otherwise treat as string.
        let value = serde_json::from_str::<serde_json::Value>(raw_value)
            .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
        args.insert(key, value);
        remaining = &value_part[value_end + param_close.len()..];
    }
    Some(ToolCallRequest {
        name,
        arguments: serde_json::Value::Object(args),
    })
}

/// Remove any `<tool_call>...</tool_call>` and hallucinated
/// `<tool_response>...</tool_response>` blocks from `text` so the final answer
/// doesn't leak protocol artifacts to the caller.
fn strip_tool_call_blocks(text: &str) -> String {
    let stripped = strip_xml_block(text, "tool_call");
    let stripped = strip_xml_block(&stripped, "tool_response");
    stripped.trim().to_string()
}

fn strip_xml_block(text: &str, tag: &str) -> String {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(open_idx) = remaining.find(&open) {
        out.push_str(&remaining[..open_idx]);
        match remaining[open_idx + open.len()..].find(&close) {
            Some(close_rel) => {
                remaining = &remaining[open_idx + open.len() + close_rel + close.len()..];
            }
            None => {
                // Unterminated tag — drop the rest of the text after the open tag.
                remaining = "";
                break;
            }
        }
    }
    out.push_str(remaining);
    // Also remove orphan closing tags (the model sometimes emits </tool_call>
    // without a matching opening tag, especially on tool-trained models that
    // mix the protocol with prose).
    out.replace(&close, "")
}

fn build_tool_system_prompt(base: &str, tools: &[&mut dyn Tool]) -> String {
    let specs: Vec<(String, String)> = tools
        .iter()
        .map(|t| (t.name().to_string(), t.description().to_string()))
        .collect();
    let spec_refs: Vec<(&str, &str)> = specs
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_str()))
        .collect();
    build_tool_system_prompt_from_specs(base, &spec_refs)
}

/// Compose the tools system prompt from static `(name, description)` pairs.
///
/// Public so a consumer's build step can reproduce the exact runtime prompt
/// text when rendering a prefill cache (see
/// [`EmbeddedRuntime::render_prefill_cache`]) — both paths must be
/// byte-identical or the cached token prefix will not match.
pub fn build_tool_system_prompt_from_specs(base: &str, specs: &[(&str, &str)]) -> String {
    // Use the Qwen3 official `# Tools` block with <tools><tool>...</tool></tools>
    // JSON schemas, the format the model was trained on. The instructions tell
    // the model to use tools only when they're actually relevant to the query,
    // and to write a plain prose answer otherwise — without this, the model
    // tends to call tools eagerly for questions that don't need them.
    // Explicit key order, not `serde_json::json!`: map key order there depends
    // on the crate's `preserve_order` feature, which Cargo unifies per build
    // graph — a build-script render and the runtime binary can disagree,
    // silently breaking prefill-cache prefix matching. `to_string` on the
    // individual strings is used only for escaping and is feature-independent.
    let tool_specs: Vec<String> = specs
        .iter()
        .map(|(name, description)| {
            format!(
                "{{\"name\":{},\"description\":{}}}",
                serde_json::to_string(name).unwrap_or_default(),
                serde_json::to_string(description).unwrap_or_default(),
            )
        })
        .collect();

    format!(
        "{base}\n\n\
# Tools\n\n\
You have access to the following functions, defined as JSON specs inside <tools></tools>:\n\
<tools>\n{}\n</tools>\n\n\
When a question needs information that one of the above functions provides, call it by emitting:\n\
<tool_call>\n{{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}}\n</tool_call>\n\n\
The user (system) will reply with the result wrapped in <tool_response>...</tool_response>. Write your final answer to the user in plain prose using that result. If the question can be answered from general knowledge without any tool, just answer directly.",
        tool_specs.join("\n")
    )
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn build_messages(history: &[(String, String)], input: &str) -> Vec<ChatMessage> {
    let mut messages: Vec<ChatMessage> = history
        .iter()
        .flat_map(|(user, assistant)| {
            [
                ChatMessage {
                    role: ChatRole::User,
                    content: user.clone(),
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: assistant.clone(),
                },
            ]
        })
        .collect();
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: input.to_string(),
    });
    messages
}

#[cfg(test)]
mod stream_filter_tests {
    use super::*;

    /// Feed `fragments` through a `TurnStream` and return `(emitted, suppressed)`.
    fn run(fragments: &[&str]) -> (String, bool) {
        let out = Arc::new(Mutex::new(String::new()));
        let out_cb = Arc::clone(&out);
        let sink: TokenSink = Arc::new(Mutex::new(Box::new(move |s: &str| {
            out_cb.lock().unwrap().push_str(s);
        })));
        let mut ts = TurnStream::new(sink);
        for f in fragments {
            ts.on_output(f);
        }
        ts.finish();
        let emitted = out.lock().unwrap().clone();
        (emitted, ts.suppressing)
    }

    #[test]
    fn plain_prose_streams_fully() {
        let (out, suppressed) = run(&["Hel", "lo, ", "world"]);
        assert_eq!(out, "Hello, world");
        assert!(!suppressed);
    }

    #[test]
    fn pure_tool_call_is_suppressed() {
        let (out, suppressed) = run(&["<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>"]);
        assert_eq!(out, "");
        assert!(suppressed);
    }

    #[test]
    fn tool_call_split_across_fragments_is_suppressed() {
        // The `<tool_call>` open tag straddles fragment boundaries; no partial
        // tag must leak to the sink.
        let (out, suppressed) = run(&["<to", "ol_c", "all>{\"name\":\"x\"}</tool_call>"]);
        assert_eq!(out, "");
        assert!(suppressed);
    }

    #[test]
    fn prose_before_tool_call_streams_then_suppresses() {
        let (out, suppressed) = run(&["Let me check. ", "<tool_call>{}</tool_call>"]);
        assert_eq!(out, "Let me check. ");
        assert!(suppressed);
    }

    #[test]
    fn raw_retains_full_turn_for_extraction() {
        let out = Arc::new(Mutex::new(String::new()));
        let out_cb = Arc::clone(&out);
        let sink: TokenSink = Arc::new(Mutex::new(Box::new(move |s: &str| {
            out_cb.lock().unwrap().push_str(s);
        })));
        let mut ts = TurnStream::new(sink);
        ts.on_output("<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>");
        ts.finish();
        assert!(ts.raw.contains("<tool_call>"));
        assert!(extract_tool_call(&ts.raw).is_some());
    }

    #[test]
    fn partial_tag_suffix_len_holds_prefixes_only() {
        assert_eq!(partial_tag_suffix_len("hello<tool_c"), 7); // "<tool_c"
        assert_eq!(partial_tag_suffix_len("abc<"), 1);
        assert_eq!(partial_tag_suffix_len("plain text"), 0);
        assert_eq!(partial_tag_suffix_len(""), 0);
    }
}
