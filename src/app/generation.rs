use crate::cli::CliOptions;
use crate::engine::io::parse_gguf_file;
use crate::engine::kernels::{argmax, sample, softmax, TopKSampler};
use crate::engine::profiling::{prof_end, prof_start, record_forward_pass, PROF_TRANSFORMER_NS};
use crate::engine::types::{
    Config, GGUFFile, LazyModelLoader, Tokenizer, TransformerWeights, XorShiftRng, GEMMA3_END_TURN,
};
use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn time_in_ms() -> i64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (now.as_secs() * 1000 + (now.subsec_nanos() as u64 / 1_000_000)) as i64
}

pub(crate) struct GenerationSettings {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) repeat_penalty: f32,
    pub(crate) repeat_last_n: usize,
    pub(crate) max_tokens: usize,
    pub(crate) profiling_mode: bool,
    pub(crate) show_tokens: bool,
    pub(crate) debug_mode: bool,
}

pub(crate) struct ModelRuntime {
    gguf: GGUFFile,
    config: Config,
    tokenizer: Tokenizer,
    weights: TransformerWeights,
    settings: GenerationSettings,
    lazy_debug_loader: Option<Arc<LazyModelLoader>>,
    next_lazy_debug_ms: i64,
    kv_cache_format_logged: bool,
}

impl ModelRuntime {
    pub(crate) fn load(cli: &CliOptions) -> Result<Self, String> {
        let mut max_tokens = cli.max_tokens;
        let debug_mode = cli.debug;
        let checkpoint = &cli.model;
        let model_url = cli.url.as_deref();
        if debug_mode {
            eprintln!("Loading GGUF model: {checkpoint}");
            eprintln!(
                "Sampling: temperature={}, top_k={}, top_p={}, repeat_penalty={}, repeat_last_n={}",
                cli.temperature, cli.top_k, cli.top_p, cli.repeat_penalty, cli.repeat_last_n
            );
        }

        let gguf = parse_gguf_file(checkpoint, model_url, debug_mode)?;
        let lazy_debug_loader = gguf.lazy_loader.as_ref().map(Arc::clone);
        let next_lazy_debug_ms = time_in_ms() + 2_000;

        if debug_mode {
            eprintln!(
                "GGUF metadata: version={}, tensors={}, kv={}, tensor_data_start={} bytes",
                gguf.version, gguf.n_tensors, gguf.n_kv, gguf.tensor_data_start
            );
            if let Some(loader) = &lazy_debug_loader {
                eprintln!("{}", loader.debug_stats_line());
            }
        }

        let mut config = crate::vendors::build_config_from_gguf(&gguf, debug_mode)?;
        let mut tokenizer =
            crate::engine::tokenizer::init_tokenizer_from_gguf(&gguf, &mut config, debug_mode)?;
        tokenizer.use_sentencepiece = config.is_gemma3;

        crate::engine::runtime::apply_context_size_overrides(
            &mut config,
            cli.context_size,
            debug_mode,
        );
        if max_tokens == 0 || max_tokens > config.seq_len {
            max_tokens = config.seq_len;
        }

        if let Some(n_threads) = cli.threads {
            crate::engine::runtime::configure_rayon_threads(n_threads, debug_mode);
        }

        let weights = crate::engine::weights::init_weights_from_gguf(&gguf, &config, debug_mode)?;
        let settings = GenerationSettings {
            temperature: cli.temperature,
            top_k: cli.top_k,
            top_p: cli.top_p,
            repeat_penalty: cli.repeat_penalty,
            repeat_last_n: cli.repeat_last_n,
            max_tokens,
            profiling_mode: cli.profiling,
            show_tokens: cli.show_tokens,
            debug_mode,
        };

        Ok(Self {
            gguf,
            config,
            tokenizer,
            weights,
            settings,
            lazy_debug_loader,
            next_lazy_debug_ms,
            kv_cache_format_logged: false,
        })
    }

    pub(crate) fn generate_text(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        stream_stdout: bool,
    ) -> Result<String, String> {
        let temperature = self.settings.temperature;
        let top_k = self.settings.top_k;
        let top_p = self.settings.top_p;
        let repetition_penalty = self.settings.repeat_penalty;
        let repeat_last_n = self.settings.repeat_last_n;
        let max_tokens = self.settings.max_tokens;
        let profiling_mode = self.settings.profiling_mode;
        let show_tokens = self.settings.show_tokens;
        let debug_mode = self.settings.debug_mode;

        let mut prompt_tokens: Vec<i32> = crate::vendors::encode_chat_prompt(
            &mut self.tokenizer,
            &self.config,
            prompt,
            system_prompt,
        );

        if prompt_tokens.is_empty() {
            prompt_tokens.push(self.tokenizer.bos_token);
        }
        if prompt_tokens.len() > self.config.seq_len {
            prompt_tokens.truncate(self.config.seq_len);
        }
        if debug_mode {
            eprintln!("Prompt tokens: {}", prompt_tokens.len());
            let preview = prompt_tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("Prompt token ids: [{preview}]");
        }

        let mut token = prompt_tokens[0];
        let mut next: i32;
        let mut pos = 0usize;
        let mut start = 0i64;

        let mut state = crate::engine::runtime::malloc_run_state(&self.config)?;
        if debug_mode && !self.kv_cache_format_logged {
            eprintln!("KV cache format: {:?}", state.kv_cache_format);
            self.kv_cache_format_logged = true;
        }
        let mut rng = XorShiftRng::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
        let mut topk_sampler = TopKSampler::new();
        let mut warned_top_p_without_top_k = false;

        let use_repetition_penalty = repetition_penalty != 1.0 && repeat_last_n > 0;
        let mut recent_tokens = if use_repetition_penalty {
            VecDeque::with_capacity(repeat_last_n)
        } else {
            VecDeque::new()
        };
        let mut unique_recent_tokens = if use_repetition_penalty {
            HashSet::with_capacity(repeat_last_n)
        } else {
            HashSet::new()
        };
        let mut pending_newline = false;
        let mut output = String::new();

        let gemma3_end_turn = if self.config.is_gemma3 {
            self.tokenizer
                .find_special_token("<end_of_turn>")
                .unwrap_or(GEMMA3_END_TURN)
        } else {
            -1
        };
        let qwen_im_end = if self.config.is_qwen2
            || self.config.is_qwen3moe
            || self.config.is_qwen3next
            || self.config.is_deepseek2
        {
            self.tokenizer
                .find_special_token("<|im_end|>")
                .unwrap_or(-1)
        } else {
            -1
        };

        while pos < max_tokens {
            if token < 0 || token as usize >= self.config.vocab_size {
                return Err(format!("token id out of bounds: {token}"));
            }

            let prof_t0 = prof_start();
            crate::engine::runtime::transformer(
                token as usize,
                pos,
                &self.config,
                &mut state,
                &self.weights,
                self.gguf.mapped.as_slice(),
            )?;
            prof_end(&PROF_TRANSFORMER_NS, prof_t0);
            if profiling_mode {
                record_forward_pass();
            }

            if debug_mode {
                if let Some(loader) = &self.lazy_debug_loader {
                    let now = time_in_ms();
                    if now >= self.next_lazy_debug_ms {
                        eprintln!("{}", loader.debug_stats_line());
                        self.next_lazy_debug_ms = now + 2_000;
                    }
                }
            }

            if debug_mode
                && pos >= prompt_tokens.len().saturating_sub(1)
                && pos < prompt_tokens.len() + 3
            {
                let mut top: Vec<(usize, f32)> = state.logits[..self.config.vocab_size]
                    .iter()
                    .copied()
                    .enumerate()
                    .collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                eprint!("[DEBUG pos={pos}] Top 5 logits: ");
                for (id, v) in top.into_iter().take(5) {
                    let decoded = self
                        .tokenizer
                        .decode_token(id as i32)
                        .unwrap_or_else(|| "?".to_string())
                        .replace('\n', "\\n")
                        .replace('\r', "\\r");
                    eprint!("{id}({v:.2},\"{decoded}\") ");
                }
                eprintln!();
            }

            if pos < prompt_tokens.len().saturating_sub(1) {
                next = prompt_tokens[pos + 1];
            } else {
                if use_repetition_penalty {
                    unique_recent_tokens.clear();
                    for &tok in &recent_tokens {
                        unique_recent_tokens.insert(tok);
                    }
                    for tok in unique_recent_tokens.iter().copied() {
                        if tok >= 0 && (tok as usize) < self.config.vocab_size {
                            let idx = tok as usize;
                            if state.logits[idx] > 0.0 {
                                state.logits[idx] /= repetition_penalty;
                            } else {
                                state.logits[idx] *= repetition_penalty;
                            }
                        }
                    }
                }

                if temperature == 0.0 {
                    next = argmax(&state.logits[..self.config.vocab_size]) as i32;
                } else if top_k > 0 {
                    next = topk_sampler.sample_top_k_top_p(
                        &state.logits[..self.config.vocab_size],
                        temperature,
                        top_k,
                        top_p,
                        &mut rng,
                    ) as i32;
                } else {
                    if top_p < 1.0 && debug_mode && !warned_top_p_without_top_k {
                        eprintln!("Note: -top_p is ignored unless -top_k > 0");
                        warned_top_p_without_top_k = true;
                    }
                    for q in 0..self.config.vocab_size {
                        state.logits[q] /= temperature;
                    }
                    softmax(
                        &mut state.logits[..self.config.vocab_size],
                        self.config.vocab_size,
                    );
                    next = sample(&state.logits[..self.config.vocab_size], &mut rng) as i32;
                }

                if use_repetition_penalty {
                    if recent_tokens.len() == repeat_last_n {
                        recent_tokens.pop_front();
                    }
                    recent_tokens.push_back(next);
                }
            }

            if pos >= prompt_tokens.len().saturating_sub(1)
                && next != self.tokenizer.eot_token
                && next != self.tokenizer.eos_token
            {
                if let Some(decoded) = self.tokenizer.decode_token(next) {
                    if decoded == "\n" {
                        pending_newline = true;
                    } else {
                        if pending_newline {
                            output.push('\n');
                            if stream_stdout {
                                println!();
                            }
                            pending_newline = false;
                        }
                        output.push_str(&decoded);
                        if stream_stdout {
                            print!("{decoded}");
                            let _ = io::stdout().flush();
                        }
                    }
                }
            }

            token = next;
            pos += 1;

            if start == 0 {
                start = time_in_ms();
            }

            if pos >= prompt_tokens.len().saturating_sub(1) {
                if token == self.tokenizer.eos_token || token == self.tokenizer.eot_token {
                    break;
                }
                if self.config.is_gemma3 && token == gemma3_end_turn {
                    break;
                }
                if (self.config.is_qwen2
                    || self.config.is_qwen3moe
                    || self.config.is_qwen3next
                    || self.config.is_deepseek2)
                    && qwen_im_end >= 0
                    && token == qwen_im_end
                {
                    break;
                }
            }
        }

        let end = time_in_ms();
        if (debug_mode || show_tokens) && pos > 1 {
            let elapsed_ms = (end - start).max(1) as f64;
            eprintln!(
                "\nachieved tok/s: {:.3}",
                (pos - 1) as f64 / elapsed_ms * 1000.0
            );
        } else if stream_stdout {
            println!();
        }

        Ok(output)
    }
}
