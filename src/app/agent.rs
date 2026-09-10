// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.

use crate::app::events::{
    RunnerStatus, RuntimeEvent, RuntimeEventCallback, RuntimeLog, emit_runtime_event,
};
use crate::app::generation::ModelRuntime;
use crate::cli::{CliOptions, ShellCommandDescriptionSpec, ToolPromptSpec};
use crate::tools::{SEARCH_KNOWLEDGE, ToolExecutor};
use crate::vendors::{ChatMessage, ChatRole};
use serde::Deserialize;
use serde_json::{Value, json};

struct AgentMessage {
    role: &'static str,
    content: String,
}

struct AgentPlanner<'a> {
    runtime: &'a mut ModelRuntime,
    system_prompt: &'a str,
    callback: Option<&'a RuntimeEventCallback>,
    debug: bool,
}

struct ToolRunner<'a> {
    tool_exec: &'a ToolExecutor,
    callback: Option<&'a RuntimeEventCallback>,
}

struct FinalAnswerGenerator<'a> {
    runtime: &'a mut ModelRuntime,
    system_prompt: &'a str,
    callback: Option<&'a RuntimeEventCallback>,
}

enum PlannerOutcome {
    Response {
        response: AgentResponse,
        raw: String,
    },
    ParseError {
        parse_error: String,
        raw: String,
    },
}

struct ToolExecution {
    assistant_transcript: String,
    tool_transcript: String,
}

struct ToolRunRequest {
    turn: usize,
    max_turns: usize,
    max_tool_calls: usize,
    tool: String,
    args: Option<Value>,
}

pub(crate) enum AgentRunEvent {
    Log(RuntimeLog),
    Output(String),
}

pub(crate) struct AgentRunResult {
    pub(crate) events: Vec<AgentRunEvent>,
}

fn push_agent_event(
    events: &mut Vec<AgentRunEvent>,
    event: AgentRunEvent,
    callback: Option<&RuntimeEventCallback>,
) {
    let runtime_event = match &event {
        AgentRunEvent::Log(log) => RuntimeEvent::Log(log.clone()),
        AgentRunEvent::Output(text) => RuntimeEvent::Output(text.clone()),
    };
    emit_runtime_event(callback, runtime_event);
    events.push(event);
}

fn emit_agent_status(callback: Option<&RuntimeEventCallback>, status: RunnerStatus) {
    emit_runtime_event(callback, RuntimeEvent::Status(status));
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AgentResponse {
    Final { content: Option<String> },
    ToolCall { tool: String, args: Option<Value> },
}

impl<'a> AgentPlanner<'a> {
    fn plan_turn(
        &mut self,
        transcript: &[AgentMessage],
        turn: usize,
        max_turns: usize,
        events: &mut Vec<AgentRunEvent>,
    ) -> Result<PlannerOutcome, String> {
        emit_agent_status(self.callback, RunnerStatus::Planning { turn, max_turns });
        if self.debug {
            push_agent_event(
                events,
                AgentRunEvent::Log(RuntimeLog::debug(format!("Agent turn {turn}/{max_turns}"))),
                self.callback,
            );
        }
        let turn_prompt = build_turn_prompt(transcript);
        let original_callback = self.runtime.runtime_event_callback();
        let filtered_callback = self.callback.map(|outer| {
            let outer = outer.clone();
            std::sync::Arc::new(move |event: RuntimeEvent| {
                if !matches!(event, RuntimeEvent::Output(_)) {
                    outer(event);
                }
            }) as RuntimeEventCallback
        });
        self.runtime.set_runtime_event_callback(filtered_callback);
        let raw = self
            .runtime
            .generate_text_for_agent(&turn_prompt, self.system_prompt, false);
        self.runtime.set_runtime_event_callback(original_callback);
        let raw = raw?;
        if self.debug {
            push_agent_event(
                events,
                AgentRunEvent::Log(RuntimeLog::debug(format!(
                    "Agent raw output bytes: {}",
                    raw.len()
                ))),
                self.callback,
            );
        }
        Ok(match parse_agent_response(&raw) {
            Ok(response) => PlannerOutcome::Response { response, raw },
            Err(parse_error) => PlannerOutcome::ParseError { parse_error, raw },
        })
    }
}

impl<'a> ToolRunner<'a> {
    fn execute(
        &self,
        tool_calls: &mut usize,
        request: ToolRunRequest,
        events: &mut Vec<AgentRunEvent>,
    ) -> Result<ToolExecution, String> {
        emit_agent_status(
            self.callback,
            RunnerStatus::Tool {
                turn: request.turn,
                max_turns: request.max_turns,
                tool: request.tool.clone(),
            },
        );
        if *tool_calls >= request.max_tool_calls {
            return Err(format!(
                "max-tool-calls ({}) reached before final response",
                request.max_tool_calls
            ));
        }
        *tool_calls += 1;
        let tool = request.tool;
        let args = request.args.unwrap_or_else(|| json!({}));
        let args_json =
            serde_json::to_string(&args).unwrap_or_else(|_| "<invalid-json>".to_string());
        push_agent_event(
            events,
            AgentRunEvent::Log(RuntimeLog::system(format!(
                "Tool call [{}]: {} args={}",
                *tool_calls, tool, args_json
            ))),
            self.callback,
        );
        let tool_result = match self.tool_exec.execute(&tool, &args) {
            Ok(v) => v,
            Err(e) => {
                if tool == "shell_exec" {
                    push_agent_event(
                        events,
                        AgentRunEvent::Log(RuntimeLog::error(format!("shell_exec error: {e}"))),
                        self.callback,
                    );
                }
                json!({
                    "ok": false,
                    "tool": tool,
                    "error": e
                })
            }
        };
        if tool == "shell_exec"
            && tool_result.get("ok").and_then(Value::as_bool) == Some(false)
            && tool_result.get("exit_code").is_some()
        {
            let exit_code = tool_result
                .get("exit_code")
                .and_then(Value::as_i64)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let stderr = tool_result
                .get("stderr")
                .and_then(Value::as_str)
                .unwrap_or("");
            let stderr_preview = stderr.trim().chars().take(240).collect::<String>();
            if stderr_preview.is_empty() {
                push_agent_event(
                    events,
                    AgentRunEvent::Log(RuntimeLog::error(format!(
                        "shell_exec failed (exit_code={exit_code})"
                    ))),
                    self.callback,
                );
            } else {
                push_agent_event(
                    events,
                    AgentRunEvent::Log(RuntimeLog::error(format!(
                        "shell_exec failed (exit_code={exit_code}): {}",
                        stderr_preview
                    ))),
                    self.callback,
                );
            }
        }
        let tool_result_text =
            serde_json::to_string_pretty(&tool_result).map_err(|e| e.to_string())?;
        Ok(ToolExecution {
            assistant_transcript: format!(
                "tool_call\n{}",
                json!({
                    "tool": tool,
                    "args": args
                })
            ),
            tool_transcript: tool_result_text,
        })
    }
}

impl<'a> FinalAnswerGenerator<'a> {
    fn generate(
        &mut self,
        prior_chat_history: &[ChatMessage],
        transcript: &[AgentMessage],
        prompt: &str,
        inline_content: Option<&str>,
    ) -> Result<String, String> {
        if transcript.iter().all(|message| message.role != "tool") {
            return fallback_to_plain_chat(
                self.runtime,
                prior_chat_history,
                prompt,
                self.system_prompt,
                self.callback,
            );
        }

        let synthesis_prompt = build_tool_answer_prompt(prompt, transcript, inline_content);
        let original_callback = self.runtime.runtime_event_callback();
        self.runtime
            .set_runtime_event_callback(self.callback.cloned());
        let output =
            self.runtime
                .generate_text_without_think(&synthesis_prompt, self.system_prompt, false);
        self.runtime.set_runtime_event_callback(original_callback);
        output
    }
}

pub(crate) fn run_agent_loop(
    runtime: &mut ModelRuntime,
    cli: &CliOptions,
    prompt: &str,
) -> Result<(), String> {
    let result = run_agent_loop_collect(runtime, cli, prompt)?;
    for event in result.events {
        match event {
            AgentRunEvent::Log(log) => eprintln!("{}", log.message),
            AgentRunEvent::Output(text) => println!("{text}"),
        }
    }
    Ok(())
}

pub(crate) fn run_agent_loop_collect(
    runtime: &mut ModelRuntime,
    cli: &CliOptions,
    prompt: &str,
) -> Result<AgentRunResult, String> {
    run_agent_loop_collect_with_history_callback(runtime, cli, &[], prompt, None)
}

pub(crate) fn run_agent_loop_collect_with_history_callback(
    runtime: &mut ModelRuntime,
    cli: &CliOptions,
    prior_chat_history: &[ChatMessage],
    prompt: &str,
    callback: Option<&RuntimeEventCallback>,
) -> Result<AgentRunResult, String> {
    let tool_exec = ToolExecutor::new(
        cli.tool_root.as_deref(),
        cli.tool_enablement.clone(),
        &cli.allow_shell_commands,
    )?;
    let decode_policy = runtime.vendor_decode_policy();
    let has_rag = runtime.has_rag_index();
    let system_prompt = build_agent_system_prompt(
        &cli.system_prompt,
        &tool_exec,
        &cli.tool_prompt_specs,
        &cli.shell_command_description_specs,
        has_rag,
    );
    let mut transcript = Vec::new();
    for message in prior_chat_history {
        transcript.push(AgentMessage {
            role: match message.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            },
            content: message.content.clone(),
        });
    }
    transcript.push(AgentMessage {
        role: "user",
        content: prompt.to_string(),
    });
    let mut tool_calls = 0usize;
    let mut protocol_failures = 0usize;
    let mut events = Vec::new();
    let max_protocol_failures = decode_policy.agent_protocol_max_failures.max(1);
    let max_turns = cli
        .max_tool_calls
        .saturating_mul(3)
        .saturating_add(8)
        .min(64);
    let require_tool_before_final =
        prompt_requires_filesystem(prompt) && tool_exec.has_any_filesystem_tool();
    let mut planner = AgentPlanner {
        runtime,
        system_prompt: &system_prompt,
        callback,
        debug: cli.debug,
    };
    let tool_runner = ToolRunner {
        tool_exec: &tool_exec,
        callback,
    };

    for turn in 0..max_turns {
        match planner.plan_turn(&transcript, turn + 1, max_turns, &mut events)? {
            PlannerOutcome::Response {
                response: AgentResponse::Final { content },
                raw: _raw,
            } => {
                emit_agent_status(
                    callback,
                    RunnerStatus::Finalizing {
                        turn: turn + 1,
                        max_turns,
                    },
                );
                if require_tool_before_final && tool_calls == 0 {
                    protocol_failures += 1;
                    if protocol_failures > max_protocol_failures {
                        return Err(
                            "model returned final response without using any filesystem tool"
                                .to_string(),
                        );
                    }
                    transcript.push(AgentMessage {
                        role: "user",
                        content: "You must call at least one filesystem tool before finalizing this request. Reply with exactly one JSON object: either a tool_call or final."
                            .to_string(),
                    });
                    continue;
                }
                let mut final_answer_generator = FinalAnswerGenerator {
                    runtime: planner.runtime,
                    system_prompt: &cli.system_prompt,
                    callback,
                };
                let content = final_answer_generator.generate(
                    prior_chat_history,
                    &transcript,
                    prompt,
                    content.as_deref(),
                )?;
                push_agent_event(&mut events, AgentRunEvent::Output(content), callback);
                return Ok(AgentRunResult { events });
            }
            PlannerOutcome::Response {
                response: AgentResponse::ToolCall { tool, args },
                raw: _raw,
            } => {
                // search_knowledge is handled directly — not via ToolExecutor.
                if tool == SEARCH_KNOWLEDGE {
                    if tool_calls >= cli.max_tool_calls {
                        return Err(format!(
                            "max-tool-calls ({}) reached before final response",
                            cli.max_tool_calls
                        ));
                    }
                    tool_calls += 1;
                    let query = args
                        .as_ref()
                        .and_then(|a| a.get("query").and_then(|v| v.as_str()))
                        .unwrap_or("")
                        .to_string();
                    let top_k = args
                        .as_ref()
                        .and_then(|a| a.get("top_k").and_then(|v| v.as_u64()))
                        .map(|v| v as usize)
                        .unwrap_or(planner.runtime.settings().rag_top_k);
                    push_agent_event(
                        &mut events,
                        AgentRunEvent::Log(RuntimeLog::system(format!(
                            "Tool call [{}]: search_knowledge query={:?} top_k={top_k}",
                            tool_calls, query
                        ))),
                        callback,
                    );
                    let result_text = match planner.runtime.search_rag(&query, top_k) {
                        Ok(text) => text,
                        Err(e) => format!("search_knowledge error: {e}"),
                    };
                    let tool_result = serde_json::json!({
                        "ok": true,
                        "tool": SEARCH_KNOWLEDGE,
                        "query": query,
                        "results": result_text
                    });
                    let tool_result_text =
                        serde_json::to_string_pretty(&tool_result).map_err(|e| e.to_string())?;
                    transcript.push(AgentMessage {
                        role: "assistant",
                        content: format!(
                            "tool_call\n{}",
                            serde_json::json!({"tool": SEARCH_KNOWLEDGE, "args": args.unwrap_or_else(|| serde_json::json!({}))})
                        ),
                    });
                    transcript.push(AgentMessage {
                        role: "tool",
                        content: tool_result_text,
                    });
                } else {
                    let tool_execution = tool_runner.execute(
                        &mut tool_calls,
                        ToolRunRequest {
                            turn: turn + 1,
                            max_turns,
                            max_tool_calls: cli.max_tool_calls,
                            tool,
                            args,
                        },
                        &mut events,
                    )?;
                    transcript.push(AgentMessage {
                        role: "assistant",
                        content: tool_execution.assistant_transcript,
                    });
                    transcript.push(AgentMessage {
                        role: "tool",
                        content: tool_execution.tool_transcript,
                    });
                }
            }
            PlannerOutcome::ParseError { parse_error, raw } => {
                if cli.debug {
                    let preview = raw.trim().chars().take(180).collect::<String>();
                    push_agent_event(
                        &mut events,
                        AgentRunEvent::Log(RuntimeLog::error(format!(
                            "Agent protocol parse error: {} | preview: {}",
                            parse_error, preview
                        ))),
                        callback,
                    );
                }
                let fallback = raw.trim();
                let fallback_looks_like_json = fallback.starts_with('{');
                if tool_calls > 0
                    && !fallback.is_empty()
                    && !fallback_looks_like_json
                    && looks_like_reasonable_fallback_text(fallback)
                {
                    if cli.debug {
                        push_agent_event(
                            &mut events,
                            AgentRunEvent::Log(RuntimeLog::debug(format!(
                                "Agent protocol fallback after tool call(s): {}",
                                parse_error
                            ))),
                            callback,
                        );
                    }
                    push_agent_event(
                        &mut events,
                        AgentRunEvent::Output(fallback.to_string()),
                        callback,
                    );
                    return Ok(AgentRunResult { events });
                }
                protocol_failures += 1;
                if protocol_failures > max_protocol_failures {
                    if !require_tool_before_final {
                        if let Some(content) = extract_agent_final_content(&raw) {
                            push_agent_event(&mut events, AgentRunEvent::Output(content), callback);
                            return Ok(AgentRunResult { events });
                        }
                        if tool_calls == 0
                            && decode_policy.agent_plain_chat_fallback_after_protocol_failures
                        {
                            if cli.debug {
                                push_agent_event(
                                    &mut events,
                                    AgentRunEvent::Log(RuntimeLog::debug(
                                        "Agent protocol exhausted; falling back to plain chat",
                                    )),
                                    callback,
                                );
                            }
                            emit_agent_status(
                                callback,
                                RunnerStatus::Recovering {
                                    turn: turn + 1,
                                    max_turns,
                                },
                            );
                            let content = fallback_to_plain_chat(
                                runtime,
                                prior_chat_history,
                                prompt,
                                &cli.system_prompt,
                                callback,
                            )?;
                            push_agent_event(&mut events, AgentRunEvent::Output(content), callback);
                            return Ok(AgentRunResult { events });
                        }
                        let fallback = raw.trim();
                        if !fallback.is_empty() {
                            push_agent_event(
                                &mut events,
                                AgentRunEvent::Output(fallback.to_string()),
                                callback,
                            );
                            return Ok(AgentRunResult { events });
                        }
                    }
                    let raw_preview = raw.trim().chars().take(240).collect::<String>();
                    return Err(format!(
                        "model did not follow agent JSON protocol after {} attempts: {}. Last output: {}",
                        max_protocol_failures, parse_error, raw_preview
                    ));
                }
                let mut protocol_msg = "Protocol error: reply with exactly one JSON object only. Use either {\"type\":\"tool_call\",\"tool\":\"...\",\"args\":{...}} or {\"type\":\"final\"}."
                    .to_string();
                if require_tool_before_final && tool_calls == 0 {
                    protocol_msg.push_str(
                        " For this request you must call a filesystem tool before final.",
                    );
                }
                transcript.push(AgentMessage {
                    role: "user",
                    content: protocol_msg,
                });
            }
        }
    }

    Err("agent loop reached maximum turns without final response".to_string())
}

fn fallback_to_plain_chat(
    runtime: &mut ModelRuntime,
    prior_chat_history: &[ChatMessage],
    prompt: &str,
    system_prompt: &str,
    callback: Option<&RuntimeEventCallback>,
) -> Result<String, String> {
    let mut messages = prior_chat_history.to_vec();
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: prompt.to_string(),
    });
    let original_callback = runtime.runtime_event_callback();
    runtime.set_runtime_event_callback(callback.cloned());
    let output = runtime.generate_chat_messages_without_think_for_repl(&messages, system_prompt);
    runtime.set_runtime_event_callback(original_callback);
    output
}

fn build_agent_system_prompt(
    base_system_prompt: &str,
    tool_exec: &ToolExecutor,
    tool_prompt_specs: &[ToolPromptSpec],
    shell_command_description_specs: &[ShellCommandDescriptionSpec],
    has_rag: bool,
) -> String {
    let _metadata_bytes: usize = tool_prompt_specs
        .iter()
        .map(|s| s.name.len() + s.description.len() + s.when_to_use.len())
        .sum::<usize>()
        + shell_command_description_specs
            .iter()
            .map(|s| s.command.len() + s.description.len())
            .sum::<usize>();
    let write_state = if tool_exec.write_file_enabled() {
        "enabled"
    } else {
        "disabled"
    };
    let mut enabled_tools = tool_exec.enabled_tool_names();
    if has_rag {
        enabled_tools.push(SEARCH_KNOWLEDGE);
    }
    let allowed_tool_names = if enabled_tools.is_empty() {
        "<none>".to_string()
    } else {
        enabled_tools.join("|")
    };
    let shell_allowed_commands = if tool_exec.shell_allowed_commands().is_empty() {
        "<none>".to_string()
    } else {
        tool_exec.shell_allowed_commands().join(", ")
    };
    let tool_catalog = render_tool_catalog(
        tool_exec,
        tool_prompt_specs,
        shell_command_description_specs,
        has_rag,
    );
    let tool_rules = render_tool_rules(tool_exec, has_rag);
    format!(
        "{base_system_prompt}\n\n\
You are running with host tools.\n\
Always respond with exactly one JSON object and no surrounding markdown.\n\
Use compact JSON on a single line, with keys in the exact order shown below.\n\
Allowed response schemas:\n\
1) Tool call:\n\
{{\"type\":\"tool_call\",\"tool\":\"{}\",\"args\":{{...}}}}\n\
2) Final decision:\n\
{{\"type\":\"final\"}}\n\
Available tools:\n\
{}\n\
Rules:\n\
{}\n\
Runtime constraints:\n\
- tool_root: {}\n\
- write_file: {}\n\
- shell allowed commands: {}\n\
- max read/write payload per call: 262144 bytes\n\
Output rules:\n\
- If the user asks about files/repo contents and filesystem tools are enabled, call a filesystem tool before final.\n\
- Use `type=final` when you are done deciding. Do not put long natural-language answers inside the JSON object.\n\
- If a tool call fails due args shape, fix the args and retry.\n\
- Avoid prose outside the single JSON object.\n",
        allowed_tool_names,
        tool_catalog,
        tool_rules,
        tool_exec.root().display(),
        write_state,
        shell_allowed_commands
    )
}

fn looks_like_reasonable_fallback_text(text: &str) -> bool {
    if text.len() < 24 || text.contains('\0') {
        return false;
    }
    let mut total = 0usize;
    let mut humanish = 0usize;
    let mut alpha = 0usize;
    for c in text.chars() {
        total += 1;
        if c.is_ascii_alphabetic() {
            alpha += 1;
            humanish += 1;
            continue;
        }
        if c.is_ascii_whitespace() || c.is_ascii_punctuation() || c.is_ascii_digit() {
            humanish += 1;
        }
    }
    if total == 0 {
        return false;
    }
    let humanish_ratio = humanish as f32 / total as f32;
    let alpha_ratio = alpha as f32 / total as f32;
    humanish_ratio > 0.92 && alpha_ratio > 0.08
}

fn render_tool_rules(tool_exec: &ToolExecutor, has_rag: bool) -> String {
    let mut rules = vec![
        "- Use tools when you need filesystem state.".to_string(),
        "- If the user asks about files, code, directories, or repository contents, call a filesystem tool before answering when such tools are enabled.".to_string(),
        "- If the user asks you to create or modify files, call `write_file` with the full replacement content instead of only describing the change.".to_string(),
        "- Keep tool arguments minimal and valid JSON.".to_string(),
        "- If a tool fails, adjust and retry.".to_string(),
        "- Return `type=final` when done.".to_string(),
    ];
    if has_rag {
        rules.push(
            "- `search_knowledge` queries the loaded knowledge base. Args: {\"query\":\"<search terms>\",\"top_k\":5}. Call it when the user asks about domain-specific topics not in your training data.".to_string(),
        );
    }
    if tool_exec.shell_exec_enabled() {
        rules.push(
            "- `shell_exec` can only execute commands already in the shell allowed list."
                .to_string(),
        );
        rules.push(
            "- `shell_exec` args must include a command key (or alias cmd). Example: {\"type\":\"tool_call\",\"tool\":\"shell_exec\",\"args\":{\"command\":\"ls\",\"args\":[\"-la\"]}}."
                .to_string(),
        );
    }
    if tool_exec.shell_list_allowed_enabled() {
        rules.push(
            "- Use `shell_list_allowed` when you need to inspect shell allowed commands or internal tool status.".to_string(),
        );
    }
    if tool_exec.shell_request_allowed_enabled() {
        rules.push("- If a needed command is missing from the shell allowed list, call `shell_request_allowed` with command + reason.".to_string());
    }
    rules.join("\n")
}

fn render_tool_catalog(
    tool_exec: &ToolExecutor,
    tool_prompt_specs: &[ToolPromptSpec],
    shell_command_description_specs: &[ShellCommandDescriptionSpec],
    has_rag: bool,
) -> String {
    let enabled_tools = tool_exec.enabled_tool_names();
    let mut lines = Vec::new();
    for spec in tool_prompt_specs {
        if !enabled_tools.contains(&spec.name.as_str()) {
            continue;
        }
        let mut line = format!(
            "- {}: {} When to use: {}",
            spec.name, spec.description, spec.when_to_use
        );
        if let Some(args_hint) = tool_args_hint(&spec.name) {
            line.push_str(" Args: ");
            line.push_str(args_hint);
        }
        lines.push(line);
    }
    if has_rag {
        lines.push("- search_knowledge: Search the loaded knowledge base. When to use: Query the attached RAG corpus for facts not in the current conversation. Args: {\"query\":\"terms\",\"top_k\":5}".to_string());
    }
    if tool_exec.shell_exec_enabled() && !shell_command_description_specs.is_empty() {
        let command_lines = shell_command_description_specs
            .iter()
            .map(|spec| format!("{}={}", spec.command, spec.description))
            .collect::<Vec<_>>()
            .join("; ");
        lines.push(format!("- shell command descriptions: {}", command_lines));
    }
    if lines.is_empty() {
        "- <none>".to_string()
    } else {
        lines.join("\n")
    }
}

fn tool_args_hint(tool_name: &str) -> Option<&'static str> {
    match tool_name {
        "read_file" => Some("{\"path\":\"relative/or/absolute/path\",\"max_bytes\":262144}"),
        "list_dir" => Some("{\"path\":\"dir-or-.\",\"max_entries\":200}"),
        "write_file" => Some(
            "{\"path\":\"relative/or/absolute/path\",\"content\":\"full utf8 file text\",\"append\":false}",
        ),
        "mkdir" => Some("{\"path\":\"dir/path\"}"),
        "rmdir" => Some("{\"path\":\"dir/path\"}"),
        "shell_list_allowed" => Some("{}"),
        "shell_exec" => Some(
            "{\"command\":\"<allowed>\",\"args\":[...],\"cwd\":\".\",\"max_output_bytes\":131072}",
        ),
        "shell_request_allowed" => Some("{\"command\":\"<needed>\",\"reason\":\"why needed\"}"),
        _ => None,
    }
}

fn build_turn_prompt(transcript: &[AgentMessage]) -> String {
    let mut out = String::from("Transcript:\n");
    for msg in transcript {
        out.push_str("<<<");
        out.push_str(msg.role);
        out.push_str(">>>\n");
        out.push_str(&msg.content);
        out.push('\n');
    }
    out.push_str("Respond with one JSON object now.");
    out
}

fn build_tool_answer_prompt(
    prompt: &str,
    transcript: &[AgentMessage],
    inline_content: Option<&str>,
) -> String {
    let mut out = String::from(
        "Use the tool transcript below to answer the user's request. Respond with plain text only.\n\n",
    );
    out.push_str("User request:\n");
    out.push_str(prompt);
    out.push_str("\n\nTool transcript:\n");
    for msg in transcript {
        if msg.role == "tool" || msg.role == "assistant" {
            out.push_str("<<<");
            out.push_str(msg.role);
            out.push_str(">>>\n");
            out.push_str(&msg.content);
            out.push('\n');
        }
    }
    if let Some(content) = inline_content {
        let trimmed = content.trim();
        if !trimmed.is_empty() {
            out.push_str("\nDraft final answer:\n");
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    out.push_str("\nAnswer the user now.");
    out
}

fn parse_agent_response(raw: &str) -> Result<AgentResponse, String> {
    if let Ok(parsed) = serde_json::from_str::<AgentResponse>(raw.trim()) {
        return Ok(parsed);
    }

    let mut saw_json_object = false;
    for value in extract_json_objects(raw) {
        saw_json_object = true;
        if let Ok(parsed) = serde_json::from_value::<AgentResponse>(value.clone()) {
            return Ok(parsed);
        }
        if let Ok(parsed) = parse_agent_response_from_value(value) {
            return Ok(parsed);
        }
    }
    if saw_json_object {
        Err("unsupported agent response payload".to_string())
    } else {
        Err("no JSON object found in model output".to_string())
    }
}

fn extract_agent_final_content(raw: &str) -> Option<String> {
    if let Ok(AgentResponse::Final {
        content: Some(content),
    }) = serde_json::from_str::<AgentResponse>(raw.trim())
    {
        return Some(content);
    }
    for value in extract_json_objects(raw) {
        if let Some(content) = value.get("content").and_then(Value::as_str)
            && (value.get("type").and_then(Value::as_str) == Some("final")
                || value.get("type").is_none())
        {
            return Some(content.to_string());
        }
    }
    None
}

fn parse_agent_response_from_value(value: Value) -> Result<AgentResponse, String> {
    if value.get("type").and_then(Value::as_str) == Some("final") {
        return Ok(AgentResponse::Final {
            content: value
                .get("content")
                .and_then(Value::as_str)
                .map(|content| content.to_string()),
        });
    }
    if value.get("type").is_none() && value.get("content").and_then(Value::as_str).is_some() {
        return Ok(AgentResponse::Final {
            content: value
                .get("content")
                .and_then(Value::as_str)
                .map(|content| content.to_string()),
        });
    }
    if let Some(tool) = value.get("tool").and_then(Value::as_str) {
        let args = value.get("args").cloned();
        if value.get("type").and_then(Value::as_str) == Some("tool_call")
            || value.get("type").is_none()
        {
            return Ok(AgentResponse::ToolCall {
                tool: tool.to_string(),
                args,
            });
        }
    }
    Err("unsupported agent response payload".to_string())
}

fn extract_json_objects(raw: &str) -> Vec<Value> {
    let mut values = Vec::new();
    for (idx, ch) in raw.char_indices() {
        if ch != '{' {
            continue;
        }
        if let Some(v) = parse_first_json_value(&raw[idx..])
            && v.is_object()
        {
            values.push(v);
        }
    }
    values
}

fn parse_first_json_value(s: &str) -> Option<Value> {
    let mut de = serde_json::Deserializer::from_str(s);
    Value::deserialize(&mut de).ok()
}

fn prompt_requires_filesystem(prompt: &str) -> bool {
    let p = prompt.to_ascii_lowercase();
    let hints = [
        "file",
        "directory",
        "folder",
        "project",
        "workspace",
        "src/",
        ".rs",
        ".toml",
        "inspect",
        "read",
        "list",
        "repo",
        "repository",
        "codebase",
    ];
    hints.iter().any(|h| p.contains(h))
}

fn prompt_requests_filesystem_edit(prompt: &str) -> bool {
    let p = prompt.to_ascii_lowercase();
    let hints = [
        "edit ",
        "edit the",
        "modify",
        "change",
        "update",
        "replace",
        "rename",
        "rewrite",
        "patch",
        "fix",
        "create file",
        "write file",
        "save",
        "append",
    ];
    hints.iter().any(|h| p.contains(h))
}

pub(crate) fn prompt_likely_requires_tools(
    prompt: &str,
    filesystem_tools_enabled: bool,
    shell_tools_enabled: bool,
) -> bool {
    if filesystem_tools_enabled && prompt_requires_filesystem(prompt) {
        return true;
    }
    if shell_tools_enabled {
        let p = prompt.to_ascii_lowercase();
        let shell_hints = [
            "shell", "command", "terminal", "bash", "zsh", "sh ", "run ", "execute ", "exec ",
            "ls", "pwd", "find ", "grep ", "rg ", "cat ", "cargo ", "git ", "curl ",
        ];
        if shell_hints.iter().any(|h| p.contains(h)) {
            return true;
        }
    }
    false
}

pub(crate) fn conversation_likely_requires_tools(
    prompt: &str,
    prior_chat_history: &[ChatMessage],
    filesystem_tools_enabled: bool,
    shell_tools_enabled: bool,
) -> bool {
    if prompt_likely_requires_tools(prompt, filesystem_tools_enabled, shell_tools_enabled) {
        return true;
    }
    if !filesystem_tools_enabled || !prompt_requests_filesystem_edit(prompt) {
        return false;
    }
    prior_chat_history
        .iter()
        .rev()
        .take(8)
        .any(|message| prompt_requires_filesystem(&message.content))
}

#[cfg(test)]
mod tests {
    use super::{
        AgentResponse, build_agent_system_prompt, conversation_likely_requires_tools,
        extract_agent_final_content, looks_like_reasonable_fallback_text, parse_agent_response,
        prompt_likely_requires_tools,
    };
    use crate::cli::{AgentToolEnablement, ToolPromptSpec};
    use crate::tools::ToolExecutor;
    use crate::vendors::{ChatMessage, ChatRole};

    #[test]
    fn fallback_text_guard_rejects_numeric_gibberish() {
        let gibberish = "000003 010 1000 200000101 01 600006 00000 000106 40 00016560";
        assert!(!looks_like_reasonable_fallback_text(gibberish));
    }

    #[test]
    fn extract_agent_final_content_from_json_prefix_with_trailing_text() {
        let raw = "{\"type\":\"final\",\"content\":\"Paris is a good weekend destination.\"}Paris is a good weekend destination.";
        assert_eq!(
            extract_agent_final_content(raw).as_deref(),
            Some("Paris is a good weekend destination.")
        );
    }

    #[test]
    fn parse_agent_response_accepts_final_without_content() {
        let parsed = parse_agent_response("{\"type\":\"final\"}").expect("final response");
        match parsed {
            AgentResponse::Final { content } => assert!(content.is_none()),
            AgentResponse::ToolCall { .. } => panic!("expected final response"),
        }
    }

    #[test]
    fn prompt_tool_routing_stays_off_for_plain_chat() {
        assert!(!prompt_likely_requires_tools(
            "what is the capital of france?",
            true,
            true
        ));
    }

    #[test]
    fn prompt_tool_routing_detects_repo_requests() {
        assert!(prompt_likely_requires_tools(
            "can you inspect my Cargo.toml and list the dependencies?",
            true,
            false
        ));
    }

    #[test]
    fn prompt_tool_routing_keeps_edit_followups_in_agent_mode() {
        let history = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "Inspect this repository and summarize the config files.".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "I found Cargo.toml and docs/agent-config.example.toml.".to_string(),
            },
        ];
        assert!(conversation_likely_requires_tools(
            "Now update the config values to match that behavior.",
            &history,
            true,
            false
        ));
    }

    #[test]
    fn agent_system_prompt_lists_write_file_args() {
        let tool_exec =
            ToolExecutor::new(None, AgentToolEnablement::default(), &[]).expect("tool executor");
        let prompt = build_agent_system_prompt(
            "base system",
            &tool_exec,
            &[ToolPromptSpec {
                name: "write_file".to_string(),
                description: "Write UTF-8 file content.".to_string(),
                when_to_use: "Use for explicit file edits.".to_string(),
            }],
            &[],
            false,
        );
        assert!(prompt.contains("write_file"));
        assert!(prompt.contains("{\"path\":\"relative/or/absolute/path\",\"content\":\"full utf8 file text\",\"append\":false}"));
        assert!(prompt.contains("call `write_file`"));
    }
}
