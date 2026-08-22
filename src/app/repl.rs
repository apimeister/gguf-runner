// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use crate::app::agent;
use crate::app::events::{
    RunnerStatus, RuntimeEvent, RuntimeEventCallback, RuntimeLog, RuntimeLogKind, RuntimePhase,
    RuntimeProgress, emit_runtime_event,
};
use crate::app::generation::ModelRuntime;
use crate::app::{collect_debug_banner_lines, expand_repl_tab_completion, handle_repl_command};
use crate::cli::CliOptions;
use crate::vendors::{ChatMessage, ChatRole};
use crossterm::cursor::{Hide, MoveTo, RestorePosition, SavePosition, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::style::Print;
use crossterm::terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size};
use std::cmp::min;
use std::io::{self, Stdout, Write};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

const FOOTER_HEIGHT: u16 = 2;

struct ReplApp {
    input: String,
    cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    active_images: Vec<String>,
    rag_loaded: bool,
    chat_history: Vec<ChatMessage>,
    pending_user_prompt: Option<String>,
    pending_assistant_output: String,
    assistant_line_open: bool,
    status: ReplStatus,
    status_override_left: Option<String>,
    session_output_tokens: usize,
    last_progress_decode_tokens: usize,
    busy: bool,
    runtime_ready: bool,
}

struct ReplStatus {
    left: String,
    right: String,
}

impl ReplApp {
    fn new() -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: None,
            active_images: Vec::new(),
            rag_loaded: false,
            chat_history: Vec::new(),
            pending_user_prompt: None,
            pending_assistant_output: String::new(),
            assistant_line_open: false,
            status: ReplStatus {
                left: "Loading model...".to_string(),
                right: String::new(),
            },
            status_override_left: None,
            session_output_tokens: 0,
            last_progress_decode_tokens: 0,
            busy: false,
            runtime_ready: false,
        }
    }

    fn set_status(&mut self, text: impl Into<String>) {
        self.status_override_left = None;
        self.status.left = text.into();
        self.status.right.clear();
    }

    fn set_ready_status(&mut self) {
        self.status_override_left = None;
        self.status.left = "Ready".to_string();
    }

    fn set_status_override(&mut self, status: RunnerStatus) {
        self.status_override_left = Some(format_runner_status(&status));
        if let Some(left) = &self.status_override_left {
            self.status.left = left.clone();
        }
    }

    fn clear_status_override(&mut self) {
        self.status_override_left = None;
    }

    fn set_progress(&mut self, progress: RuntimeProgress) {
        if progress.decode_tokens < self.last_progress_decode_tokens {
            self.last_progress_decode_tokens = 0;
        }
        let delta = progress
            .decode_tokens
            .saturating_sub(self.last_progress_decode_tokens);
        self.session_output_tokens = self.session_output_tokens.saturating_add(delta);
        self.last_progress_decode_tokens = progress.decode_tokens;

        let computed_left = if progress.hidden_thinking {
            format!(
                "thinking {} tok | decode {} tok",
                progress.hidden_think_tokens, progress.decode_tokens
            )
        } else {
            match progress.phase {
                RuntimePhase::Ready => "Ready".to_string(),
                RuntimePhase::Prefill | RuntimePhase::Decode => format!(
                    "prefill {} tok | decode {} tok",
                    progress.prefill_tokens, progress.decode_tokens
                ),
            }
        };
        let left = self.status_override_left.clone().unwrap_or(computed_left);
        let right = if progress.context_limit == 0 {
            String::new()
        } else {
            let mut parts = Vec::new();
            parts.push(format!(
                "out {} tok",
                format_compact_token_count(self.session_output_tokens)
            ));
            if let Some(tok_s) = progress.tokens_per_second {
                parts.push(format!("{tok_s:.1} tok/s"));
            }
            parts.push(format!(
                "ctx {}",
                context_gauge(progress.context_used, progress.context_limit, 10)
            ));
            parts.join(" | ")
        };
        self.status = ReplStatus { left, right };
    }

    fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.history_index = None;
    }

    fn finish_turn_progress(&mut self) {
        self.last_progress_decode_tokens = 0;
    }

    fn absorb_assistant_output(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        if !self.pending_assistant_output.is_empty()
            && text.starts_with(&self.pending_assistant_output)
        {
            let suffix = text[self.pending_assistant_output.len()..].to_string();
            self.pending_assistant_output = text.to_string();
            return suffix;
        }
        self.pending_assistant_output.push_str(text);
        text.to_string()
    }

    fn insert_char(&mut self, ch: char) {
        self.input.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self
            .input
            .char_indices()
            .take_while(|(idx, _)| *idx < self.cursor)
            .map(|(idx, _)| idx)
            .last()
            .unwrap_or(0);
        self.input.drain(prev..self.cursor);
        self.cursor = prev;
    }

    fn delete(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let next = self
            .input
            .char_indices()
            .find(|(idx, _)| *idx > self.cursor)
            .map(|(idx, _)| idx)
            .unwrap_or(self.input.len());
        self.input.drain(self.cursor..next);
    }

    fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor = self
            .input
            .char_indices()
            .take_while(|(idx, _)| *idx < self.cursor)
            .map(|(idx, _)| idx)
            .last()
            .unwrap_or(0);
    }

    fn move_right(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        self.cursor = self
            .input
            .char_indices()
            .find(|(idx, _)| *idx > self.cursor)
            .map(|(idx, _)| idx)
            .unwrap_or(self.input.len());
    }

    fn apply_tab_completion(&mut self) {
        let completed = expand_repl_tab_completion(&self.input);
        self.input = completed;
        self.cursor = self.input.len();
    }

    fn push_history(&mut self, entry: &str) {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.history.last().map(|last| last.as_str()) == Some(trimmed) {
            return;
        }
        self.history.push(trimmed.to_string());
        self.history_index = None;
    }

    fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next_index = match self.history_index {
            Some(0) => 0,
            Some(idx) => idx.saturating_sub(1),
            None => self.history.len().saturating_sub(1),
        };
        self.history_index = Some(next_index);
        self.input = self.history[next_index].clone();
        self.cursor = self.input.len();
    }

    fn history_down(&mut self) {
        let Some(idx) = self.history_index else {
            return;
        };
        if idx + 1 >= self.history.len() {
            self.history_index = None;
            self.input.clear();
            self.cursor = 0;
            return;
        }
        let next_index = idx + 1;
        self.history_index = Some(next_index);
        self.input = self.history[next_index].clone();
        self.cursor = self.input.len();
    }
}

struct TerminalGuard {
    stdout: Stdout,
    width: u16,
    height: u16,
}

impl TerminalGuard {
    fn new() -> Result<Self, String> {
        enable_raw_mode().map_err(|e| format!("failed to enable raw mode: {e}"))?;
        let mut stdout = io::stdout();
        execute!(stdout, Hide).map_err(|e| format!("failed to hide cursor: {e}"))?;
        writeln!(stdout).map_err(|e| format!("failed to initialize repl footer: {e}"))?;
        writeln!(stdout).map_err(|e| format!("failed to initialize repl footer: {e}"))?;
        stdout
            .flush()
            .map_err(|e| format!("failed to flush terminal: {e}"))?;
        let (width, height) = size().map_err(|e| format!("failed to get terminal size: {e}"))?;
        let mut guard = Self {
            stdout,
            width,
            height,
        };
        guard.configure_scroll_region()?;
        Ok(guard)
    }

    fn refresh_size(&mut self) -> Result<(), String> {
        let (width, height) = size().map_err(|e| format!("failed to get terminal size: {e}"))?;
        let changed = width != self.width || height != self.height;
        self.width = width;
        self.height = height;
        if changed {
            self.configure_scroll_region()?;
        }
        Ok(())
    }

    fn configure_scroll_region(&mut self) -> Result<(), String> {
        let output_bottom = self.output_bottom_row_1based();
        write!(self.stdout, "\x1b[1;{}r", output_bottom)
            .map_err(|e| format!("failed to set scroll region: {e}"))?;
        self.stdout
            .flush()
            .map_err(|e| format!("failed to flush scroll region: {e}"))
    }

    fn output_bottom_row_1based(&self) -> u16 {
        self.height.saturating_sub(FOOTER_HEIGHT).max(1)
    }

    fn footer_status_row(&self) -> u16 {
        self.height.saturating_sub(FOOTER_HEIGHT)
    }

    fn footer_input_row(&self) -> u16 {
        self.height.saturating_sub(1)
    }

    fn ensure_output_line_closed(&mut self, app: &mut ReplApp) -> Result<(), String> {
        if app.assistant_line_open {
            execute!(self.stdout, Print("\r\n"))
                .map_err(|e| format!("failed to end assistant line: {e}"))?;
            self.stdout
                .flush()
                .map_err(|e| format!("failed to flush assistant line ending: {e}"))?;
            app.assistant_line_open = false;
        }
        Ok(())
    }

    fn print_prefixed_lines(
        &mut self,
        app: &mut ReplApp,
        prefix: &str,
        text: &str,
    ) -> Result<(), String> {
        self.ensure_output_line_closed(app)?;
        let output_row = self.output_bottom_row_1based().saturating_sub(1);
        for line in text.lines() {
            execute!(
                self.stdout,
                MoveTo(0, output_row),
                Clear(ClearType::CurrentLine),
                Print(prefix),
                Print(line),
                Print("\r\n")
            )
            .map_err(|e| format!("failed to print repl output: {e}"))?;
        }
        if text.is_empty() {
            execute!(
                self.stdout,
                MoveTo(0, output_row),
                Clear(ClearType::CurrentLine),
                Print(prefix),
                Print("\r\n")
            )
            .map_err(|e| format!("failed to print repl output: {e}"))?;
        }
        self.stdout
            .flush()
            .map_err(|e| format!("failed to flush repl output: {e}"))
    }

    fn print_assistant_chunk(&mut self, app: &mut ReplApp, chunk: &str) -> Result<(), String> {
        if chunk.is_empty() {
            return Ok(());
        }
        if should_animate_assistant_chunk(app, chunk) {
            for segment in chunk_segments_for_animation(chunk) {
                self.print_assistant_chunk_immediate(app, segment)?;
                thread::sleep(Duration::from_millis(6));
            }
            return Ok(());
        }
        self.print_assistant_chunk_immediate(app, chunk)
    }

    fn print_assistant_chunk_immediate(
        &mut self,
        app: &mut ReplApp,
        chunk: &str,
    ) -> Result<(), String> {
        let output_row = self.output_bottom_row_1based().saturating_sub(1);
        for segment in chunk.split_inclusive('\n') {
            if !app.assistant_line_open {
                execute!(
                    self.stdout,
                    MoveTo(0, output_row),
                    Clear(ClearType::CurrentLine),
                    Print("[llm] ")
                )
                .map_err(|e| format!("failed to start assistant line: {e}"))?;
                app.assistant_line_open = true;
            }
            execute!(self.stdout, Print(segment))
                .map_err(|e| format!("failed to stream assistant output: {e}"))?;
            if segment.ends_with('\n') {
                app.assistant_line_open = false;
            }
        }
        self.stdout
            .flush()
            .map_err(|e| format!("failed to flush assistant output: {e}"))
    }

    fn render_footer(&mut self, app: &ReplApp) -> Result<(), String> {
        self.refresh_size()?;
        let preserve_output_cursor = app.busy;
        if preserve_output_cursor {
            execute!(self.stdout, SavePosition, Hide)
                .map_err(|e| format!("failed to save cursor: {e}"))?;
        } else {
            execute!(self.stdout, Show).map_err(|e| format!("failed to show cursor: {e}"))?;
        }

        let status_line = compose_status_line(&app.status, self.width as usize);
        let (input_line, cursor_col) =
            compose_input_line(&app.input, app.cursor, self.width as usize);
        let status_row = self.footer_status_row();
        let input_row = self.footer_input_row();

        execute!(
            self.stdout,
            MoveTo(0, status_row),
            Clear(ClearType::CurrentLine),
            Print(status_line),
            MoveTo(0, input_row),
            Clear(ClearType::CurrentLine),
            Print(input_line)
        )
        .map_err(|e| format!("failed to render repl footer: {e}"))?;

        if preserve_output_cursor {
            execute!(self.stdout, RestorePosition)
                .map_err(|e| format!("failed to restore cursor: {e}"))?;
        } else {
            execute!(self.stdout, MoveTo(cursor_col, input_row))
                .map_err(|e| format!("failed to position input cursor: {e}"))?;
        }

        self.stdout
            .flush()
            .map_err(|e| format!("failed to flush repl footer: {e}"))
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let status_row = self.footer_status_row();
        let input_row = self.footer_input_row();
        let _ = write!(self.stdout, "\x1b[r");
        let _ = execute!(
            self.stdout,
            MoveTo(0, status_row),
            Clear(ClearType::CurrentLine),
            MoveTo(0, input_row),
            Clear(ClearType::CurrentLine),
            Show
        );
        let _ = writeln!(self.stdout);
        let _ = self.stdout.flush();
        let _ = disable_raw_mode();
    }
}

enum WorkerCommand {
    RunPrompt {
        prompt: String,
        chat_history: Vec<ChatMessage>,
        active_images: Vec<String>,
    },
    LoadRag {
        encoder_path: Option<String>,
        source_dir: String,
    },
    ClearRag,
    Shutdown,
}

enum WorkerEvent {
    RuntimeReady(Result<(), String>),
    Runtime(RuntimeEvent),
    TurnFinished(Result<(), String>),
    RagLoaded(Result<String, String>),
    RagProgress(String),
}

pub(crate) fn run(cli: &CliOptions) -> Result<(), String> {
    // Install a panic hook that restores the terminal before printing the panic
    // message.  Without this, a panic in any thread (e.g. a worker thread) leaves
    // the terminal in raw mode with a broken scroll region, making the shell
    // unusable until the user runs `reset`.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Best-effort: ignore errors — we are already in a bad state.
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            Show,                // un-hide the cursor
            MoveTo(0, u16::MAX), // move to bottom so output doesn't clobber history
        );
        // Reset scroll region and emit a newline so the shell prompt appears cleanly.
        let _ = writeln!(io::stdout(), "\x1b[r");
        let _ = io::stdout().flush();
        prev_hook(info);
    }));

    let mut terminal = TerminalGuard::new()?;
    let mut app = ReplApp::new();
    let (command_tx, event_rx) = spawn_worker(cli.clone());
    let mut startup_prompt = if cli.prompt.trim().is_empty() {
        None
    } else {
        Some(cli.prompt.trim().to_string())
    };

    terminal.print_prefixed_lines(
        &mut app,
        "[sys] ",
        "Entering repl mode. Type /help for commands.",
    )?;
    if cli.debug {
        for line in collect_debug_banner_lines(cli) {
            terminal.print_prefixed_lines(&mut app, "[dbg] ", &line)?;
        }
    }
    terminal.render_footer(&app)?;

    loop {
        drain_worker_events(&mut app, &mut terminal, &event_rx)?;
        if app.runtime_ready
            && !app.busy
            && let Some(prompt) = startup_prompt.take()
            && dispatch_prompt(&mut app, &mut terminal, &command_tx, cli, &prompt)?
        {
            break;
        }

        terminal.render_footer(&app)?;

        if !event::poll(Duration::from_millis(50)).map_err(|e| format!("event poll failed: {e}"))? {
            continue;
        }
        match event::read().map_err(|e| format!("event read failed: {e}"))? {
            Event::Key(key)
                if handle_key_event(&mut app, &mut terminal, &command_tx, cli, key)? =>
            {
                break;
            }
            Event::Resize(_, _) => {
                terminal.render_footer(&app)?;
            }
            _ => {}
        }
    }

    let _ = command_tx.send(WorkerCommand::Shutdown);

    // Remove our panic hook now that raw mode is no longer active.
    // take_hook() pops the current hook and reinstates the default.
    let _ = std::panic::take_hook();
    Ok(())
}

fn spawn_worker(cli: CliOptions) -> (Sender<WorkerCommand>, Receiver<WorkerEvent>) {
    let (command_tx, command_rx) = mpsc::channel::<WorkerCommand>();
    let (event_tx, event_rx) = mpsc::channel::<WorkerEvent>();
    thread::spawn(move || worker_main(cli, command_rx, event_tx));
    (command_tx, event_rx)
}

fn worker_main(
    cli: CliOptions,
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
) {
    let callback_tx = event_tx.clone();
    let callback: RuntimeEventCallback = std::sync::Arc::new(move |event| {
        let _ = callback_tx.send(WorkerEvent::Runtime(event));
    });

    match ModelRuntime::load_for_repl(&cli) {
        Ok(mut runtime) => {
            runtime.set_debug_mode(cli.debug);
            let _ = event_tx.send(WorkerEvent::RuntimeReady(Ok(())));
            while let Ok(command) = command_rx.recv() {
                match command {
                    WorkerCommand::RunPrompt {
                        prompt,
                        chat_history,
                        active_images,
                    } => {
                        runtime.set_runtime_event_callback(Some(callback.clone()));
                        let result = run_worker_turn(
                            &mut runtime,
                            &cli,
                            &prompt,
                            &chat_history,
                            &active_images,
                            &callback,
                        );
                        runtime.set_runtime_event_callback(None);
                        let _ = event_tx.send(WorkerEvent::TurnFinished(result));
                    }
                    WorkerCommand::LoadRag {
                        encoder_path,
                        source_dir,
                    } => {
                        let progress_tx =
                            std::sync::Arc::new(std::sync::Mutex::new(event_tx.clone()));
                        let progress_cb: std::sync::Arc<dyn Fn(String) + Send + Sync> =
                            std::sync::Arc::new(move |msg| {
                                if let Ok(tx) = progress_tx.lock() {
                                    let _ = tx.send(WorkerEvent::RagProgress(msg));
                                }
                            });
                        let result = runtime.load_rag_from_dir(
                            encoder_path.as_deref(),
                            &source_dir,
                            Some(progress_cb),
                        );
                        let _ = event_tx.send(WorkerEvent::RagLoaded(result));
                    }
                    WorkerCommand::ClearRag => {
                        runtime.clear_rag();
                        let _ = event_tx
                            .send(WorkerEvent::RagLoaded(Ok("RAG index cleared.".to_string())));
                    }
                    WorkerCommand::Shutdown => break,
                }
            }
        }
        Err(err) => {
            let _ = event_tx.send(WorkerEvent::RuntimeReady(Err(err)));
        }
    }
}

fn run_worker_turn(
    runtime: &mut ModelRuntime,
    cli: &CliOptions,
    prompt: &str,
    chat_history: &[ChatMessage],
    active_images: &[String],
    callback: &RuntimeEventCallback,
) -> Result<(), String> {
    if !active_images.is_empty() {
        if cli.tools_enabled {
            emit_runtime_event(
                Some(callback),
                RuntimeEvent::Log(RuntimeLog::system(
                    "Active image context detected; using native multimodal path for this turn.",
                )),
            );
        }
        let request =
            build_repl_multimodal_request(prompt, &cli.system_prompt, chat_history, active_images);
        let output = runtime.generate_request(&request, false)?;
        if output.trim().is_empty() {
            emit_runtime_event(
                Some(callback),
                RuntimeEvent::Log(RuntimeLog::system("<empty response>")),
            );
        }
        return Ok(());
    }

    let shell_tools_enabled = cli.tool_enablement.shell_exec
        || cli.tool_enablement.shell_list_allowed
        || cli.tool_enablement.shell_request_allowed;
    let filesystem_tools_enabled = cli.tool_enablement.read_file
        || cli.tool_enablement.list_dir
        || cli.tool_enablement.write_file
        || cli.tool_enablement.mkdir
        || cli.tool_enablement.rmdir;

    if cli.tools_enabled
        && (runtime.has_rag_index()
            || agent::conversation_likely_requires_tools(
                prompt,
                chat_history,
                filesystem_tools_enabled,
                shell_tools_enabled,
            ))
    {
        agent::run_agent_loop_collect_with_history_callback(
            runtime,
            cli,
            chat_history,
            prompt,
            Some(callback),
        )?;
        return Ok(());
    }

    let mut messages = chat_history.to_vec();
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: prompt.to_string(),
    });
    let output =
        runtime.generate_chat_messages_without_think_for_repl(&messages, &cli.system_prompt)?;
    if output.trim().is_empty() {
        emit_runtime_event(
            Some(callback),
            RuntimeEvent::Log(RuntimeLog::system("<empty response>")),
        );
    }
    Ok(())
}

fn drain_worker_events(
    app: &mut ReplApp,
    terminal: &mut TerminalGuard,
    event_rx: &Receiver<WorkerEvent>,
) -> Result<(), String> {
    loop {
        match event_rx.try_recv() {
            Ok(WorkerEvent::RuntimeReady(Ok(()))) => {
                app.runtime_ready = true;
                app.set_ready_status();
                terminal.print_prefixed_lines(app, "[sys] ", "Runtime ready.")?;
            }
            Ok(WorkerEvent::RuntimeReady(Err(err))) => {
                app.runtime_ready = false;
                app.busy = false;
                app.set_status("Load failed");
                terminal.print_prefixed_lines(
                    app,
                    "[err] ",
                    &format!("Runtime load failed: {err}"),
                )?;
            }
            Ok(WorkerEvent::Runtime(event)) => match event {
                RuntimeEvent::Output(text) => {
                    let render = app.absorb_assistant_output(&text);
                    if !render.is_empty() {
                        terminal.print_assistant_chunk(app, &render)?;
                    }
                }
                RuntimeEvent::Log(log) => match log.kind {
                    RuntimeLogKind::Debug => {
                        terminal.print_prefixed_lines(app, "[dbg] ", &log.message)?
                    }
                    RuntimeLogKind::System => {
                        terminal.print_prefixed_lines(app, "[sys] ", &log.message)?
                    }
                    RuntimeLogKind::Error => {
                        terminal.print_prefixed_lines(app, "[err] ", &log.message)?
                    }
                },
                RuntimeEvent::Status(status) => app.set_status_override(status),
                RuntimeEvent::Progress(progress) => app.set_progress(progress),
            },
            Ok(WorkerEvent::TurnFinished(Ok(()))) => {
                app.busy = false;
                app.finish_turn_progress();
                app.clear_status_override();
                terminal.ensure_output_line_closed(app)?;
                if let Some(prompt) = app.pending_user_prompt.take() {
                    app.chat_history.push(ChatMessage {
                        role: ChatRole::User,
                        content: prompt,
                    });
                    let assistant = app.pending_assistant_output.trim().to_string();
                    if !assistant.is_empty() {
                        app.chat_history.push(ChatMessage {
                            role: ChatRole::Assistant,
                            content: assistant,
                        });
                    }
                }
                app.pending_assistant_output.clear();
                if app.runtime_ready {
                    app.set_ready_status();
                }
            }
            Ok(WorkerEvent::TurnFinished(Err(err))) => {
                app.busy = false;
                app.finish_turn_progress();
                app.clear_status_override();
                app.pending_user_prompt = None;
                app.pending_assistant_output.clear();
                terminal.ensure_output_line_closed(app)?;
                app.set_status("Error");
                terminal.print_prefixed_lines(app, "[err] ", &format!("Turn failed: {err}"))?;
            }
            Ok(WorkerEvent::RagProgress(msg)) => {
                app.set_status(msg);
            }
            Ok(WorkerEvent::RagLoaded(Ok(msg))) => {
                app.busy = false;
                app.rag_loaded = !msg.contains("cleared");
                terminal.print_prefixed_lines(app, "[sys] ", &msg)?;
                if app.runtime_ready {
                    app.set_ready_status();
                }
            }
            Ok(WorkerEvent::RagLoaded(Err(err))) => {
                app.busy = false;
                app.rag_loaded = false;
                terminal.print_prefixed_lines(app, "[err] ", &format!("RAG load failed: {err}"))?;
                if app.runtime_ready {
                    app.set_ready_status();
                }
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                return Err("repl worker disconnected unexpectedly".to_string());
            }
        }
    }
    Ok(())
}

fn handle_key_event(
    app: &mut ReplApp,
    terminal: &mut TerminalGuard,
    command_tx: &Sender<WorkerCommand>,
    cli: &CliOptions,
    key: KeyEvent,
) -> Result<bool, String> {
    if !should_handle_key_event(key) {
        return Ok(false);
    }
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(true),
        KeyCode::Esc => return Ok(true),
        KeyCode::Enter => {
            let submitted = app.input.trim().to_string();
            if submitted.is_empty() {
                return Ok(false);
            }
            app.push_history(&submitted);
            app.clear_input();
            if dispatch_prompt(app, terminal, command_tx, cli, &submitted)? {
                return Ok(true);
            }
        }
        KeyCode::Tab => app.apply_tab_completion(),
        KeyCode::Backspace => app.backspace(),
        KeyCode::Delete => app.delete(),
        KeyCode::Left => app.move_left(),
        KeyCode::Right => app.move_right(),
        KeyCode::Home => app.cursor = 0,
        KeyCode::End => app.cursor = app.input.len(),
        KeyCode::Up => app.history_up(),
        KeyCode::Down => app.history_down(),
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => app.insert_char(ch),
        _ => {}
    }
    Ok(false)
}

fn should_handle_key_event(key: KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
}

fn dispatch_prompt(
    app: &mut ReplApp,
    terminal: &mut TerminalGuard,
    command_tx: &Sender<WorkerCommand>,
    cli: &CliOptions,
    input: &str,
) -> Result<bool, String> {
    terminal.print_prefixed_lines(app, "[you] ", &format!("> {input}"))?;
    match handle_repl_command(cli, input) {
        crate::app::ReplCommandAction::Exit => return Ok(true),
        crate::app::ReplCommandAction::Messages(lines) => {
            for line in lines {
                terminal.print_prefixed_lines(app, "[sys] ", &line)?;
            }
            if !app.busy && app.runtime_ready {
                app.set_ready_status();
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::AttachImage(path) => {
            let canonical = crate::app::validate_repl_image_path(&path)?;
            if app.active_images.contains(&canonical) {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    &format!("Image already attached: {canonical}"),
                )?;
            } else {
                app.active_images.push(canonical.clone());
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    &format!("Attached image: {canonical}"),
                )?;
            }
            if !app.busy && app.runtime_ready {
                app.set_ready_status();
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::ListImages => {
            if app.active_images.is_empty() {
                terminal.print_prefixed_lines(app, "[sys] ", "No active image attachments.")?;
            } else {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    &format!("Active images ({}):", app.active_images.len()),
                )?;
                for image in app.active_images.clone() {
                    terminal.print_prefixed_lines(app, "[sys] ", &format!("  {image}"))?;
                }
            }
            if !app.busy && app.runtime_ready {
                app.set_ready_status();
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::ClearImages => {
            let cleared = app.active_images.len();
            app.active_images.clear();
            terminal.print_prefixed_lines(
                app,
                "[sys] ",
                &format!("Cleared {cleared} active image attachment(s)."),
            )?;
            if !app.busy && app.runtime_ready {
                app.set_ready_status();
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::ClearState => {
            if app.busy {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    "Cannot clear state while a turn is running.",
                )?;
                return Ok(false);
            }
            let cleared_messages = app.chat_history.len();
            let cleared_images = app.active_images.len();
            app.chat_history.clear();
            app.active_images.clear();
            app.pending_user_prompt = None;
            app.pending_assistant_output.clear();
            app.assistant_line_open = false;
            terminal.print_prefixed_lines(
                app,
                "[sys] ",
                &format!(
                    "Cleared chat state: {} message(s), {} image attachment(s).",
                    cleared_messages, cleared_images
                ),
            )?;
            if app.runtime_ready {
                app.set_ready_status();
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::LoadDocSource(source_dir) => {
            if !app.runtime_ready {
                terminal.print_prefixed_lines(app, "[err] ", "Runtime is not ready yet.")?;
                return Ok(false);
            }
            if app.busy {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    "A turn is already running. Wait for it to finish before loading documents.",
                )?;
                return Ok(false);
            }
            app.busy = true;
            app.set_status("Building knowledge index…");
            let _ = command_tx.send(WorkerCommand::LoadRag {
                encoder_path: cli.rag_encoder.clone(),
                source_dir,
            });
            return Ok(false);
        }
        crate::app::ReplCommandAction::ClearDocs => {
            if !app.runtime_ready {
                terminal.print_prefixed_lines(app, "[err] ", "Runtime is not ready yet.")?;
                return Ok(false);
            }
            if app.busy {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    "A turn is already running. Wait for it to finish before clearing documents.",
                )?;
                return Ok(false);
            }
            app.busy = true;
            let _ = command_tx.send(WorkerCommand::ClearRag);
            return Ok(false);
        }
        crate::app::ReplCommandAction::DocStatus => {
            let lines = if app.rag_loaded {
                vec![format!("Knowledge active — top-k: {}", cli.rag_top_k,)]
            } else {
                vec!["No knowledge loaded. Use /doc <dir> to load a wiki directory.".to_string()]
            };
            for line in lines {
                terminal.print_prefixed_lines(app, "[sys] ", &line)?;
            }
            if !app.busy && app.runtime_ready {
                app.set_ready_status();
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::ModelPrompt(prompt) => {
            if !app.runtime_ready {
                terminal.print_prefixed_lines(app, "[err] ", "Runtime is not ready yet.")?;
                return Ok(false);
            }
            if app.busy {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    "A turn is already running. Wait for it to finish before submitting another prompt.",
                )?;
                return Ok(false);
            }
            app.busy = true;
            app.set_status("Running model...");
            app.pending_user_prompt = Some(prompt.clone());
            app.pending_assistant_output.clear();
            app.assistant_line_open = false;
            command_tx
                .send(WorkerCommand::RunPrompt {
                    prompt,
                    chat_history: app.chat_history.clone(),
                    active_images: app.active_images.clone(),
                })
                .map_err(|e| format!("failed to send prompt to repl worker: {e}"))?;
        }
    }
    Ok(false)
}

fn build_repl_multimodal_request(
    prompt: &str,
    system_prompt: &str,
    chat_history: &[ChatMessage],
    active_images: &[String],
) -> crate::engine::types::GenerationRequest {
    const MAX_HISTORY_MESSAGES: usize = 12;
    let history_slice = if chat_history.len() > MAX_HISTORY_MESSAGES {
        &chat_history[chat_history.len() - MAX_HISTORY_MESSAGES..]
    } else {
        chat_history
    };

    let mut prompt_text = String::new();
    if !history_slice.is_empty() {
        prompt_text.push_str("Conversation so far:\n");
        for message in history_slice {
            match message.role {
                ChatRole::User => prompt_text.push_str("User: "),
                ChatRole::Assistant => prompt_text.push_str("Assistant: "),
            }
            prompt_text.push_str(&message.content);
            prompt_text.push('\n');
        }
        prompt_text.push('\n');
    }
    prompt_text.push_str(&format!(
        "There are {} active image attachment(s) for this conversation.\nCurrent user message: {}",
        active_images.len(),
        prompt
    ));

    let mut parts = Vec::with_capacity(active_images.len().saturating_add(1));
    for path in active_images {
        parts.push(crate::engine::types::ContentPart::Image(
            crate::engine::types::MediaRef { path: path.clone() },
        ));
    }
    parts.push(crate::engine::types::ContentPart::Text(prompt_text));
    crate::engine::types::GenerationRequest {
        system_prompt: system_prompt.to_string(),
        parts,
        include_empty_system_prompt: false,
        assistant_prefill: None,
    }
}

fn compose_status_line(status: &ReplStatus, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let left = truncate_for_width(&status.left, width);
    if status.right.is_empty() {
        return left;
    }
    let right = truncate_for_width(&status.right, width);
    let left_len = left.chars().count();
    let right_len = right.chars().count();
    if left_len + right_len + 1 > width {
        return truncate_for_width(&format!("{} {}", left, right), width);
    }
    format!(
        "{}{}{}",
        left,
        " ".repeat(width - left_len - right_len),
        right
    )
}

fn format_runner_status(status: &RunnerStatus) -> String {
    match status {
        RunnerStatus::Planning { turn, max_turns } => {
            format!("agent turn {turn}/{max_turns} | planning")
        }
        RunnerStatus::Tool {
            turn,
            max_turns,
            tool,
        } => format!("agent turn {turn}/{max_turns} | tool {tool}"),
        RunnerStatus::Finalizing { turn, max_turns } => {
            format!("agent turn {turn}/{max_turns} | finalizing")
        }
        RunnerStatus::Recovering { turn, max_turns } => {
            format!("agent turn {turn}/{max_turns} | recovering")
        }
    }
}

fn compose_input_line(input: &str, cursor: usize, width: usize) -> (String, u16) {
    let prefix = "> ";
    if width <= prefix.len() {
        return (prefix[..width].to_string(), width as u16);
    }
    let available = width - prefix.len();
    let chars = input.chars().collect::<Vec<_>>();
    let cursor_chars = input[..min(cursor, input.len())].chars().count();
    let start = cursor_chars.saturating_sub(available.saturating_sub(1));
    let end = min(chars.len(), start + available);
    let visible = chars[start..end].iter().collect::<String>();
    let mut line = String::with_capacity(width);
    line.push_str(prefix);
    line.push_str(&visible);
    let visible_len = visible.chars().count();
    if prefix.len() + visible_len < width {
        line.push_str(&" ".repeat(width - prefix.len() - visible_len));
    }
    let cursor_col = (prefix.len() + cursor_chars.saturating_sub(start)).min(width);
    (line, cursor_col as u16)
}

fn truncate_for_width(text: &str, width: usize) -> String {
    text.chars().take(width).collect()
}

fn should_animate_assistant_chunk(app: &ReplApp, chunk: &str) -> bool {
    !app.assistant_line_open && chunk.chars().count() >= 96
}

fn chunk_segments_for_animation(text: &str) -> Vec<&str> {
    const TARGET_SEGMENT_CHARS: usize = 20;
    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut current_len = 0usize;
    let mut saw_split = false;

    for (idx, ch) in text.char_indices() {
        current_len += 1;
        let boundary = ch == '\n' || (ch.is_whitespace() && current_len >= TARGET_SEGMENT_CHARS);
        if boundary {
            let end = idx + ch.len_utf8();
            if end > start {
                segments.push(&text[start..end]);
                start = end;
                current_len = 0;
                saw_split = true;
            }
        }
    }

    if start < text.len() {
        segments.push(&text[start..]);
        saw_split = true;
    }

    if saw_split { segments } else { vec![text] }
}

fn context_gauge(used: usize, limit: usize, width: usize) -> String {
    if limit == 0 || width == 0 {
        return String::new();
    }
    let used = used.min(limit);
    let filled = ((used * width) + (limit / 2)) / limit;
    let percent = ((used * 100) + (limit / 2)) / limit;
    let filled = filled.min(width);
    format!(
        "[{}{}] {:>3}%",
        "#".repeat(filled),
        ".".repeat(width - filled),
        percent
    )
}

fn format_compact_token_count(count: usize) -> String {
    if count < 1_000 {
        return count.to_string();
    }
    if count < 1_000_000 {
        let value = count as f64 / 1_000.0;
        if value < 10.0 {
            format!("{value:.1}k")
        } else {
            format!("{:.0}k", value)
        }
    } else {
        let value = count as f64 / 1_000_000.0;
        if value < 10.0 {
            format!("{value:.1}M")
        } else {
            format!("{:.0}M", value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ReplApp, chunk_segments_for_animation, format_compact_token_count,
        should_animate_assistant_chunk, should_handle_key_event,
    };
    use crate::app::events::{RunnerStatus, RuntimePhase, RuntimeProgress};
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    #[test]
    fn key_event_filter_ignores_release_events() {
        let press = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        let repeat = KeyEvent {
            code: KeyCode::Char('a'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Repeat,
            state: KeyEventState::NONE,
        };
        let release = KeyEvent {
            code: KeyCode::Char('a'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        };

        assert!(should_handle_key_event(press));
        assert!(should_handle_key_event(repeat));
        assert!(!should_handle_key_event(release));
    }

    #[test]
    fn hidden_think_progress_updates_status_label() {
        let mut app = ReplApp::new();
        app.set_progress(RuntimeProgress {
            phase: RuntimePhase::Decode,
            prefill_tokens: 128,
            decode_tokens: 48,
            hidden_thinking: true,
            hidden_think_tokens: 32,
            tokens_per_second: Some(12.5),
            context_used: 176,
            context_limit: 1024,
        });

        assert_eq!(app.status.left, "thinking 32 tok | decode 48 tok");
        assert!(app.status.right.contains("out 48 tok"));
        assert!(app.status.right.contains("12.5 tok/s"));
        assert!(app.status.right.contains("ctx ["));
    }

    #[test]
    fn status_override_survives_progress_updates() {
        let mut app = ReplApp::new();
        app.set_status_override(RunnerStatus::Tool {
            turn: 2,
            max_turns: 8,
            tool: "read_file".to_string(),
        });
        app.set_progress(RuntimeProgress {
            phase: RuntimePhase::Decode,
            prefill_tokens: 256,
            decode_tokens: 64,
            hidden_thinking: false,
            hidden_think_tokens: 0,
            tokens_per_second: Some(9.4),
            context_used: 320,
            context_limit: 2048,
        });

        assert_eq!(app.status.left, "agent turn 2/8 | tool read_file");
        assert!(app.status.right.contains("out 64 tok"));
        assert!(app.status.right.contains("9.4 tok/s"));
        assert!(app.status.right.contains("ctx ["));
    }

    #[test]
    fn session_output_counter_accumulates_across_turns() {
        let mut app = ReplApp::new();
        app.set_progress(RuntimeProgress {
            phase: RuntimePhase::Decode,
            prefill_tokens: 100,
            decode_tokens: 20,
            hidden_thinking: false,
            hidden_think_tokens: 0,
            tokens_per_second: Some(8.0),
            context_used: 120,
            context_limit: 1024,
        });
        app.set_progress(RuntimeProgress {
            phase: RuntimePhase::Ready,
            prefill_tokens: 100,
            decode_tokens: 30,
            hidden_thinking: false,
            hidden_think_tokens: 0,
            tokens_per_second: Some(8.0),
            context_used: 130,
            context_limit: 1024,
        });
        app.finish_turn_progress();
        app.set_progress(RuntimeProgress {
            phase: RuntimePhase::Decode,
            prefill_tokens: 80,
            decode_tokens: 5,
            hidden_thinking: false,
            hidden_think_tokens: 0,
            tokens_per_second: Some(7.5),
            context_used: 85,
            context_limit: 1024,
        });

        assert!(app.status.right.contains("out 35 tok"));
    }

    #[test]
    fn compact_token_count_formats_large_values() {
        assert_eq!(format_compact_token_count(500), "500");
        assert_eq!(format_compact_token_count(1_500), "1.5k");
        assert_eq!(format_compact_token_count(12_300), "12k");
        assert_eq!(format_compact_token_count(1_700_000), "1.7M");
    }

    #[test]
    fn ready_status_preserves_session_stats() {
        let mut app = ReplApp::new();
        app.set_progress(RuntimeProgress {
            phase: RuntimePhase::Decode,
            prefill_tokens: 128,
            decode_tokens: 48,
            hidden_thinking: false,
            hidden_think_tokens: 0,
            tokens_per_second: Some(12.5),
            context_used: 176,
            context_limit: 1024,
        });

        let previous_right = app.status.right.clone();
        app.set_ready_status();

        assert_eq!(app.status.left, "Ready");
        assert_eq!(app.status.right, previous_right);
    }

    #[test]
    fn non_ready_status_clears_previous_stats() {
        let mut app = ReplApp::new();
        app.set_progress(RuntimeProgress {
            phase: RuntimePhase::Decode,
            prefill_tokens: 128,
            decode_tokens: 48,
            hidden_thinking: false,
            hidden_think_tokens: 0,
            tokens_per_second: Some(12.5),
            context_used: 176,
            context_limit: 1024,
        });

        app.set_status("Running model...");

        assert_eq!(app.status.left, "Running model...");
        assert!(app.status.right.is_empty());
    }

    #[test]
    fn large_single_output_chunk_is_animated() {
        let app = ReplApp::new();
        let chunk = "This is a long assistant output chunk that should be replayed visibly in smaller pieces instead of appearing all at once in the terminal footer UI.";
        assert!(should_animate_assistant_chunk(&app, chunk));
        let segments = chunk_segments_for_animation(chunk);
        assert!(segments.len() > 1);
        assert_eq!(segments.concat(), chunk);
    }
}
