use clap::Parser;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use crate::tools;

fn parse_allowed_tools(raw: &str) -> Result<BTreeSet<String>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(
            "invalid value '': expected comma-separated tool names, or all/none".to_string(),
        );
    }
    if trimmed.eq_ignore_ascii_case("none") {
        return Ok(BTreeSet::new());
    }
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok(tools::all_tool_name_set());
    }
    let mut parsed = BTreeSet::new();
    for entry in trimmed.split(',') {
        let tool = entry.trim();
        if tool.is_empty() {
            return Err(format!(
                "invalid value '{raw}': empty tool name in comma-separated list"
            ));
        }
        if !tools::is_valid_tool_name(tool) {
            return Err(format!(
                "invalid tool '{tool}': expected one of {}",
                tools::ALL_TOOL_NAMES.join(", ")
            ));
        }
        parsed.insert(tool.to_string());
    }
    Ok(parsed)
}

fn parse_top_p(raw: &str) -> Result<f32, String> {
    let v = raw
        .parse::<f32>()
        .map_err(|e| format!("invalid value '{raw}': {e}"))?;
    if v > 0.0 && v <= 1.0 {
        Ok(v)
    } else {
        Err(format!("invalid value '{raw}': expected value in (0, 1]"))
    }
}

fn parse_positive_f32(raw: &str) -> Result<f32, String> {
    let v = raw
        .parse::<f32>()
        .map_err(|e| format!("invalid value '{raw}': {e}"))?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err(format!("invalid value '{raw}': expected > 0"))
    }
}

fn parse_positive_usize(raw: &str) -> Result<usize, String> {
    let v = raw
        .parse::<usize>()
        .map_err(|e| format!("invalid value '{raw}': {e}"))?;
    if v > 0 {
        Ok(v)
    } else {
        Err(format!("invalid value '{raw}': expected >= 1"))
    }
}

#[cfg(target_arch = "aarch64")]
fn parse_nonnegative_usize(raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|e| format!("invalid value '{raw}': {e}"))
}

fn parse_boolish(raw: &str) -> Result<bool, String> {
    let v = raw.trim();
    if v.eq_ignore_ascii_case("1")
        || v.eq_ignore_ascii_case("true")
        || v.eq_ignore_ascii_case("yes")
        || v.eq_ignore_ascii_case("on")
    {
        return Ok(true);
    }
    if v.eq_ignore_ascii_case("0")
        || v.eq_ignore_ascii_case("false")
        || v.eq_ignore_ascii_case("no")
        || v.eq_ignore_ascii_case("off")
    {
        return Ok(false);
    }
    Err(format!(
        "invalid value '{raw}': expected one of 1/0/true/false/yes/no/on/off"
    ))
}

fn parse_think_mode(raw: &str) -> Result<crate::engine::types::ThinkMode, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "yes" | "on" | "true" | "1" => Ok(crate::engine::types::ThinkMode::Yes),
        "no" | "off" | "false" | "0" => Ok(crate::engine::types::ThinkMode::No),
        "hidden" => Ok(crate::engine::types::ThinkMode::Hidden),
        _ => Err(format!("invalid value '{raw}': expected yes/no/hidden")),
    }
}

fn parse_kv_cache_mode(raw: &str) -> Result<CliKvCacheMode, String> {
    let v = raw.trim();
    if v.eq_ignore_ascii_case("q8") {
        Ok(CliKvCacheMode::Q8)
    } else if v.eq_ignore_ascii_case("turbo") || v.eq_ignore_ascii_case("tq") {
        Ok(CliKvCacheMode::Turbo)
    } else {
        Err(format!("invalid value '{raw}': expected one of q8/turbo"))
    }
}

fn parse_operation_mode(raw: &str) -> Result<CliOperationMode, String> {
    let v = raw.trim();
    if v.eq_ignore_ascii_case("oneshot") || v.eq_ignore_ascii_case("single") {
        Ok(CliOperationMode::Oneshot)
    } else if v.eq_ignore_ascii_case("repl")
        || v.eq_ignore_ascii_case("interactive")
        || v.eq_ignore_ascii_case("shell")
    {
        Ok(CliOperationMode::Repl)
    } else {
        Err(format!(
            "invalid value '{raw}': expected one of oneshot/repl"
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CliKvCacheMode {
    Q8,
    Turbo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CliOperationMode {
    Oneshot,
    Repl,
}

#[derive(Clone, Debug)]
pub(crate) struct ToolPromptSpec {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) when_to_use: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ShellCommandDescriptionSpec {
    pub(crate) command: String,
    pub(crate) description: String,
}

#[derive(Clone, Debug)]
pub(crate) struct AgentToolEnablement {
    pub(crate) read_file: bool,
    pub(crate) list_dir: bool,
    pub(crate) write_file: bool,
    pub(crate) mkdir: bool,
    pub(crate) rmdir: bool,
    pub(crate) shell_list_allowed: bool,
    pub(crate) shell_exec: bool,
    pub(crate) shell_request_allowed: bool,
}

impl Default for AgentToolEnablement {
    fn default() -> Self {
        Self {
            read_file: true,
            list_dir: true,
            write_file: true,
            mkdir: true,
            rmdir: true,
            shell_list_allowed: true,
            shell_exec: true,
            shell_request_allowed: true,
        }
    }
}

impl AgentToolEnablement {
    fn disabled() -> Self {
        Self {
            read_file: false,
            list_dir: false,
            write_file: false,
            mkdir: false,
            rmdir: false,
            shell_list_allowed: false,
            shell_exec: false,
            shell_request_allowed: false,
        }
    }

    fn apply_allowlist(&mut self, allowlist: &BTreeSet<String>) {
        self.read_file &= allowlist.contains(tools::READ_FILE);
        self.list_dir &= allowlist.contains(tools::LIST_DIR);
        self.write_file &= allowlist.contains(tools::WRITE_FILE);
        self.mkdir &= allowlist.contains(tools::MKDIR);
        self.rmdir &= allowlist.contains(tools::RMDIR);
        self.shell_list_allowed &= allowlist.contains(tools::SHELL_LIST_ALLOWED);
        self.shell_exec &= allowlist.contains(tools::SHELL_EXEC);
        self.shell_request_allowed &= allowlist.contains(tools::SHELL_REQUEST_ALLOWED);
    }

    fn has_any_enabled(&self) -> bool {
        self.read_file
            || self.list_dir
            || self.write_file
            || self.mkdir
            || self.rmdir
            || self.shell_list_allowed
            || self.shell_exec
            || self.shell_request_allowed
    }

    pub(crate) fn enabled_tool_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.read_file {
            names.push(tools::READ_FILE);
        }
        if self.list_dir {
            names.push(tools::LIST_DIR);
        }
        if self.write_file {
            names.push(tools::WRITE_FILE);
        }
        if self.mkdir {
            names.push(tools::MKDIR);
        }
        if self.rmdir {
            names.push(tools::RMDIR);
        }
        if self.shell_list_allowed {
            names.push(tools::SHELL_LIST_ALLOWED);
        }
        if self.shell_exec {
            names.push(tools::SHELL_EXEC);
        }
        if self.shell_request_allowed {
            names.push(tools::SHELL_REQUEST_ALLOWED);
        }
        names
    }
}

#[derive(Debug, Default, Deserialize)]
struct RunnerConfig {
    shell: Option<RunnerShellConfig>,
    tools: Option<RunnerToolsConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct RunnerShellConfig {
    #[serde(alias = "md")]
    cmd: Option<BTreeMap<String, String>>,
    allowed_commands: Option<Vec<RunnerAllowedCommandEntry>>,
    allowed_command_descriptions: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Default, Deserialize)]
struct RunnerToolsConfig {
    read_file: Option<bool>,
    list_dir: Option<bool>,
    write_file: Option<bool>,
    mkdir: Option<bool>,
    rmdir: Option<bool>,
    shell_list_allowed: Option<bool>,
    shell_exec: Option<bool>,
    shell_request_allowed: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RunnerAllowedCommandEntry {
    Name(String),
    Spec(RunnerAllowedCommandSpec),
}

#[derive(Debug, Deserialize)]
struct RunnerAllowedCommandSpec {
    name: String,
    description: Option<String>,
}

#[derive(Default)]
struct LoadedShellConfig {
    allowed_commands: Vec<String>,
    description_specs: Vec<ShellCommandDescriptionSpec>,
}

fn default_tool_prompt_specs() -> Vec<ToolPromptSpec> {
    vec![
        ToolPromptSpec {
            name: tools::READ_FILE.to_string(),
            description: "Read UTF-8 file content under tool_root with a bounded byte limit."
                .to_string(),
            when_to_use: "Use when you need the contents of a specific file before reasoning or editing."
                .to_string(),
        },
        ToolPromptSpec {
            name: tools::LIST_DIR.to_string(),
            description: "List directory entries under tool_root.".to_string(),
            when_to_use: "Use when you need to discover paths before reading or writing files."
                .to_string(),
        },
        ToolPromptSpec {
            name: tools::WRITE_FILE.to_string(),
            description: "Write or append UTF-8 file content under tool_root.".to_string(),
            when_to_use: "Use only when the user explicitly requests file creation/modification."
                .to_string(),
        },
        ToolPromptSpec {
            name: tools::MKDIR.to_string(),
            description: "Create a directory under tool_root recursively (mkdir -p behavior)."
                .to_string(),
            when_to_use:
                "Use when a directory path is needed before creating files in nested locations."
                    .to_string(),
        },
        ToolPromptSpec {
            name: tools::RMDIR.to_string(),
            description:
                "Remove a directory under tool_root recursively (including all children)."
                    .to_string(),
            when_to_use:
                "Use only when the user explicitly asks to delete directories and their contents."
                    .to_string(),
        },
        ToolPromptSpec {
            name: tools::SHELL_LIST_ALLOWED.to_string(),
            description: "Return currently enabled tools and allowed shell commands.".to_string(),
            when_to_use:
                "Use first when you are unsure which tool operations/commands are currently allowed."
                    .to_string(),
        },
        ToolPromptSpec {
            name: tools::SHELL_EXEC.to_string(),
            description:
                "Run an allowed external command with explicit argv (no shell expression). Args schema: {\"command\":\"<allowed>\",\"args\":[...],\"cwd\":\"optional\",\"max_output_bytes\":131072}. Supports built-in helper command `cwd`."
                    .to_string(),
            when_to_use:
                "Use when command output is needed and the command exists in allowed shell commands."
                    .to_string(),
        },
        ToolPromptSpec {
            name: tools::SHELL_REQUEST_ALLOWED.to_string(),
            description:
                "Request operator approval for a command that is not currently in allowed shell commands."
                    .to_string(),
            when_to_use:
                "Use when shell_exec cannot run because a needed command is not currently allowed."
                    .to_string(),
        },
    ]
}

fn config_paths() -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join(".gguf-runner").join("config.toml");
        paths.push(p);
    }
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read current directory: {e}"))?;
    paths.push(cwd.join(".gguf-runner").join("config.toml"));
    Ok(paths)
}

fn load_shell_config_from_config() -> Result<LoadedShellConfig, String> {
    let mut allowed_commands = Vec::new();
    let mut descriptions = BTreeMap::new();
    for path in config_paths()? {
        let content = match fs::read_to_string(&path) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("cannot read config '{}': {e}", path.display())),
        };
        let parsed: RunnerConfig = toml::from_str(&content)
            .map_err(|e| format!("invalid config TOML '{}': {e}", path.display()))?;
        if let Some(shell) = parsed.shell {
            if let Some(cmd_entries) = shell.cmd {
                let (new_allowed_commands, new_descriptions) = parse_shell_cmd_entries(cmd_entries);
                allowed_commands = new_allowed_commands;
                descriptions = new_descriptions;
                continue;
            }
            if let Some(commands) = shell.allowed_commands {
                let (new_allowed_commands, legacy_descriptions) =
                    parse_allowed_command_entries(commands);
                allowed_commands = new_allowed_commands;
                descriptions = legacy_descriptions;
            }
            if let Some(extra_descriptions) = shell.allowed_command_descriptions {
                for (raw_command, raw_description) in extra_descriptions {
                    let Some(description) = normalize_description_text(raw_description) else {
                        continue;
                    };
                    for command in split_shell_command_names(&raw_command) {
                        descriptions.insert(command, description.clone());
                    }
                }
            }
        }
    }
    let description_specs = allowed_commands
        .iter()
        .filter_map(|command| {
            descriptions
                .get(command)
                .map(|description| ShellCommandDescriptionSpec {
                    command: command.clone(),
                    description: description.clone(),
                })
        })
        .collect();
    Ok(LoadedShellConfig {
        allowed_commands,
        description_specs,
    })
}

fn load_tool_enablement_from_config() -> Result<AgentToolEnablement, String> {
    let mut tool_enablement = AgentToolEnablement::default();
    for path in config_paths()? {
        let content = match fs::read_to_string(&path) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("cannot read config '{}': {e}", path.display())),
        };
        let parsed: RunnerConfig = toml::from_str(&content)
            .map_err(|e| format!("invalid config TOML '{}': {e}", path.display()))?;
        if let Some(tools) = parsed.tools {
            if let Some(v) = tools.read_file {
                tool_enablement.read_file = v;
            }
            if let Some(v) = tools.list_dir {
                tool_enablement.list_dir = v;
            }
            if let Some(v) = tools.write_file {
                tool_enablement.write_file = v;
            }
            if let Some(v) = tools.mkdir {
                tool_enablement.mkdir = v;
            }
            if let Some(v) = tools.rmdir {
                tool_enablement.rmdir = v;
            }
            if let Some(v) = tools.shell_list_allowed {
                tool_enablement.shell_list_allowed = v;
            }
            if let Some(v) = tools.shell_exec {
                tool_enablement.shell_exec = v;
            }
            if let Some(v) = tools.shell_request_allowed {
                tool_enablement.shell_request_allowed = v;
            }
        }
    }
    Ok(tool_enablement)
}

fn parse_shell_cmd_entries(
    cmd_entries: BTreeMap<String, String>,
) -> (Vec<String>, BTreeMap<String, String>) {
    let mut raw_names = Vec::new();
    let mut descriptions = BTreeMap::new();
    for (raw_command, raw_description) in cmd_entries {
        let names = split_shell_command_names(&raw_command);
        raw_names.extend(names.iter().cloned());
        if let Some(description) = normalize_description_text(raw_description) {
            for name in names {
                descriptions.insert(name, description.clone());
            }
        }
    }
    let allowed_commands = normalize_shell_command_values(raw_names);
    (allowed_commands, descriptions)
}

fn normalize_shell_command_values<I>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut uniq = BTreeSet::new();
    for raw in values {
        for part in raw.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                uniq.insert(trimmed.to_string());
            }
        }
    }
    uniq.into_iter().collect()
}

fn parse_allowed_command_entries(
    entries: Vec<RunnerAllowedCommandEntry>,
) -> (Vec<String>, BTreeMap<String, String>) {
    let mut raw_names = Vec::new();
    let mut descriptions = BTreeMap::new();
    for entry in entries {
        match entry {
            RunnerAllowedCommandEntry::Name(name) => raw_names.push(name),
            RunnerAllowedCommandEntry::Spec(spec) => {
                let names = split_shell_command_names(&spec.name);
                raw_names.extend(names.iter().cloned());
                if let Some(description) = spec.description.and_then(normalize_description_text) {
                    for name in names {
                        descriptions.insert(name, description.clone());
                    }
                }
            }
        }
    }
    let allowed_commands = normalize_shell_command_values(raw_names);
    (allowed_commands, descriptions)
}

fn split_shell_command_names(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn normalize_description_text(raw: String) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[derive(Parser, Debug)]
#[command(
    about = "Run GGUF language models",
    long_about = None,
    disable_help_subcommand = true
)]
struct Cli {
    #[arg(
        long,
        required_unless_present = "show_features",
        default_value = "",
        value_name = "model.gguf"
    )]
    model: String,

    #[arg(long, default_value = "")]
    prompt: String,

    /// Render a prefill-cache blob for --system-prompt and exit.
    #[arg(long = "render-prefill-cache", value_name = "path")]
    render_prefill_cache: Option<String>,

    /// Load a prefill-cache blob before generation.
    #[arg(long = "prefill-cache", value_name = "path")]
    prefill_cache: Option<String>,

    #[arg(
        long = "show-features",
        help = "Print CPU features (compiled-in vs runtime) and exit"
    )]
    show_features: bool,

    #[arg(long = "image", value_name = "path")]
    images: Vec<String>,

    #[arg(long = "video", value_name = "path")]
    videos: Vec<String>,

    #[arg(long = "audio", value_name = "path")]
    audios: Vec<String>,

    /// Force the language for Qwen3-ASR transcription (for example, English).
    #[arg(long = "audio-language", value_name = "language")]
    audio_language: Option<String>,

    /// Process a JSONL manifest of independent offline audio transcription requests.
    #[arg(long = "audio-batch", value_name = "manifest.jsonl")]
    audio_batch: Option<String>,

    /// Sampling temperature (default: model hint or 0.9).
    #[arg(long)]
    temperature: Option<f32>,

    /// Top-k sampling (default: model hint or 0 = disabled).
    #[arg(long = "top-k")]
    top_k: Option<usize>,

    /// Top-p nucleus sampling; only used when top-k > 0 (default: model hint or 1.0).
    #[arg(
        long = "top-p",
        value_parser = parse_top_p,
    )]
    top_p: Option<f32>,

    #[arg(long, env = "GGUF_SEED")]
    seed: Option<u64>,

    #[arg(
        long = "repeat-penalty",
        value_parser = parse_positive_f32,
        default_value_t = 1.0
    )]
    repeat_penalty: f32,

    #[arg(long = "repeat-last-n", default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long = "max-tokens", default_value_t = 0)]
    max_tokens: usize,

    #[arg(long = "context-size", default_value_t = 0)]
    context_size: usize,

    #[arg(
        long,
        env = "GGUF_RAYON_THREADS",
        value_parser = parse_positive_usize
    )]
    threads: Option<usize>,

    #[arg(long = "system-prompt", default_value = "You are a helpful assistant.")]
    system_prompt: String,

    #[arg(
        long = "mode",
        value_parser = parse_operation_mode,
        default_value = "oneshot",
        help = "Operation mode: oneshot (single request) or repl (interactive loop)"
    )]
    mode: CliOperationMode,

    #[arg(
        long = "allowed-tools",
        env = "GGUF_ALLOWED_TOOLS",
        value_name = "list",
        help = "Allowed tool names: comma-separated list, or all/none. Use none to disable all tools. Defaults: oneshot=none, repl=all"
    )]
    allowed_tools: Option<String>,

    #[arg(long, hide = true)]
    agent: bool,

    #[arg(long = "tool-root", value_name = "path")]
    tool_root: Option<String>,

    #[arg(
        long = "allow-shell-command",
        value_name = "command",
        env = "GGUF_ALLOW_SHELL_COMMANDS",
        value_delimiter = ','
    )]
    allow_shell_commands: Vec<String>,

    #[arg(
        long = "max-tool-calls",
        value_parser = parse_positive_usize,
        default_value_t = 256
    )]
    max_tool_calls: usize,

    #[arg(long)]
    profiling: bool,

    #[arg(long = "show-tokens")]
    show_tokens: bool,

    #[arg(long = "show-timings")]
    show_timings: bool,

    #[arg(
        long,
        value_parser = parse_think_mode,
        default_value = "yes",
        help = "Control thinking output for reasoning models: yes (show), no (disable), hidden (suppress)"
    )]
    think: crate::engine::types::ThinkMode,

    #[arg(
        long = "think-cap",
        default_value_t = 0,
        help = "Cap hidden-mode reasoning tokens per generation; 0 = vendor-derived default"
    )]
    think_cap: usize,

    #[arg(long)]
    debug: bool,

    #[arg(
        long = "kv-cache-mode",
        hide = true,
        env = "GGUF_KV_CACHE_MODE",
        value_parser = parse_kv_cache_mode
    )]
    kv_cache_mode: Option<CliKvCacheMode>,

    #[arg(
        long = "par-matmul-min-rows",
        hide = true,
        env = "GGUF_PAR_MATMUL_MIN_ROWS",
        value_parser = parse_positive_usize
    )]
    par_matmul_min_rows: Option<usize>,

    #[arg(
        long = "par-matmul-chunk-rows",
        hide = true,
        env = "GGUF_PAR_MATMUL_CHUNK_ROWS",
        value_parser = parse_positive_usize
    )]
    par_matmul_chunk_rows: Option<usize>,

    #[arg(
        long = "par-attn-min-heads",
        hide = true,
        env = "GGUF_PAR_ATTN_MIN_HEADS",
        value_parser = parse_positive_usize
    )]
    par_attn_min_heads: Option<usize>,

    #[arg(
        long = "par-qwen3next-min-heads",
        hide = true,
        env = "GGUF_PAR_QWEN3NEXT_MIN_HEADS",
        value_parser = parse_positive_usize
    )]
    par_qwen3next_min_heads: Option<usize>,

    #[cfg(target_arch = "aarch64")]
    #[arg(
        long = "aarch64-matmul-prefetch-rows",
        hide = true,
        env = "GGUF_AARCH64_MATMUL_PREFETCH_ROWS",
        value_parser = parse_nonnegative_usize
    )]
    aarch64_matmul_prefetch_rows: Option<usize>,

    #[cfg(target_arch = "aarch64")]
    #[arg(
        long = "aarch64-dotprod-q8",
        hide = true,
        env = "GGUF_AARCH64_DOTPROD_Q8",
        value_parser = parse_boolish
    )]
    aarch64_dotprod_q8: Option<bool>,

    #[cfg(target_arch = "aarch64")]
    #[arg(
        long = "aarch64-qk-mr4",
        hide = true,
        env = "GGUF_AARCH64_QK_MR4",
        value_parser = parse_boolish
    )]
    aarch64_qk_mr4: Option<bool>,

    #[cfg(target_arch = "aarch64")]
    #[arg(
        long = "aarch64-i8mm",
        hide = true,
        env = "GGUF_AARCH64_I8MM",
        value_parser = parse_boolish
    )]
    aarch64_i8mm: Option<bool>,

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-avx2",
        hide = true,
        env = "GGUF_X86_AVX2",
        value_parser = parse_boolish
    )]
    x86_avx2: Option<bool>,

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-f16c",
        hide = true,
        env = "GGUF_X86_F16C",
        value_parser = parse_boolish
    )]
    x86_f16c: Option<bool>,

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-qk-mr4",
        hide = true,
        env = "GGUF_X86_QK_MR4",
        value_parser = parse_boolish
    )]
    x86_qk_mr4: Option<bool>,

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-avxvnni",
        hide = true,
        env = "GGUF_X86_AVXVNNI",
        value_parser = parse_boolish
    )]
    x86_avxvnni: Option<bool>,

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-avx512vnni-q8",
        hide = true,
        env = "GGUF_X86_AVX512VNNI_Q8",
        value_parser = parse_boolish
    )]
    x86_avx512vnni_q8: Option<bool>,

    #[arg(
        long = "layer-debug",
        hide = true,
        env = "GGUF_LAYER_DEBUG",
        value_parser = parse_boolish
    )]
    layer_debug: Option<bool>,

    #[arg(long = "layer-debug-pos", hide = true, env = "GGUF_LAYER_DEBUG_POS")]
    layer_debug_pos: Option<usize>,

    // --- RAG flags ---
    #[arg(
        long = "rag-encoder",
        value_name = "encoder.gguf",
        help = "Path to embedding sidecar GGUF used for RAG retrieval"
    )]
    rag_encoder: Option<String>,

    #[arg(
        long = "rag-index",
        value_name = "path",
        help = "Path to a pre-built .ragidx file to load or save"
    )]
    rag_index: Option<String>,

    #[arg(
        long = "rag-source",
        value_name = "dir",
        help = "Wiki source directory; used to build the index when --rag-index is missing or --rag-build is set"
    )]
    rag_source: Option<String>,

    #[arg(
        long = "rag-top-k",
        default_value_t = 5,
        help = "Number of chunks to inject per turn"
    )]
    rag_top_k: usize,

    #[arg(
        long = "rag-max-chars-per-chunk",
        default_value_t = 1800,
        help = "Soft character limit per indexed chunk"
    )]
    rag_max_chars_per_chunk: usize,

    #[arg(
        long = "rag-max-tokens-per-chunk",
        default_value_t = 0,
        help = "Optional token cap per indexed chunk after tokenization (0 disables the cap)"
    )]
    rag_max_tokens_per_chunk: usize,

    #[arg(
        long = "rag-build",
        help = "Build the RAG index from --rag-source, save to --rag-index, then exit"
    )]
    rag_build: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct CliOptions {
    pub(crate) model: String,
    pub(crate) prompt: String,
    pub(crate) render_prefill_cache: Option<String>,
    pub(crate) prefill_cache: Option<String>,
    pub(crate) images: Vec<String>,
    pub(crate) videos: Vec<String>,
    pub(crate) audios: Vec<String>,
    pub(crate) audio_language: Option<String>,
    pub(crate) audio_batch: Option<String>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) top_p: Option<f32>,
    pub(crate) seed: Option<u64>,
    pub(crate) repeat_penalty: f32,
    pub(crate) repeat_last_n: usize,
    pub(crate) max_tokens: usize,
    pub(crate) context_size: usize,
    pub(crate) threads: Option<usize>,
    pub(crate) system_prompt: String,
    pub(crate) mode: CliOperationMode,
    pub(crate) tools_enabled: bool,
    pub(crate) tool_root: Option<String>,
    pub(crate) tool_enablement: AgentToolEnablement,
    pub(crate) allow_shell_commands: Vec<String>,
    pub(crate) shell_command_description_specs: Vec<ShellCommandDescriptionSpec>,
    pub(crate) tool_prompt_specs: Vec<ToolPromptSpec>,
    pub(crate) max_tool_calls: usize,
    pub(crate) show_features: bool,
    pub(crate) profiling: bool,
    pub(crate) show_tokens: bool,
    pub(crate) show_timings: bool,
    pub(crate) think_mode: crate::engine::types::ThinkMode,
    pub(crate) hidden_think_token_cap: usize,
    pub(crate) debug: bool,
    pub(crate) kv_cache_mode: Option<CliKvCacheMode>,
    pub(crate) par_matmul_min_rows: Option<usize>,
    pub(crate) par_matmul_chunk_rows: Option<usize>,
    pub(crate) par_attn_min_heads: Option<usize>,
    pub(crate) par_qwen3next_min_heads: Option<usize>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_matmul_prefetch_rows: Option<usize>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_dotprod_q8: Option<bool>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_qk_mr4: Option<bool>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_i8mm: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avx2: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_f16c: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_qk_mr4: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avxvnni: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avx512vnni_q8: Option<bool>,
    pub(crate) layer_debug: Option<bool>,
    pub(crate) layer_debug_pos: Option<usize>,
    pub(crate) rag_encoder: Option<String>,
    pub(crate) rag_index: Option<String>,
    pub(crate) rag_source: Option<String>,
    pub(crate) rag_top_k: usize,
    pub(crate) rag_max_chars_per_chunk: usize,
    pub(crate) rag_max_tokens_per_chunk: usize,
    pub(crate) rag_build: bool,
}

impl CliOptions {
    pub(crate) fn parse() -> Result<Self, String> {
        let cli = Cli::try_parse().map_err(|e| e.to_string())?;
        let mode = cli.mode;
        let mut allowed_tools = match cli.allowed_tools.as_deref() {
            Some(raw) => parse_allowed_tools(raw)?,
            None => {
                if matches!(mode, CliOperationMode::Repl) {
                    tools::all_tool_name_set()
                } else {
                    BTreeSet::new()
                }
            }
        };
        if cli.agent {
            if cli.allowed_tools.is_none() {
                allowed_tools = tools::all_tool_name_set();
            } else if allowed_tools.is_empty() {
                return Err(
                    "conflicting flags: --agent cannot be combined with --allowed-tools=none"
                        .to_string(),
                );
            }
        }
        let requested_tools_enabled = !allowed_tools.is_empty();
        if !cli.show_features
            && !cli.rag_build
            && cli.render_prefill_cache.is_none()
            && matches!(mode, CliOperationMode::Oneshot)
            && cli.prompt.trim().is_empty()
            && cli.audios.is_empty()
            && cli.audio_batch.is_none()
        {
            return Err("`--prompt` is required in oneshot mode".to_string());
        }
        validate_audio_options(&cli, mode, requested_tools_enabled)?;
        let tool_prompt_specs = default_tool_prompt_specs();
        let loaded_shell = if requested_tools_enabled {
            load_shell_config_from_config()?
        } else {
            LoadedShellConfig::default()
        };
        let mut tool_enablement = if requested_tools_enabled {
            load_tool_enablement_from_config()?
        } else {
            AgentToolEnablement::disabled()
        };
        tool_enablement.apply_allowlist(&allowed_tools);
        let tools_enabled = tool_enablement.has_any_enabled();
        let LoadedShellConfig {
            allowed_commands: mut allow_shell_commands,
            description_specs: shell_command_description_specs,
        } = loaded_shell;
        if requested_tools_enabled {
            allow_shell_commands.extend(cli.allow_shell_commands);
        }
        let allow_shell_commands = normalize_shell_command_values(allow_shell_commands);

        Ok(Self {
            model: cli.model,
            prompt: cli.prompt,
            render_prefill_cache: cli.render_prefill_cache,
            prefill_cache: cli.prefill_cache,
            images: cli.images,
            videos: cli.videos,
            audios: cli.audios,
            audio_language: cli.audio_language,
            audio_batch: cli.audio_batch,
            temperature: cli.temperature,
            top_k: cli.top_k,
            top_p: cli.top_p,
            seed: cli.seed,
            repeat_penalty: cli.repeat_penalty,
            repeat_last_n: cli.repeat_last_n,
            max_tokens: cli.max_tokens,
            context_size: cli.context_size,
            threads: cli.threads,
            system_prompt: cli.system_prompt,
            mode,
            tools_enabled,
            tool_root: cli.tool_root,
            tool_enablement,
            allow_shell_commands,
            shell_command_description_specs,
            tool_prompt_specs,
            max_tool_calls: cli.max_tool_calls,
            show_features: cli.show_features,
            profiling: cli.profiling,
            show_tokens: cli.show_tokens,
            show_timings: cli.show_timings,
            think_mode: cli.think,
            hidden_think_token_cap: cli.think_cap,
            debug: cli.debug,
            kv_cache_mode: cli.kv_cache_mode,
            par_matmul_min_rows: cli.par_matmul_min_rows,
            par_matmul_chunk_rows: cli.par_matmul_chunk_rows,
            par_attn_min_heads: cli.par_attn_min_heads,
            par_qwen3next_min_heads: cli.par_qwen3next_min_heads,
            #[cfg(target_arch = "aarch64")]
            aarch64_matmul_prefetch_rows: cli.aarch64_matmul_prefetch_rows,
            #[cfg(target_arch = "aarch64")]
            aarch64_dotprod_q8: cli.aarch64_dotprod_q8,
            #[cfg(target_arch = "aarch64")]
            aarch64_qk_mr4: cli.aarch64_qk_mr4,
            #[cfg(target_arch = "aarch64")]
            aarch64_i8mm: cli.aarch64_i8mm,
            #[cfg(target_arch = "x86_64")]
            x86_avx2: cli.x86_avx2,
            #[cfg(target_arch = "x86_64")]
            x86_f16c: cli.x86_f16c,
            #[cfg(target_arch = "x86_64")]
            x86_qk_mr4: cli.x86_qk_mr4,
            #[cfg(target_arch = "x86_64")]
            x86_avxvnni: cli.x86_avxvnni,
            #[cfg(target_arch = "x86_64")]
            x86_avx512vnni_q8: cli.x86_avx512vnni_q8,
            layer_debug: cli.layer_debug,
            layer_debug_pos: cli.layer_debug_pos,
            rag_encoder: cli.rag_encoder,
            rag_index: cli.rag_index,
            rag_source: cli.rag_source,
            rag_top_k: cli.rag_top_k,
            rag_max_chars_per_chunk: cli.rag_max_chars_per_chunk,
            rag_max_tokens_per_chunk: cli.rag_max_tokens_per_chunk,
            rag_build: cli.rag_build,
        })
    }
}

fn validate_audio_options(
    cli: &Cli,
    mode: CliOperationMode,
    requested_tools_enabled: bool,
) -> Result<(), String> {
    if cli.audio_language.is_some() && cli.audios.is_empty() && cli.audio_batch.is_none() {
        return Err("`--audio-language` requires `--audio <path>`".to_string());
    }
    if cli.audio_batch.is_none() {
        return Ok(());
    }
    if !matches!(mode, CliOperationMode::Oneshot) {
        return Err("`--audio-batch` is supported only in oneshot mode".to_string());
    }
    if requested_tools_enabled {
        return Err("`--audio-batch` cannot be combined with tools mode".to_string());
    }
    if !cli.audios.is_empty() || !cli.images.is_empty() || !cli.videos.is_empty() {
        return Err(
            "`--audio-batch` cannot be combined with `--audio`, `--image`, or `--video`"
                .to_string(),
        );
    }
    if cli.audio_language.is_some() || !cli.prompt.trim().is_empty() {
        return Err(
            "batch language and context belong in each manifest record; do not combine `--audio-batch` with `--audio-language` or `--prompt`"
                .to_string(),
        );
    }
    if cli.render_prefill_cache.is_some() || cli.prefill_cache.is_some() || cli.rag_build {
        return Err(
            "`--audio-batch` cannot be combined with prefill-cache or RAG build modes".to_string(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn help_mentions_allowed_tools_none() {
        let mut cmd = Cli::command();
        let mut help = Vec::new();
        cmd.write_long_help(&mut help)
            .expect("failed to render long help");
        let help_text = String::from_utf8(help).expect("help output was not valid utf-8");
        assert!(
            help_text.contains("--allowed-tools <list>"),
            "expected --allowed-tools to be present in help output"
        );
        assert!(
            help_text.contains("Use none to disable all tools"),
            "expected --allowed-tools help to mention that none disables all tools"
        );
    }

    #[test]
    fn audio_batch_does_not_require_a_prompt() {
        let cli = Cli::try_parse_from([
            "gguf-runner",
            "--model",
            "model.gguf",
            "--audio-batch",
            "batch.jsonl",
        ])
        .expect("parse batch flags");
        assert!(cli.prompt.is_empty());
        assert!(
            validate_audio_options(&cli, cli.mode, false).is_ok(),
            "batch-only CLI should be valid"
        );
    }

    #[test]
    fn audio_batch_rejects_per_invocation_audio_context() {
        let cli = Cli::try_parse_from([
            "gguf-runner",
            "--model",
            "model.gguf",
            "--audio-batch",
            "batch.jsonl",
            "--prompt",
            "global context",
        ])
        .expect("parse batch flags");
        let error = validate_audio_options(&cli, cli.mode, false).expect_err("conflicting prompt");
        assert!(error.contains("each manifest record"));
    }

    #[test]
    fn parse_allowed_tools_none_disables_all_tools() {
        let parsed = parse_allowed_tools("none").expect("failed to parse none");
        assert!(parsed.is_empty(), "none should disable all tools");
    }

    #[test]
    fn parse_kv_cache_mode_accepts_only_q8_and_turbo() {
        assert_eq!(parse_kv_cache_mode("q8"), Ok(CliKvCacheMode::Q8));
        assert_eq!(parse_kv_cache_mode("turbo"), Ok(CliKvCacheMode::Turbo));
        assert_eq!(parse_kv_cache_mode("tq"), Ok(CliKvCacheMode::Turbo));
        assert!(parse_kv_cache_mode("auto").is_err());
        assert!(parse_kv_cache_mode("q4").is_err());
    }
}
