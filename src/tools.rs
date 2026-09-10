// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.

use crate::cli::AgentToolEnablement;
use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{fmt, marker::PhantomData};

const MAX_READ_BYTES: usize = 256 * 1024;
const MAX_WRITE_BYTES: usize = 256 * 1024;
const MAX_LIST_ENTRIES: usize = 200;
const MAX_SHELL_ARGS: usize = 64;
const MAX_SHELL_ARG_BYTES: usize = 4096;
const MAX_SHELL_OUTPUT_BYTES: usize = 128 * 1024;

#[derive(Clone, Debug)]
struct FlexiblePath(String);

impl FlexiblePath {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for FlexiblePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FlexiblePathVisitor {
            marker: PhantomData<fn() -> FlexiblePath>,
        }

        impl<'de> Visitor<'de> for FlexiblePathVisitor {
            type Value = FlexiblePath;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string path or single-item array containing a string path")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(FlexiblePath(v.to_string()))
            }

            fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(FlexiblePath(v))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let first: Option<String> = seq.next_element()?;
                let Some(path) = first else {
                    return Err(de::Error::invalid_length(0, &self));
                };
                if seq.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom(
                        "path array must contain exactly one string element",
                    ));
                }
                Ok(FlexiblePath(path))
            }
        }

        deserializer.deserialize_any(FlexiblePathVisitor {
            marker: PhantomData,
        })
    }
}

pub(crate) const READ_FILE: &str = "read_file";
pub(crate) const LIST_DIR: &str = "list_dir";
pub(crate) const WRITE_FILE: &str = "write_file";
pub(crate) const MKDIR: &str = "mkdir";
pub(crate) const RMDIR: &str = "rmdir";
pub(crate) const SHELL_LIST_ALLOWED: &str = "shell_list_allowed";
pub(crate) const SHELL_EXEC: &str = "shell_exec";
pub(crate) const SHELL_REQUEST_ALLOWED: &str = "shell_request_allowed";
/// RAG knowledge search — handled directly in the agent loop, not via ToolExecutor.
pub(crate) const SEARCH_KNOWLEDGE: &str = "search_knowledge";

pub(crate) const ALL_TOOL_NAMES: [&str; 8] = [
    READ_FILE,
    LIST_DIR,
    WRITE_FILE,
    MKDIR,
    RMDIR,
    SHELL_LIST_ALLOWED,
    SHELL_EXEC,
    SHELL_REQUEST_ALLOWED,
];

pub(crate) fn all_tool_name_set() -> BTreeSet<String> {
    ALL_TOOL_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

pub(crate) fn is_valid_tool_name(name: &str) -> bool {
    ALL_TOOL_NAMES.contains(&name)
}

pub(crate) struct ToolExecutor {
    root: PathBuf,
    tool_enablement: AgentToolEnablement,
    allow_shell_commands: Vec<String>,
}

impl ToolExecutor {
    pub(crate) fn new(
        tool_root: Option<&str>,
        tool_enablement: AgentToolEnablement,
        allow_shell_commands: &[String],
    ) -> Result<Self, String> {
        let root = match tool_root {
            Some(raw) => PathBuf::from(raw),
            None => std::env::current_dir()
                .map_err(|e| format!("cannot read current directory: {e}"))?,
        };
        let root = root
            .canonicalize()
            .map_err(|e| format!("cannot canonicalize tool root '{}': {e}", root.display()))?;
        if !root.is_dir() {
            return Err(format!("tool root is not a directory: {}", root.display()));
        }
        let mut uniq = BTreeSet::new();
        for raw in allow_shell_commands {
            let normalized = normalize_shell_command(raw)
                .map_err(|e| format!("invalid shell command '{raw}': {e}"))?;
            uniq.insert(normalized);
        }
        Ok(Self {
            root,
            tool_enablement,
            allow_shell_commands: uniq.into_iter().collect(),
        })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn write_file_enabled(&self) -> bool {
        self.tool_enablement.write_file
    }

    pub(crate) fn shell_exec_enabled(&self) -> bool {
        self.tool_enablement.shell_exec
    }

    pub(crate) fn shell_list_allowed_enabled(&self) -> bool {
        self.tool_enablement.shell_list_allowed
    }

    pub(crate) fn shell_request_allowed_enabled(&self) -> bool {
        self.tool_enablement.shell_request_allowed
    }

    pub(crate) fn has_any_filesystem_tool(&self) -> bool {
        self.tool_enablement.read_file
            || self.tool_enablement.list_dir
            || self.write_file_enabled()
            || self.tool_enablement.mkdir
            || self.tool_enablement.rmdir
    }

    pub(crate) fn enabled_tool_names(&self) -> Vec<&'static str> {
        self.tool_enablement.enabled_tool_names()
    }

    pub(crate) fn shell_allowed_commands(&self) -> &[String] {
        &self.allow_shell_commands
    }

    pub(crate) fn execute(&self, tool: &str, args: &Value) -> Result<Value, String> {
        match tool {
            READ_FILE => {
                if !self.tool_enablement.read_file {
                    return Err(
                        "tool 'read_file' is disabled by config ([tools].read_file=false)"
                            .to_string(),
                    );
                }
                self.read_file(args)
            }
            WRITE_FILE => {
                if !self.tool_enablement.write_file {
                    return Err(
                        "tool 'write_file' is disabled by config ([tools].write_file=false)"
                            .to_string(),
                    );
                }
                self.write_file(args)
            }
            LIST_DIR => {
                if !self.tool_enablement.list_dir {
                    return Err(
                        "tool 'list_dir' is disabled by config ([tools].list_dir=false)"
                            .to_string(),
                    );
                }
                self.list_dir(args)
            }
            MKDIR => {
                if !self.tool_enablement.mkdir {
                    return Err(
                        "tool 'mkdir' is disabled by config ([tools].mkdir=false)".to_string()
                    );
                }
                self.mkdir(args)
            }
            RMDIR => {
                if !self.tool_enablement.rmdir {
                    return Err(
                        "tool 'rmdir' is disabled by config ([tools].rmdir=false)".to_string()
                    );
                }
                self.rmdir(args)
            }
            SHELL_LIST_ALLOWED => {
                if !self.tool_enablement.shell_list_allowed {
                    return Err(
                        "tool 'shell_list_allowed' is disabled by config ([tools].shell_list_allowed=false)"
                            .to_string(),
                    );
                }
                Ok(self.shell_list_allowed())
            }
            SHELL_EXEC => {
                if !self.tool_enablement.shell_exec {
                    return Err(
                        "tool 'shell_exec' is disabled by config ([tools].shell_exec=false)"
                            .to_string(),
                    );
                }
                self.shell_exec(args)
            }
            SHELL_REQUEST_ALLOWED => {
                if !self.tool_enablement.shell_request_allowed {
                    return Err("tool 'shell_request_allowed' is disabled by config ([tools].shell_request_allowed=false)".to_string());
                }
                self.shell_request_allowed(args)
            }
            _ => Err(format!("unknown tool '{tool}'")),
        }
    }

    fn resolve_existing_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        let joined = self.join_tool_path(raw_path)?;
        let canonical = joined
            .canonicalize()
            .map_err(|e| format!("cannot resolve path '{}': {e}", joined.display()))?;
        if !canonical.starts_with(&self.root) {
            return Err(format!(
                "path '{}' escapes tool root '{}'",
                canonical.display(),
                self.root.display()
            ));
        }
        Ok(canonical)
    }

    fn resolve_write_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        let joined = self.join_tool_path(raw_path)?;
        let parent = joined
            .parent()
            .ok_or_else(|| format!("cannot determine parent for '{}'", joined.display()))?;
        let canonical_parent = parent.canonicalize().map_err(|e| {
            format!(
                "cannot resolve parent directory '{}': {e}",
                parent.display()
            )
        })?;
        if !canonical_parent.starts_with(&self.root) {
            return Err(format!(
                "path '{}' escapes tool root '{}'",
                joined.display(),
                self.root.display()
            ));
        }
        let file_name = joined
            .file_name()
            .ok_or_else(|| format!("cannot determine file name for '{}'", joined.display()))?;
        let final_path = canonical_parent.join(file_name);
        if final_path.exists() {
            let md = fs::symlink_metadata(&final_path)
                .map_err(|e| format!("cannot stat '{}': {e}", final_path.display()))?;
            if md.file_type().is_symlink() {
                return Err(format!(
                    "refusing to write through symlink '{}'",
                    final_path.display()
                ));
            }
        }
        Ok(final_path)
    }

    fn resolve_dir_create_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        let joined = self.join_tool_path(raw_path)?;
        let mut nearest_existing = joined.clone();
        while !nearest_existing.exists() {
            nearest_existing = nearest_existing
                .parent()
                .ok_or_else(|| format!("cannot resolve parent for '{}'", joined.display()))?
                .to_path_buf();
        }
        let canonical_existing = nearest_existing.canonicalize().map_err(|e| {
            format!(
                "cannot resolve parent directory '{}': {e}",
                nearest_existing.display()
            )
        })?;
        if !canonical_existing.starts_with(&self.root) {
            return Err(format!(
                "path '{}' escapes tool root '{}'",
                joined.display(),
                self.root.display()
            ));
        }
        let suffix = joined
            .strip_prefix(&nearest_existing)
            .map_err(|e| format!("cannot normalize path '{}': {e}", joined.display()))?;
        let final_path = if suffix.as_os_str().is_empty() {
            canonical_existing
        } else {
            canonical_existing.join(suffix)
        };
        if final_path.exists() {
            let md = fs::symlink_metadata(&final_path)
                .map_err(|e| format!("cannot stat '{}': {e}", final_path.display()))?;
            if md.file_type().is_symlink() {
                return Err(format!(
                    "refusing to create through symlink '{}'",
                    final_path.display()
                ));
            }
        }
        Ok(final_path)
    }

    fn join_tool_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        if raw_path.is_empty() {
            return Err("path cannot be empty".to_string());
        }
        let path = Path::new(raw_path);
        Ok(if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        })
    }

    fn read_file(&self, args: &Value) -> Result<Value, String> {
        let args: ReadFileArgs = serde_json::from_value(args.clone())
            .map_err(|e| format!("invalid read_file args: {e}"))?;
        let path = self.resolve_existing_path(args.path.as_str())?;
        if !path.is_file() {
            return Err(format!("not a regular file: {}", path.display()));
        }
        let read_limit = args.max_bytes.unwrap_or(MAX_READ_BYTES).min(MAX_READ_BYTES);
        let mut file =
            fs::File::open(&path).map_err(|e| format!("cannot open '{}': {e}", path.display()))?;
        let mut buf = Vec::new();
        std::io::Read::by_ref(&mut file)
            .take((read_limit + 1) as u64)
            .read_to_end(&mut buf)
            .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
        let truncated = buf.len() > read_limit;
        if truncated {
            buf.truncate(read_limit);
        }
        let content = String::from_utf8_lossy(&buf).to_string();
        Ok(json!({
            "ok": true,
            "tool": "read_file",
            "path": path.display().to_string(),
            "bytes": buf.len(),
            "truncated": truncated,
            "content": content
        }))
    }

    fn write_file(&self, args: &Value) -> Result<Value, String> {
        let args: WriteFileArgs = serde_json::from_value(args.clone())
            .map_err(|e| format!("invalid write_file args: {e}"))?;
        let path = self.resolve_write_path(args.path.as_str())?;
        let bytes = args.content.as_bytes();
        if bytes.len() > MAX_WRITE_BYTES {
            return Err(format!(
                "write_file content too large: {} bytes > {} bytes limit",
                bytes.len(),
                MAX_WRITE_BYTES
            ));
        }

        let mut opts = OpenOptions::new();
        opts.create(true).write(true);
        if args.append.unwrap_or(false) {
            opts.append(true);
        } else {
            opts.truncate(true);
        }

        let mut file = opts
            .open(&path)
            .map_err(|e| format!("cannot open '{}' for write: {e}", path.display()))?;
        file.write_all(bytes)
            .map_err(|e| format!("cannot write '{}': {e}", path.display()))?;
        file.flush()
            .map_err(|e| format!("cannot flush '{}': {e}", path.display()))?;
        Ok(json!({
            "ok": true,
            "tool": "write_file",
            "path": path.display().to_string(),
            "bytes_written": bytes.len(),
            "append": args.append.unwrap_or(false)
        }))
    }

    fn list_dir(&self, args: &Value) -> Result<Value, String> {
        let args: ListDirArgs = serde_json::from_value(args.clone())
            .map_err(|e| format!("invalid list_dir args: {e}"))?;
        let path = args.path.as_ref().map(FlexiblePath::as_str).unwrap_or(".");
        let path = self.resolve_existing_path(path)?;
        if !path.is_dir() {
            return Err(format!("not a directory: {}", path.display()));
        }
        let mut entries = Vec::new();
        for entry in
            fs::read_dir(&path).map_err(|e| format!("cannot list '{}': {e}", path.display()))?
        {
            let entry = entry.map_err(|e| format!("cannot read directory entry: {e}"))?;
            let ft = entry
                .file_type()
                .map_err(|e| format!("cannot read file type: {e}"))?;
            let kind = if ft.is_dir() {
                "dir"
            } else if ft.is_file() {
                "file"
            } else if ft.is_symlink() {
                "symlink"
            } else {
                "other"
            };
            entries.push(json!({
                "name": entry.file_name().to_string_lossy().to_string(),
                "kind": kind
            }));
            if entries.len()
                >= args
                    .max_entries
                    .unwrap_or(MAX_LIST_ENTRIES)
                    .min(MAX_LIST_ENTRIES)
            {
                break;
            }
        }
        entries.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        Ok(json!({
            "ok": true,
            "tool": "list_dir",
            "path": path.display().to_string(),
            "entries": entries
        }))
    }

    fn mkdir(&self, args: &Value) -> Result<Value, String> {
        let args: MkdirArgs =
            serde_json::from_value(args.clone()).map_err(|e| format!("invalid mkdir args: {e}"))?;
        let path = self.resolve_dir_create_path(args.path.as_str())?;
        let existed = path.exists();
        if existed {
            if !path.is_dir() {
                return Err(format!(
                    "cannot create directory because path exists and is not a directory: {}",
                    path.display()
                ));
            }
        } else {
            fs::create_dir_all(&path)
                .map_err(|e| format!("cannot create directory '{}': {e}", path.display()))?;
        }
        Ok(json!({
            "ok": true,
            "tool": "mkdir",
            "path": path.display().to_string(),
            "recursive": true,
            "created": !existed
        }))
    }

    fn rmdir(&self, args: &Value) -> Result<Value, String> {
        let args: RmdirArgs =
            serde_json::from_value(args.clone()).map_err(|e| format!("invalid rmdir args: {e}"))?;
        let path = self.resolve_existing_path(args.path.as_str())?;
        if path == self.root {
            return Err("refusing to remove tool root".to_string());
        }
        if !path.is_dir() {
            return Err(format!("not a directory: {}", path.display()));
        }
        let joined = self.join_tool_path(args.path.as_str())?;
        let joined_md = fs::symlink_metadata(&joined)
            .map_err(|e| format!("cannot stat '{}': {e}", joined.display()))?;
        if joined_md.file_type().is_symlink() {
            return Err(format!("refusing to remove symlink '{}'", joined.display()));
        }
        fs::remove_dir_all(&path)
            .map_err(|e| format!("cannot remove directory '{}': {e}", path.display()))?;
        Ok(json!({
            "ok": true,
            "tool": "rmdir",
            "path": path.display().to_string(),
            "recursive": true
        }))
    }

    fn is_shell_command_allowed(&self, command: &str) -> bool {
        self.allow_shell_commands
            .binary_search_by(|allowed| allowed.as_str().cmp(command))
            .is_ok()
    }

    fn shell_list_allowed(&self) -> Value {
        json!({
            "ok": true,
            "tool": "shell_list_allowed",
            "shell_allowed_commands": self.allow_shell_commands.clone(),
            "internal_tool_status": {
                "read_file": self.tool_enablement.read_file,
                "list_dir": self.tool_enablement.list_dir,
                "write_file": self.write_file_enabled(),
                "mkdir": self.tool_enablement.mkdir,
                "rmdir": self.tool_enablement.rmdir,
                "shell_list_allowed": self.tool_enablement.shell_list_allowed,
                "shell_exec": self.tool_enablement.shell_exec,
                "shell_request_allowed": self.tool_enablement.shell_request_allowed
            }
        })
    }

    fn shell_exec(&self, args: &Value) -> Result<Value, String> {
        let args: RunShellArgs = serde_json::from_value(args.clone())
            .map_err(|e| {
                format!(
                    "invalid shell_exec args: {e}. expected object like {{\"command\":\"<allowed>\",\"args\":[...],\"cwd\":\".\",\"max_output_bytes\":131072}} (aliases accepted: cmd, argv, workdir)"
                )
            })?;
        let (command, argv) = normalize_shell_exec_invocation(&args.command, args.args)?;
        if !self.is_shell_command_allowed(&command) {
            let allowed = if self.allow_shell_commands.is_empty() {
                "<none>".to_string()
            } else {
                self.allow_shell_commands.join(", ")
            };
            return Err(format!(
                "command '{}' is not allowed. allowed commands: {}",
                command, allowed
            ));
        }

        if argv.len() > MAX_SHELL_ARGS {
            return Err(format!(
                "too many shell_exec args: {} > {}",
                argv.len(),
                MAX_SHELL_ARGS
            ));
        }
        for (idx, arg) in argv.iter().enumerate() {
            if arg.as_bytes().contains(&0) {
                return Err(format!("shell_exec arg {} contains NUL byte", idx));
            }
            if arg.len() > MAX_SHELL_ARG_BYTES {
                return Err(format!(
                    "shell_exec arg {} too large: {} bytes > {} bytes limit",
                    idx,
                    arg.len(),
                    MAX_SHELL_ARG_BYTES
                ));
            }
        }

        let cwd = if let Some(raw_cwd) = args.cwd {
            let dir = self.resolve_existing_path(&raw_cwd)?;
            if !dir.is_dir() {
                return Err(format!(
                    "shell_exec cwd is not a directory: {}",
                    dir.display()
                ));
            }
            dir
        } else {
            self.root.clone()
        };
        let output_limit = args
            .max_output_bytes
            .unwrap_or(MAX_SHELL_OUTPUT_BYTES)
            .min(MAX_SHELL_OUTPUT_BYTES);

        if command == "cwd" {
            if !argv.is_empty() {
                return Err("shell_exec command 'cwd' does not accept args".to_string());
            }
            let stdout_raw = format!("{}\n", cwd.display()).into_bytes();
            let (stdout, stdout_truncated) = truncate_output(&stdout_raw, output_limit);
            return Ok(json!({
                "ok": true,
                "tool": "shell_exec",
                "command": command,
                "args": argv,
                "cwd": cwd.display().to_string(),
                "exit_code": 0,
                "stdout_bytes": stdout_raw.len(),
                "stderr_bytes": 0,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": false,
                "stdout": stdout,
                "stderr": ""
            }));
        }

        let output = Command::new(&command)
            .args(&argv)
            .current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| format!("shell_exec failed for '{}': {e}", command))?;

        let (stdout, stdout_truncated) = truncate_output(&output.stdout, output_limit);
        let (stderr, stderr_truncated) = truncate_output(&output.stderr, output_limit);
        Ok(json!({
            "ok": output.status.success(),
            "tool": "shell_exec",
            "command": command,
            "args": argv,
            "cwd": cwd.display().to_string(),
            "exit_code": output.status.code(),
            "stdout_bytes": output.stdout.len(),
            "stderr_bytes": output.stderr.len(),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "stdout": stdout,
            "stderr": stderr
        }))
    }

    fn shell_request_allowed(&self, args: &Value) -> Result<Value, String> {
        let args: RequestShellAllowedArgs = serde_json::from_value(args.clone())
            .map_err(|e| format!("invalid shell_request_allowed args: {e}"))?;
        let command = normalize_shell_command(&args.command)
            .map_err(|e| format!("invalid command request: {e}"))?;
        let already_allowed = self.is_shell_command_allowed(&command);
        let status = if already_allowed {
            "already_allowed"
        } else {
            "needs_user_approval"
        };
        let hint = if already_allowed {
            "Command is already allowed. Use shell_exec to execute it.".to_string()
        } else {
            format!(
                "Ask the operator to add --allow-shell-command {} (or GGUF_ALLOW_SHELL_COMMANDS).",
                command
            )
        };
        Ok(json!({
            "ok": true,
            "tool": "shell_request_allowed",
            "command": command,
            "reason": args.reason,
            "already_allowed": already_allowed,
            "status": status,
            "hint": hint
        }))
    }
}

fn normalize_shell_command(raw: &str) -> Result<String, String> {
    let command = raw.trim();
    if command.is_empty() {
        return Err("command cannot be empty".to_string());
    }
    if command.as_bytes().contains(&0) {
        return Err("command contains NUL byte".to_string());
    }
    if command
        .chars()
        .any(|c| c.is_ascii_whitespace() || c == '/' || c == '\\')
    {
        return Err(
            "command must be a bare executable name (no whitespace or path separators)".to_string(),
        );
    }
    Ok(command.to_string())
}

fn normalize_shell_exec_invocation(
    command_raw: &str,
    args: Option<Vec<String>>,
) -> Result<(String, Vec<String>), String> {
    let command_trimmed = command_raw.trim();
    let mut argv = args.unwrap_or_default();
    if argv.is_empty() && command_trimmed.chars().any(|c| c.is_ascii_whitespace()) {
        let mut parts = command_trimmed.split_whitespace();
        let head = parts
            .next()
            .ok_or_else(|| "invalid shell_exec command: command cannot be empty".to_string())?;
        let command = normalize_shell_command(head)
            .map_err(|e| format!("invalid shell_exec command: {e}"))?;
        argv.extend(parts.map(ToOwned::to_owned));
        return Ok((command, argv));
    }
    let command = normalize_shell_command(command_trimmed)
        .map_err(|e| format!("invalid shell_exec command: {e}"))?;
    Ok((command, argv))
}

fn truncate_output(bytes: &[u8], limit: usize) -> (String, bool) {
    let truncated = bytes.len() > limit;
    let slice = if truncated { &bytes[..limit] } else { bytes };
    (String::from_utf8_lossy(slice).to_string(), truncated)
}

#[derive(Deserialize)]
struct ReadFileArgs {
    #[serde(alias = "file", alias = "filepath", alias = "filename")]
    path: FlexiblePath,
    max_bytes: Option<usize>,
}

#[derive(Deserialize)]
struct WriteFileArgs {
    #[serde(alias = "file", alias = "filepath", alias = "filename")]
    path: FlexiblePath,
    #[serde(alias = "text", alias = "data")]
    content: String,
    append: Option<bool>,
}

#[derive(Deserialize)]
struct ListDirArgs {
    #[serde(alias = "dir", alias = "directory", alias = "folder")]
    path: Option<FlexiblePath>,
    max_entries: Option<usize>,
}

#[derive(Deserialize)]
struct MkdirArgs {
    #[serde(alias = "dir", alias = "directory", alias = "folder")]
    path: FlexiblePath,
}

#[derive(Deserialize)]
struct RmdirArgs {
    #[serde(alias = "dir", alias = "directory", alias = "folder")]
    path: FlexiblePath,
}

#[derive(Deserialize)]
struct RunShellArgs {
    #[serde(alias = "cmd", alias = "program", alias = "name")]
    command: String,
    #[serde(default, alias = "argv", alias = "arguments")]
    args: Option<Vec<String>>,
    #[serde(default, alias = "workdir", alias = "working_dir", alias = "dir")]
    cwd: Option<String>,
    #[serde(
        default,
        alias = "max_output",
        alias = "max_bytes",
        alias = "output_limit"
    )]
    max_output_bytes: Option<usize>,
}

#[derive(Deserialize)]
struct RequestShellAllowedArgs {
    command: String,
    reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{RunShellArgs, ToolExecutor, normalize_shell_exec_invocation};
    use crate::cli::AgentToolEnablement;
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_tool_root() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "gguf_runner_tools_test_{}_{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&path).expect("create temporary test root");
        path
    }

    #[test]
    fn run_shell_args_accepts_cmd_and_argv_aliases() {
        let raw = serde_json::json!({
            "cmd": "ls",
            "argv": ["-la"],
            "workdir": ".",
            "max_output": 1024
        });
        let parsed: RunShellArgs = serde_json::from_value(raw).expect("valid alias payload");
        assert_eq!(parsed.command, "ls");
        assert_eq!(parsed.args, Some(vec!["-la".to_string()]));
        assert_eq!(parsed.cwd, Some(".".to_string()));
        assert_eq!(parsed.max_output_bytes, Some(1024));
    }

    #[test]
    fn normalize_shell_exec_invocation_splits_command_line_when_args_missing() {
        let (command, args) =
            normalize_shell_exec_invocation("cargo check --release", None).expect("valid split");
        assert_eq!(command, "cargo");
        assert_eq!(args, vec!["check".to_string(), "--release".to_string()]);
    }

    #[test]
    fn mkdir_and_rmdir_work_recursively() {
        let root = make_temp_tool_root();
        let root_s = root.to_string_lossy().to_string();
        let tool_exec = ToolExecutor::new(Some(&root_s), AgentToolEnablement::default(), &[])
            .expect("tool executor");

        let mkdir_result = tool_exec
            .execute("mkdir", &json!({"path":"a/b/c"}))
            .expect("mkdir success");
        assert_eq!(mkdir_result.get("ok").and_then(|v| v.as_bool()), Some(true));
        assert!(root.join("a/b/c").is_dir());

        let rmdir_result = tool_exec
            .execute("rmdir", &json!({"path":"a"}))
            .expect("rmdir success");
        assert_eq!(rmdir_result.get("ok").and_then(|v| v.as_bool()), Some(true));
        assert!(!root.join("a").exists());

        fs::remove_dir_all(&root).expect("cleanup test root");
    }

    #[test]
    fn rmdir_refuses_tool_root() {
        let root = make_temp_tool_root();
        let root_s = root.to_string_lossy().to_string();
        let tool_exec = ToolExecutor::new(Some(&root_s), AgentToolEnablement::default(), &[])
            .expect("tool executor");

        let err = tool_exec
            .execute("rmdir", &json!({"path":"."}))
            .expect_err("rmdir root should fail");
        assert!(err.contains("refusing to remove tool root"));

        fs::remove_dir_all(&root).expect("cleanup test root");
    }

    #[test]
    fn read_file_accepts_single_item_path_array() {
        let root = make_temp_tool_root();
        let root_s = root.to_string_lossy().to_string();
        let file_path = root.join("Cargo.toml");
        fs::write(&file_path, "name = \"demo\"\n").expect("write fixture file");
        let tool_exec = ToolExecutor::new(Some(&root_s), AgentToolEnablement::default(), &[])
            .expect("tool executor");

        let result = tool_exec
            .execute(
                "read_file",
                &json!({
                    "path": [file_path.to_string_lossy().to_string()]
                }),
            )
            .expect("read_file should coerce single-item path array");
        assert_eq!(result.get("ok").and_then(|v| v.as_bool()), Some(true));
        assert!(
            result
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .contains("name = \"demo\"")
        );

        fs::remove_dir_all(&root).expect("cleanup test root");
    }

    #[test]
    fn read_file_accepts_file_alias_for_path() {
        let root = make_temp_tool_root();
        let root_s = root.to_string_lossy().to_string();
        fs::write(root.join("Cargo.toml"), "name = \"demo\"\n").expect("write fixture file");
        let tool_exec = ToolExecutor::new(Some(&root_s), AgentToolEnablement::default(), &[])
            .expect("tool executor");

        let result = tool_exec
            .execute(
                "read_file",
                &json!({
                    "file": "Cargo.toml"
                }),
            )
            .expect("read_file should accept file alias");
        assert_eq!(result.get("ok").and_then(|v| v.as_bool()), Some(true));

        fs::remove_dir_all(&root).expect("cleanup test root");
    }
}
