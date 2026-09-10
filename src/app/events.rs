// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.

use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeLogKind {
    Debug,
    System,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeLog {
    pub(crate) kind: RuntimeLogKind,
    pub(crate) message: String,
}

impl RuntimeLog {
    pub(crate) fn debug(message: impl Into<String>) -> Self {
        Self {
            kind: RuntimeLogKind::Debug,
            message: message.into(),
        }
    }

    pub(crate) fn system(message: impl Into<String>) -> Self {
        Self {
            kind: RuntimeLogKind::System,
            message: message.into(),
        }
    }

    pub(crate) fn error(message: impl Into<String>) -> Self {
        Self {
            kind: RuntimeLogKind::Error,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RunnerStatus {
    Planning {
        turn: usize,
        max_turns: usize,
    },
    Tool {
        turn: usize,
        max_turns: usize,
        tool: String,
    },
    Finalizing {
        turn: usize,
        max_turns: usize,
    },
    Recovering {
        turn: usize,
        max_turns: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum RuntimePhase {
    Prefill,
    Decode,
    Ready,
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeProgress {
    pub(crate) phase: RuntimePhase,
    pub(crate) prefill_tokens: usize,
    pub(crate) decode_tokens: usize,
    pub(crate) hidden_thinking: bool,
    pub(crate) hidden_think_tokens: usize,
    pub(crate) tokens_per_second: Option<f64>,
    pub(crate) context_used: usize,
    pub(crate) context_limit: usize,
}

#[derive(Clone, Debug)]
pub(crate) enum RuntimeEvent {
    Output(String),
    Log(RuntimeLog),
    Status(RunnerStatus),
    Progress(RuntimeProgress),
}

pub(crate) type RuntimeEventCallback = Arc<dyn Fn(RuntimeEvent) + Send + Sync + 'static>;

pub(crate) fn emit_runtime_event(callback: Option<&RuntimeEventCallback>, event: RuntimeEvent) {
    if let Some(callback) = callback {
        callback(event);
    }
}
