mod app;
mod cli;
mod engine;
mod rag;
mod tools;
mod vendors;

/// CLI entry point, used by the `gguf-runner` binary.
pub use app::run;

pub use app::embed::{
    AudioTranscriptionResult, EmbeddedRuntime, GenerationStats, Tool, ToolCallFormat,
    build_tool_system_prompt_from_specs, detect_tool_call_format, tool_call_format_for_gguf,
};
pub use app::speaker::{
    SpeakerAudioQuality, SpeakerDiarizationResult, SpeakerDiarizationSegment, SpeakerEmbedding,
    SpeakerEnrollmentResult, SpeakerIdentificationResult, SpeakerIndexRuntime, SpeakerMatch,
    SpeakerObservationSummary, SpeakerProfileSummary, SpeakerRuntime, SpeakerVerificationResult,
};
