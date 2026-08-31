mod app;
mod cli;
mod engine;
mod rag;
mod tools;
mod vendors;

pub use app::embed::{
    AudioTranscriptionResult, EmbeddedRuntime, GenerationStats, Tool,
    build_tool_system_prompt_from_specs,
};
pub use app::speaker::{
    SpeakerAudioQuality, SpeakerDiarizationResult, SpeakerDiarizationSegment, SpeakerEmbedding,
    SpeakerEnrollmentResult, SpeakerIdentificationResult, SpeakerIndexRuntime, SpeakerMatch,
    SpeakerObservationSummary, SpeakerProfileSummary, SpeakerRuntime, SpeakerVerificationResult,
};
