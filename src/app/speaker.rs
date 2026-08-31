use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::app::speaker_index::{SpeakerIndex, SpeakerObservation, StoredThresholds};
use crate::cli::{CliOptions, CliSpeakerAction, CliSpeakerLearningMode};
use crate::engine::io::{parse_gguf_file, parse_gguf_from_bytes};
use crate::engine::speaker::{
    SpeakerAudioQuality as EngineSpeakerAudioQuality, SpeakerEmbeddingOutput, SpeakerEncoder,
    SpeakerThresholdPolicy, cosine_similarity, normalize_embedding,
};
use crate::vendors;

const CANDIDATE_FORMAT: &str = "gguf-runner-speaker-candidate";
const CANDIDATE_VERSION: u32 = 1;
const EMBEDDING_FORMAT: &str = "gguf-runner-speaker-embedding";
const EMBEDDING_VERSION: u32 = 1;
const MAX_CANDIDATE_LINE_BYTES: usize = 4 * 1024 * 1024;
const MAX_CANDIDATE_RECORDS: usize = 100_000;
const MAX_SPEAKER_JSONL_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SpeakerAudioQuality {
    pub duration_seconds: f32,
    pub rms: f32,
    pub clipping_fraction: f32,
    pub active_fraction: f32,
    pub score: f32,
}

fn apply_identification_learning(
    runtime: &SpeakerRuntime,
    index_path: &Path,
    mode: CliSpeakerLearningMode,
    candidate_path: Option<&Path>,
    embedding: &SpeakerEmbedding,
    identification: &SpeakerIdentificationResult,
) -> Result<(), String> {
    let candidate = runtime.candidate_for(embedding, identification);
    match (mode, candidate) {
        (CliSpeakerLearningMode::Off, _) => Ok(()),
        (CliSpeakerLearningMode::Candidates, Some(candidate)) => {
            let path = candidate_path
                .ok_or_else(|| "internal error: candidate mode has no output path".to_string())?;
            write_candidate(path, &candidate)?;
            eprintln!(
                "speaker learning candidate written: {} ({})",
                candidate.candidate_id,
                path.display()
            );
            Ok(())
        }
        (CliSpeakerLearningMode::Auto, Some(candidate)) => {
            let result = runtime.auto_enroll_candidate(index_path, candidate)?;
            eprintln!(
                "speaker profile auto-refined: {} observation={} count={}",
                result.speaker_id, result.observation_id, result.observation_count
            );
            Ok(())
        }
        (CliSpeakerLearningMode::Candidates | CliSpeakerLearningMode::Auto, None) => {
            eprintln!("speaker profile unchanged: match did not meet the auto-learning gate");
            Ok(())
        }
    }
}

fn apply_diarization_learning(
    runtime: &SpeakerRuntime,
    index_path: &Path,
    mode: CliSpeakerLearningMode,
    candidate_path: Option<&Path>,
    candidates: Vec<SpeakerLearningCandidate>,
) -> Result<(), String> {
    match mode {
        CliSpeakerLearningMode::Off => Ok(()),
        CliSpeakerLearningMode::Candidates => {
            let path = candidate_path
                .ok_or_else(|| "internal error: candidate mode has no output path".to_string())?;
            let count = candidates.len();
            for candidate in candidates {
                write_candidate(path, &candidate)?;
            }
            eprintln!(
                "speaker learning candidates written: {count} ({})",
                path.display()
            );
            Ok(())
        }
        CliSpeakerLearningMode::Auto => {
            let count = runtime.auto_enroll_candidates(index_path, candidates)?;
            eprintln!("speaker meeting profiles auto-refined: {count}");
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SpeakerEmbedding {
    pub format: String,
    pub version: u32,
    pub model_name: String,
    pub model_architecture: String,
    pub model_fingerprint: String,
    pub dimension: usize,
    pub source: String,
    pub quality: SpeakerAudioQuality,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeakerEnrollmentResult {
    pub speaker_id: String,
    pub observation_id: String,
    pub observation_count: usize,
    pub index_path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeakerMatch {
    pub speaker_id: String,
    pub score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeakerVerificationResult {
    pub speaker_id: String,
    pub verified: bool,
    pub score: f32,
    pub threshold: f32,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeakerIdentificationResult {
    pub recognized: bool,
    pub speaker_id: Option<String>,
    pub score: Option<f32>,
    pub threshold: f32,
    pub second_score: Option<f32>,
    pub margin: Option<f32>,
    pub required_margin: f32,
    pub source: String,
    pub matches: Vec<SpeakerMatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeakerProfileSummary {
    pub speaker_id: String,
    pub observation_count: usize,
    pub total_duration_seconds: f32,
    pub observations: Vec<SpeakerObservationSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeakerObservationSummary {
    pub observation_id: String,
    pub source: String,
    pub duration_seconds: f32,
    pub quality_score: f32,
    pub created_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeakerDiarizationSegment {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub speaker_id: String,
    pub recognized: bool,
    pub score: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeakerDiarizationResult {
    pub source: String,
    pub sample_rate: u32,
    pub segments: Vec<SpeakerDiarizationSegment>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ThresholdOverrides {
    pub(crate) verification: Option<f32>,
    pub(crate) identification_margin: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SpeakerLearningCandidate {
    format: String,
    version: u32,
    pub(crate) candidate_id: String,
    pub(crate) accepted: bool,
    pub(crate) speaker_id: String,
    pub(crate) model_fingerprint: String,
    pub(crate) source: String,
    pub(crate) duration_seconds: f32,
    pub(crate) quality_score: f32,
    pub(crate) score: f32,
    pub(crate) margin: Option<f32>,
    pub(crate) vector: Vec<f32>,
}

struct DiarizationOutcome {
    result: SpeakerDiarizationResult,
    candidates: Vec<SpeakerLearningCandidate>,
}

pub struct SpeakerRuntime {
    encoder: SpeakerEncoder,
}

/// A retained speaker profile index for model-free operations on exported embeddings.
pub struct SpeakerIndexRuntime {
    path: PathBuf,
    index: SpeakerIndex,
}

impl SpeakerIndexRuntime {
    /// Load and validate an existing speaker index without loading its embedding model.
    pub fn load(path: &Path) -> Result<Self, String> {
        let index = SpeakerIndex::load(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            index,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.index.model_name
    }

    pub fn model_architecture(&self) -> &str {
        &self.index.model_architecture
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.index.model_fingerprint
    }

    pub fn embedding_dimension(&self) -> usize {
        self.index.embedding_dimension
    }

    /// Verify a claimed identity using a previously exported embedding.
    pub fn verify_embedding(
        &self,
        speaker_id: &str,
        embedding: &SpeakerEmbedding,
    ) -> Result<SpeakerVerificationResult, String> {
        self.validate_embedding(embedding)?;
        let profile = self
            .index
            .speaker(speaker_id)
            .ok_or_else(|| format!("speaker profile not found: {speaker_id}"))?;
        let score = cosine_similarity(&embedding.vector, &profile.centroid)?;
        Ok(SpeakerVerificationResult {
            speaker_id: speaker_id.to_string(),
            verified: score >= self.index.thresholds.verification,
            score,
            threshold: self.index.thresholds.verification,
            source: embedding.source.clone(),
        })
    }

    /// Perform open-set identification using a previously exported embedding.
    pub fn identify_embedding(
        &self,
        embedding: &SpeakerEmbedding,
    ) -> Result<SpeakerIdentificationResult, String> {
        self.validate_embedding(embedding)?;
        identify_vector(&self.index, &embedding.vector, embedding.source.clone())
    }

    /// Add one exported embedding and atomically persist the updated index.
    pub fn enroll_embedding(
        &mut self,
        speaker_id: &str,
        embedding: &SpeakerEmbedding,
        force: bool,
    ) -> Result<SpeakerEnrollmentResult, String> {
        self.validate_embedding(embedding)?;
        let (observation_id, observation_count) = self.index.enroll(
            speaker_id,
            embedding.vector.clone(),
            embedding.source.clone(),
            embedding.quality.duration_seconds,
            embedding.quality.score,
            force,
        )?;
        self.save()?;
        Ok(SpeakerEnrollmentResult {
            speaker_id: speaker_id.to_string(),
            observation_id,
            observation_count,
            index_path: self.path.display().to_string(),
        })
    }

    pub fn list_profiles(&self) -> Vec<SpeakerProfileSummary> {
        profile_summaries(&self.index)
    }

    pub fn remove_observation(&mut self, observation_id: &str) -> Result<String, String> {
        let speaker_id = self.index.remove_observation(observation_id)?;
        self.save()?;
        Ok(speaker_id)
    }

    pub fn remove_speaker(&mut self, speaker_id: &str) -> Result<usize, String> {
        let removed = self.index.remove_speaker(speaker_id)?;
        self.save()?;
        Ok(removed)
    }

    fn enroll_embeddings_atomic(
        &mut self,
        speaker_id: &str,
        embeddings: &[SpeakerEmbedding],
        force: bool,
        overrides: ThresholdOverrides,
    ) -> Result<Vec<SpeakerEnrollmentResult>, String> {
        for embedding in embeddings {
            self.validate_embedding(embedding)?;
        }
        apply_threshold_overrides(&mut self.index.thresholds, overrides)?;
        let mut results = Vec::with_capacity(embeddings.len());
        for embedding in embeddings {
            let (observation_id, observation_count) = self.index.enroll(
                speaker_id,
                embedding.vector.clone(),
                embedding.source.clone(),
                embedding.quality.duration_seconds,
                embedding.quality.score,
                force,
            )?;
            results.push(SpeakerEnrollmentResult {
                speaker_id: speaker_id.to_string(),
                observation_id,
                observation_count,
                index_path: self.path.display().to_string(),
            });
        }
        self.save()?;
        Ok(results)
    }

    fn apply_identification_learning(
        &mut self,
        mode: CliSpeakerLearningMode,
        candidate_path: Option<&Path>,
        embedding: &SpeakerEmbedding,
        identification: &SpeakerIdentificationResult,
    ) -> Result<(), String> {
        let candidate = self.candidate_for(embedding, identification);
        match (mode, candidate) {
            (CliSpeakerLearningMode::Off, _) => Ok(()),
            (CliSpeakerLearningMode::Candidates, Some(candidate)) => {
                let path = candidate_path.ok_or_else(|| {
                    "internal error: candidate mode has no output path".to_string()
                })?;
                write_candidate(path, &candidate)?;
                eprintln!(
                    "speaker learning candidate written: {} ({})",
                    candidate.candidate_id,
                    path.display()
                );
                Ok(())
            }
            (CliSpeakerLearningMode::Auto, Some(candidate)) => {
                let (observation_id, observation_count) = self.index.enroll(
                    &candidate.speaker_id,
                    candidate.vector,
                    candidate.source,
                    candidate.duration_seconds,
                    candidate.quality_score,
                    false,
                )?;
                self.save()?;
                eprintln!(
                    "speaker profile auto-refined: {} observation={} count={}",
                    candidate.speaker_id, observation_id, observation_count
                );
                Ok(())
            }
            (CliSpeakerLearningMode::Candidates | CliSpeakerLearningMode::Auto, None) => {
                eprintln!("speaker profile unchanged: match did not meet the auto-learning gate");
                Ok(())
            }
        }
    }

    fn accept_candidates(
        &mut self,
        candidate_path: &Path,
        force: bool,
    ) -> Result<(usize, usize), String> {
        let candidates = read_candidates(candidate_path)?;
        let mut accepted = 0usize;
        let mut skipped = 0usize;
        for candidate in candidates {
            if !candidate.accepted {
                skipped += 1;
                continue;
            }
            validate_candidate(
                &candidate,
                &self.index.model_fingerprint,
                self.index.embedding_dimension,
            )?;
            self.index.enroll(
                &candidate.speaker_id,
                candidate.vector,
                candidate.source,
                candidate.duration_seconds,
                candidate.quality_score,
                force,
            )?;
            accepted += 1;
        }
        if accepted > 0 {
            self.save()?;
        }
        Ok((accepted, skipped))
    }

    fn candidate_for(
        &self,
        embedding: &SpeakerEmbedding,
        identification: &SpeakerIdentificationResult,
    ) -> Option<SpeakerLearningCandidate> {
        let speaker_id = identification.speaker_id.as_ref()?;
        let score = identification.score?;
        let margin_ok = identification
            .margin
            .is_none_or(|margin| margin >= self.index.thresholds.identification_margin);
        if score < self.index.thresholds.auto_learning || !margin_ok {
            return None;
        }
        Some(SpeakerLearningCandidate {
            format: CANDIDATE_FORMAT.to_string(),
            version: CANDIDATE_VERSION,
            candidate_id: candidate_id(speaker_id, &embedding.vector, &embedding.source),
            accepted: false,
            speaker_id: speaker_id.clone(),
            model_fingerprint: embedding.model_fingerprint.clone(),
            source: embedding.source.clone(),
            duration_seconds: embedding.quality.duration_seconds,
            quality_score: embedding.quality.score,
            score,
            margin: identification.margin,
            vector: embedding.vector.clone(),
        })
    }

    fn validate_embedding(&self, embedding: &SpeakerEmbedding) -> Result<(), String> {
        validate_embedding_identity(
            embedding,
            &self.index.model_fingerprint,
            &self.index.model_name,
            &self.index.model_architecture,
            self.index.embedding_dimension,
        )
    }

    fn save(&self) -> Result<(), String> {
        self.index.save_atomic(&self.path)
    }
}

pub(crate) fn run_cli(cli: &CliOptions) -> Result<(), String> {
    let action = cli
        .speaker_action
        .as_ref()
        .ok_or_else(|| "internal error: speaker mode has no action".to_string())?;
    let stored_embeddings = read_embedding_inputs(&cli.speaker_embedding_inputs)?;
    validate_embedding_input_count(action, cli.audios.len(), stored_embeddings.len())?;
    let index_path = cli.speaker_index.as_deref().map(Path::new);
    let overrides = ThresholdOverrides {
        verification: cli.speaker_threshold,
        identification_margin: cli.speaker_margin,
    };
    if !cli_action_requires_encoder(cli, action, index_path) {
        return run_cli_without_encoder(
            cli,
            action,
            required_index_path(index_path)?,
            &stored_embeddings,
            overrides,
        );
    }
    let model_path = cli.speaker_model.as_deref().ok_or_else(|| {
        if matches!(action, CliSpeakerAction::Enroll(_)) && cli.audios.is_empty() {
            "creating a speaker index from exported embeddings requires --speaker-model so its calibrated thresholds can be stored"
                .to_string()
        } else {
            "this speaker action requires --speaker-model <speaker.gguf>".to_string()
        }
    })?;
    let runtime = SpeakerRuntime::load_from_file_with_debug(Path::new(model_path), cli.debug)?;
    match action {
        CliSpeakerAction::Embed => {
            for audio in &cli.audios {
                write_stdout_json(&runtime.embed_file(Path::new(audio))?)?;
            }
        }
        CliSpeakerAction::Enroll(speaker_id) => {
            let index_path = required_index_path(index_path)?;
            for result in runtime.enroll_inputs_atomic(
                index_path,
                speaker_id,
                &cli.audios,
                &stored_embeddings,
                cli.speaker_force_enroll,
                overrides,
            )? {
                write_stdout_json(&result)?;
            }
        }
        CliSpeakerAction::Verify(speaker_id) => {
            let index_path = required_index_path(index_path)?;
            let result = if let Some(audio) = cli.audios.first() {
                runtime.verify_file(index_path, speaker_id, Path::new(audio))?
            } else {
                runtime.verify_embedding(index_path, speaker_id, &stored_embeddings[0])?
            };
            write_stdout_json(&result)?;
        }
        CliSpeakerAction::Identify => {
            let index_path = required_index_path(index_path)?;
            let embedding = if let Some(audio) = cli.audios.first() {
                runtime.embed_file(Path::new(audio))?
            } else {
                stored_embeddings[0].clone()
            };
            let result = runtime.identify_embedding(index_path, &embedding)?;
            apply_identification_learning(
                &runtime,
                index_path,
                cli.speaker_learning,
                cli.speaker_candidates.as_deref().map(Path::new),
                &embedding,
                &result,
            )?;
            write_stdout_json(&result)?;
        }
        CliSpeakerAction::Diarize => {
            if let Some(index_path) = index_path {
                let outcome =
                    runtime.diarize_with_candidates(Path::new(&cli.audios[0]), index_path)?;
                apply_diarization_learning(
                    &runtime,
                    index_path,
                    cli.speaker_learning,
                    cli.speaker_candidates.as_deref().map(Path::new),
                    outcome.candidates,
                )?;
                write_stdout_json(&outcome.result)?;
            } else {
                write_stdout_json(&runtime.diarize_file(None, Path::new(&cli.audios[0]))?)?;
            }
        }
        CliSpeakerAction::Accept(_)
        | CliSpeakerAction::List
        | CliSpeakerAction::RemoveSpeaker(_)
        | CliSpeakerAction::RemoveObservation(_) => {
            return Err(
                "internal error: index-only action reached the speaker encoder".to_string(),
            );
        }
    }
    Ok(())
}

fn cli_action_requires_encoder(
    cli: &CliOptions,
    action: &CliSpeakerAction,
    index_path: Option<&Path>,
) -> bool {
    match action {
        CliSpeakerAction::Embed | CliSpeakerAction::Diarize => true,
        CliSpeakerAction::Enroll(_) => {
            !cli.audios.is_empty() || index_path.is_none_or(|path| !path.exists())
        }
        CliSpeakerAction::Verify(_) | CliSpeakerAction::Identify => !cli.audios.is_empty(),
        CliSpeakerAction::Accept(_)
        | CliSpeakerAction::List
        | CliSpeakerAction::RemoveSpeaker(_)
        | CliSpeakerAction::RemoveObservation(_) => false,
    }
}

fn run_cli_without_encoder(
    cli: &CliOptions,
    action: &CliSpeakerAction,
    index_path: &Path,
    stored_embeddings: &[SpeakerEmbedding],
    overrides: ThresholdOverrides,
) -> Result<(), String> {
    let mut index = SpeakerIndexRuntime::load(index_path)?;
    match action {
        CliSpeakerAction::Enroll(speaker_id) => {
            for result in index.enroll_embeddings_atomic(
                speaker_id,
                stored_embeddings,
                cli.speaker_force_enroll,
                overrides,
            )? {
                write_stdout_json(&result)?;
            }
        }
        CliSpeakerAction::Verify(speaker_id) => {
            write_stdout_json(&index.verify_embedding(speaker_id, &stored_embeddings[0])?)?;
        }
        CliSpeakerAction::Identify => {
            let result = index.identify_embedding(&stored_embeddings[0])?;
            index.apply_identification_learning(
                cli.speaker_learning,
                cli.speaker_candidates.as_deref().map(Path::new),
                &stored_embeddings[0],
                &result,
            )?;
            write_stdout_json(&result)?;
        }
        CliSpeakerAction::Accept(candidate_path) => {
            let (accepted, skipped) =
                index.accept_candidates(Path::new(candidate_path), cli.speaker_force_enroll)?;
            write_stdout_json(&serde_json::json!({
                "accepted": accepted,
                "skipped": skipped,
                "index_path": index_path.display().to_string(),
            }))?;
        }
        CliSpeakerAction::List => {
            write_stdout_json(&index.list_profiles())?;
        }
        CliSpeakerAction::RemoveSpeaker(speaker_id) => {
            let removed = index.remove_speaker(speaker_id)?;
            write_stdout_json(&serde_json::json!({
                "speaker_id": speaker_id,
                "removed_observations": removed,
                "index_path": index_path.display().to_string(),
            }))?;
        }
        CliSpeakerAction::RemoveObservation(observation_id) => {
            let speaker_id = index.remove_observation(observation_id)?;
            write_stdout_json(&serde_json::json!({
                "speaker_id": speaker_id,
                "removed_observation": observation_id,
                "index_path": index_path.display().to_string(),
            }))?;
        }
        CliSpeakerAction::Embed | CliSpeakerAction::Diarize => {
            return Err("internal error: encoder action reached index-only execution".to_string());
        }
    }
    Ok(())
}

fn required_index_path(index_path: Option<&Path>) -> Result<&Path, String> {
    index_path.ok_or_else(|| "internal error: speaker action has no index path".to_string())
}

fn validate_embedding_input_count(
    action: &CliSpeakerAction,
    audio_count: usize,
    embedding_count: usize,
) -> Result<(), String> {
    match action {
        CliSpeakerAction::Enroll(_) if audio_count + embedding_count == 0 => {
            Err("--speaker-enroll received no audio or embedding records".to_string())
        }
        CliSpeakerAction::Verify(_) | CliSpeakerAction::Identify
            if audio_count + embedding_count != 1 =>
        {
            Err(format!(
                "this speaker action requires exactly one audio or embedding record, got {}",
                audio_count + embedding_count
            ))
        }
        _ => Ok(()),
    }
}

fn write_stdout_json(value: &impl Serialize) -> Result<(), String> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, value)
        .map_err(|error| format!("cannot serialize speaker result: {error}"))?;
    output
        .write_all(b"\n")
        .map_err(|error| format!("cannot write speaker result: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("cannot flush speaker result: {error}"))
}

fn read_embedding_inputs(paths: &[String]) -> Result<Vec<SpeakerEmbedding>, String> {
    let mut records = Vec::new();
    let mut total_bytes = 0u64;
    for path in paths {
        let path = Path::new(path);
        let metadata = fs::metadata(path).map_err(|error| {
            format!(
                "cannot read speaker embedding input '{}': {error}",
                path.display()
            )
        })?;
        if !metadata.is_file() {
            return Err(format!(
                "speaker embedding input is not a file: {}",
                path.display()
            ));
        }
        total_bytes = total_bytes
            .checked_add(metadata.len())
            .ok_or_else(|| "speaker embedding input byte count overflow".to_string())?;
        if total_bytes > MAX_SPEAKER_JSONL_BYTES {
            return Err(format!(
                "speaker embedding inputs exceed {MAX_SPEAKER_JSONL_BYTES} bytes"
            ));
        }
        let file = fs::File::open(path).map_err(|error| {
            format!(
                "cannot open speaker embedding input '{}': {error}",
                path.display()
            )
        })?;
        let mut reader = BufReader::new(file);
        let mut line = Vec::new();
        let mut line_number = 0usize;
        loop {
            line.clear();
            let bytes = reader.read_until(b'\n', &mut line).map_err(|error| {
                format!(
                    "cannot read speaker embedding input '{}' at line {}: {error}",
                    path.display(),
                    line_number + 1
                )
            })?;
            if bytes == 0 {
                break;
            }
            line_number += 1;
            if line.len() > MAX_CANDIDATE_LINE_BYTES {
                return Err(format!(
                    "speaker embedding line {line_number} in '{}' exceeds {MAX_CANDIDATE_LINE_BYTES} bytes",
                    path.display()
                ));
            }
            let mut end = line.len();
            while end > 0 && matches!(line[end - 1], b'\n' | b'\r') {
                end -= 1;
            }
            let trimmed = &line[..end];
            if trimmed.iter().all(u8::is_ascii_whitespace) {
                return Err(format!(
                    "speaker embedding input '{}' contains a blank line at {line_number}",
                    path.display()
                ));
            }
            let record = serde_json::from_slice(trimmed).map_err(|error| {
                format!(
                    "invalid speaker embedding JSON in '{}' at line {line_number}: {error}",
                    path.display()
                )
            })?;
            records.push(record);
            if records.len() > MAX_CANDIDATE_RECORDS {
                return Err(format!(
                    "speaker embedding inputs exceed {MAX_CANDIDATE_RECORDS} records"
                ));
            }
        }
        if line_number == 0 {
            return Err(format!(
                "speaker embedding input '{}' contains no records",
                path.display()
            ));
        }
    }
    Ok(records)
}

impl SpeakerRuntime {
    /// Load a speaker GGUF embedded with `include_bytes!`.
    pub fn load_from_bytes(data: &'static [u8]) -> Result<Self, String> {
        let gguf = parse_gguf_from_bytes(data, false)
            .map_err(|error| format!("failed to load embedded speaker GGUF: {error}"))?;
        let policy = vendors::speaker_model_policy(&gguf)?;
        let encoder = SpeakerEncoder::new(gguf, policy)?;
        Ok(Self { encoder })
    }

    /// Load a dedicated speaker-embedding GGUF. This does not load a language model.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        Self::load_from_file_with_debug(path, false)
    }

    pub(crate) fn load_from_file_with_debug(path: &Path, debug: bool) -> Result<Self, String> {
        let path_string = path
            .to_str()
            .ok_or_else(|| "speaker model path contains non-UTF8 characters".to_string())?;
        let gguf = parse_gguf_file(path_string, debug)
            .map_err(|error| format!("failed to load speaker GGUF '{path_string}': {error}"))?;
        let policy = vendors::speaker_model_policy(&gguf)?;
        let encoder = SpeakerEncoder::new(gguf, policy)?;
        Ok(Self { encoder })
    }

    pub fn model_name(&self) -> &str {
        self.encoder.model_name()
    }

    pub fn model_architecture(&self) -> &str {
        self.encoder.architecture()
    }

    pub fn model_fingerprint(&self) -> &str {
        self.encoder.fingerprint()
    }

    pub fn embedding_dimension(&self) -> usize {
        self.encoder.embedding_dim()
    }

    /// Create one L2-normalized speaker embedding without modifying an index.
    pub fn embed_file(&self, audio_path: &Path) -> Result<SpeakerEmbedding, String> {
        let source = path_string(audio_path, "speaker audio")?;
        let output = self.encoder.embed_file(&source)?;
        Ok(self.public_embedding(source, output))
    }

    /// Add one trusted observation. The first call creates the profile; later calls refine it.
    pub fn enroll_file(
        &self,
        index_path: &Path,
        speaker_id: &str,
        audio_path: &Path,
        force: bool,
    ) -> Result<SpeakerEnrollmentResult, String> {
        self.enroll_file_with_options(
            index_path,
            speaker_id,
            audio_path,
            force,
            ThresholdOverrides::default(),
        )
    }

    /// Add a previously exported embedding without decoding or encoding its audio again.
    pub fn enroll_embedding(
        &self,
        index_path: &Path,
        speaker_id: &str,
        embedding: &SpeakerEmbedding,
        force: bool,
    ) -> Result<SpeakerEnrollmentResult, String> {
        self.enroll_embedding_with_options(
            index_path,
            speaker_id,
            embedding,
            force,
            ThresholdOverrides::default(),
        )
    }

    /// Verify a claimed speaker identity using the index threshold.
    pub fn verify_file(
        &self,
        index_path: &Path,
        speaker_id: &str,
        audio_path: &Path,
    ) -> Result<SpeakerVerificationResult, String> {
        let embedding = self.embed_file(audio_path)?;
        self.verify_embedding(index_path, speaker_id, &embedding)
    }

    /// Verify a claimed identity using a previously exported embedding.
    pub fn verify_embedding(
        &self,
        index_path: &Path,
        speaker_id: &str,
        embedding: &SpeakerEmbedding,
    ) -> Result<SpeakerVerificationResult, String> {
        self.validate_embedding(embedding)?;
        let index = self.load_index(index_path)?;
        let profile = index
            .speaker(speaker_id)
            .ok_or_else(|| format!("speaker profile not found: {speaker_id}"))?;
        let score = cosine_similarity(&embedding.vector, &profile.centroid)?;
        Ok(SpeakerVerificationResult {
            speaker_id: speaker_id.to_string(),
            verified: score >= index.thresholds.verification,
            score,
            threshold: index.thresholds.verification,
            source: embedding.source.clone(),
        })
    }

    /// Perform open-set 1:N identification. A weak or ambiguous match remains unknown.
    pub fn identify_file(
        &self,
        index_path: &Path,
        audio_path: &Path,
    ) -> Result<SpeakerIdentificationResult, String> {
        let embedding = self.embed_file(audio_path)?;
        self.identify_embedding(index_path, &embedding)
    }

    /// Identify a previously exported embedding without running the encoder again.
    pub fn identify_embedding(
        &self,
        index_path: &Path,
        embedding: &SpeakerEmbedding,
    ) -> Result<SpeakerIdentificationResult, String> {
        self.validate_embedding(embedding)?;
        let index = self.load_index(index_path)?;
        identify_vector(&index, &embedding.vector, embedding.source.clone())
    }

    /// Segment a meeting using an in-process energy VAD, then associate or cluster each segment.
    /// Overlapping speakers are not separated.
    pub fn diarize_file(
        &self,
        index_path: Option<&Path>,
        audio_path: &Path,
    ) -> Result<SpeakerDiarizationResult, String> {
        let index = index_path.map(|path| self.load_index(path)).transpose()?;
        Ok(self.diarize_with_index(audio_path, index.as_ref())?.result)
    }

    pub fn list_profiles(&self, index_path: &Path) -> Result<Vec<SpeakerProfileSummary>, String> {
        let index = self.load_index(index_path)?;
        Ok(profile_summaries(&index))
    }

    pub fn remove_observation(
        &self,
        index_path: &Path,
        observation_id: &str,
    ) -> Result<String, String> {
        let mut index = self.load_index(index_path)?;
        let speaker_id = index.remove_observation(observation_id)?;
        index.save_atomic(index_path)?;
        Ok(speaker_id)
    }

    pub fn remove_speaker(&self, index_path: &Path, speaker_id: &str) -> Result<usize, String> {
        let mut index = self.load_index(index_path)?;
        let removed = index.remove_speaker(speaker_id)?;
        index.save_atomic(index_path)?;
        Ok(removed)
    }

    pub(crate) fn enroll_file_with_options(
        &self,
        index_path: &Path,
        speaker_id: &str,
        audio_path: &Path,
        force: bool,
        overrides: ThresholdOverrides,
    ) -> Result<SpeakerEnrollmentResult, String> {
        let embedding = self.embed_file(audio_path)?;
        self.enroll_embedding_with_options(index_path, speaker_id, &embedding, force, overrides)
    }

    fn enroll_embedding_with_options(
        &self,
        index_path: &Path,
        speaker_id: &str,
        embedding: &SpeakerEmbedding,
        force: bool,
        overrides: ThresholdOverrides,
    ) -> Result<SpeakerEnrollmentResult, String> {
        self.validate_embedding(embedding)?;
        let mut index = self.load_or_create_index(index_path, overrides)?;
        let (observation_id, observation_count) = index.enroll(
            speaker_id,
            embedding.vector.clone(),
            embedding.source.clone(),
            embedding.quality.duration_seconds,
            embedding.quality.score,
            force,
        )?;
        index.save_atomic(index_path)?;
        Ok(SpeakerEnrollmentResult {
            speaker_id: speaker_id.to_string(),
            observation_id,
            observation_count,
            index_path: index_path.display().to_string(),
        })
    }

    pub(crate) fn enroll_inputs_atomic(
        &self,
        index_path: &Path,
        speaker_id: &str,
        audio_paths: &[String],
        stored_embeddings: &[SpeakerEmbedding],
        force: bool,
        overrides: ThresholdOverrides,
    ) -> Result<Vec<SpeakerEnrollmentResult>, String> {
        let mut prepared = Vec::with_capacity(audio_paths.len() + stored_embeddings.len());
        for path in audio_paths {
            prepared.push(self.embed_file(Path::new(path))?);
        }
        for embedding in stored_embeddings {
            self.validate_embedding(embedding)?;
            prepared.push(embedding.clone());
        }
        let mut index = self.load_or_create_index(index_path, overrides)?;
        let mut results = Vec::with_capacity(prepared.len());
        for embedding in prepared {
            let (observation_id, observation_count) = index.enroll(
                speaker_id,
                embedding.vector,
                embedding.source,
                embedding.quality.duration_seconds,
                embedding.quality.score,
                force,
            )?;
            results.push(SpeakerEnrollmentResult {
                speaker_id: speaker_id.to_string(),
                observation_id,
                observation_count,
                index_path: index_path.display().to_string(),
            });
        }
        index.save_atomic(index_path)?;
        Ok(results)
    }

    fn load_or_create_index(
        &self,
        index_path: &Path,
        overrides: ThresholdOverrides,
    ) -> Result<SpeakerIndex, String> {
        if index_path.exists() {
            let mut index = self.load_index(index_path)?;
            apply_threshold_overrides(&mut index.thresholds, overrides)?;
            return Ok(index);
        }
        let mut thresholds = stored_thresholds(self.encoder.thresholds());
        apply_threshold_overrides(&mut thresholds, overrides)?;
        SpeakerIndex::new(
            self.encoder.fingerprint().to_string(),
            self.encoder.model_name().to_string(),
            self.encoder.architecture().to_string(),
            self.encoder.embedding_dim(),
            thresholds,
        )
    }

    fn load_index(&self, index_path: &Path) -> Result<SpeakerIndex, String> {
        let index = SpeakerIndex::load(index_path)?;
        index.validate_model(self.encoder.fingerprint(), self.encoder.embedding_dim())?;
        Ok(index)
    }

    pub(crate) fn candidate_for(
        &self,
        embedding: &SpeakerEmbedding,
        identification: &SpeakerIdentificationResult,
    ) -> Option<SpeakerLearningCandidate> {
        let speaker_id = identification.speaker_id.as_ref()?;
        let score = identification.score?;
        let thresholds = self.encoder.thresholds();
        let margin_ok = identification
            .margin
            .is_none_or(|margin| margin >= thresholds.identification_margin);
        if score < thresholds.auto_learning || !margin_ok {
            return None;
        }
        Some(SpeakerLearningCandidate {
            format: CANDIDATE_FORMAT.to_string(),
            version: CANDIDATE_VERSION,
            candidate_id: candidate_id(speaker_id, &embedding.vector, &embedding.source),
            accepted: false,
            speaker_id: speaker_id.clone(),
            model_fingerprint: embedding.model_fingerprint.clone(),
            source: embedding.source.clone(),
            duration_seconds: embedding.quality.duration_seconds,
            quality_score: embedding.quality.score,
            score,
            margin: identification.margin,
            vector: embedding.vector.clone(),
        })
    }

    pub(crate) fn accept_candidates(
        &self,
        index_path: &Path,
        candidate_path: &Path,
        force: bool,
    ) -> Result<(usize, usize), String> {
        let candidates = read_candidates(candidate_path)?;
        let mut index = self.load_index(index_path)?;
        let mut accepted = 0usize;
        let mut skipped = 0usize;
        for candidate in candidates {
            if !candidate.accepted {
                skipped += 1;
                continue;
            }
            validate_candidate(
                &candidate,
                self.encoder.fingerprint(),
                self.encoder.embedding_dim(),
            )?;
            index.enroll(
                &candidate.speaker_id,
                candidate.vector,
                candidate.source,
                candidate.duration_seconds,
                candidate.quality_score,
                force,
            )?;
            accepted += 1;
        }
        if accepted > 0 {
            index.save_atomic(index_path)?;
        }
        Ok((accepted, skipped))
    }

    pub(crate) fn auto_enroll_candidate(
        &self,
        index_path: &Path,
        candidate: SpeakerLearningCandidate,
    ) -> Result<SpeakerEnrollmentResult, String> {
        let mut index = self.load_index(index_path)?;
        validate_candidate(
            &candidate,
            self.encoder.fingerprint(),
            self.encoder.embedding_dim(),
        )?;
        let (observation_id, observation_count) = index.enroll(
            &candidate.speaker_id,
            candidate.vector,
            candidate.source,
            candidate.duration_seconds,
            candidate.quality_score,
            false,
        )?;
        index.save_atomic(index_path)?;
        Ok(SpeakerEnrollmentResult {
            speaker_id: candidate.speaker_id,
            observation_id,
            observation_count,
            index_path: index_path.display().to_string(),
        })
    }

    pub(crate) fn auto_enroll_candidates(
        &self,
        index_path: &Path,
        candidates: Vec<SpeakerLearningCandidate>,
    ) -> Result<usize, String> {
        if candidates.is_empty() {
            return Ok(0);
        }
        let mut index = self.load_index(index_path)?;
        let count = candidates.len();
        for candidate in candidates {
            validate_candidate(
                &candidate,
                self.encoder.fingerprint(),
                self.encoder.embedding_dim(),
            )?;
            index.enroll(
                &candidate.speaker_id,
                candidate.vector,
                candidate.source,
                candidate.duration_seconds,
                candidate.quality_score,
                false,
            )?;
        }
        index.save_atomic(index_path)?;
        Ok(count)
    }

    fn diarize_with_candidates(
        &self,
        audio_path: &Path,
        index_path: &Path,
    ) -> Result<DiarizationOutcome, String> {
        let index = self.load_index(index_path)?;
        self.diarize_with_index(audio_path, Some(&index))
    }

    fn public_embedding(&self, source: String, output: SpeakerEmbeddingOutput) -> SpeakerEmbedding {
        SpeakerEmbedding {
            format: EMBEDDING_FORMAT.to_string(),
            version: EMBEDDING_VERSION,
            model_name: self.encoder.model_name().to_string(),
            model_architecture: self.encoder.architecture().to_string(),
            model_fingerprint: self.encoder.fingerprint().to_string(),
            dimension: output.vector.len(),
            source,
            quality: public_quality(output.quality),
            vector: output.vector,
        }
    }

    fn validate_embedding(&self, embedding: &SpeakerEmbedding) -> Result<(), String> {
        validate_embedding_identity(
            embedding,
            self.encoder.fingerprint(),
            self.encoder.model_name(),
            self.encoder.architecture(),
            self.encoder.embedding_dim(),
        )
    }

    fn diarize_with_index(
        &self,
        audio_path: &Path,
        index: Option<&SpeakerIndex>,
    ) -> Result<DiarizationOutcome, String> {
        let source = path_string(audio_path, "speaker audio")?;
        let decoded = self.encoder.decode_file(&source)?;
        let speech_regions = detect_speech_regions(&decoded.samples_mono_f32, decoded.sample_rate)?;
        let chunks = split_speech_regions(
            &speech_regions,
            decoded.sample_rate,
            self.encoder.min_audio_seconds(),
        );
        let mut clusters: Vec<(Vec<f32>, usize)> = Vec::new();
        let cluster_threshold = index.map_or_else(
            || self.encoder.thresholds().diarization_cluster,
            |value| value.thresholds.diarization_cluster,
        );
        let mut raw_segments = Vec::new();
        let mut candidates = Vec::new();
        for (start, end) in chunks {
            let output = self
                .encoder
                .embed_samples(&decoded.samples_mono_f32[start..end])?;
            let (speaker_id, recognized, score) = if let Some(index) = index {
                let identification = identify_vector(index, &output.vector, source.clone())?;
                if let Some(speaker_id) = identification.speaker_id.as_ref() {
                    let segment_source = format!(
                        "{}#{:.3}-{:.3}",
                        source,
                        start as f32 / decoded.sample_rate as f32,
                        end as f32 / decoded.sample_rate as f32
                    );
                    if let Some(candidate) = self.candidate_from_output(
                        speaker_id,
                        &segment_source,
                        &output,
                        &identification,
                    ) {
                        candidates.push(candidate);
                    }
                    (speaker_id.clone(), true, identification.score)
                } else {
                    let (label, cluster_score) =
                        assign_cluster(&mut clusters, &output.vector, cluster_threshold)?;
                    (label, false, cluster_score)
                }
            } else {
                let (label, cluster_score) =
                    assign_cluster(&mut clusters, &output.vector, cluster_threshold)?;
                (label, false, cluster_score)
            };
            raw_segments.push(SpeakerDiarizationSegment {
                start_seconds: start as f32 / decoded.sample_rate as f32,
                end_seconds: end as f32 / decoded.sample_rate as f32,
                speaker_id,
                recognized,
                score,
            });
        }
        Ok(DiarizationOutcome {
            result: SpeakerDiarizationResult {
                source,
                sample_rate: decoded.sample_rate,
                segments: merge_adjacent_segments(raw_segments),
            },
            candidates: coalesce_candidates(candidates)?,
        })
    }

    fn candidate_from_output(
        &self,
        speaker_id: &str,
        source: &str,
        output: &SpeakerEmbeddingOutput,
        identification: &SpeakerIdentificationResult,
    ) -> Option<SpeakerLearningCandidate> {
        let score = identification.score?;
        let thresholds = self.encoder.thresholds();
        let margin_ok = identification
            .margin
            .is_none_or(|margin| margin >= thresholds.identification_margin);
        if score < thresholds.auto_learning || !margin_ok {
            return None;
        }
        Some(SpeakerLearningCandidate {
            format: CANDIDATE_FORMAT.to_string(),
            version: CANDIDATE_VERSION,
            candidate_id: candidate_id(speaker_id, &output.vector, source),
            accepted: false,
            speaker_id: speaker_id.to_string(),
            model_fingerprint: self.encoder.fingerprint().to_string(),
            source: source.to_string(),
            duration_seconds: output.quality.duration_seconds,
            quality_score: output.quality.score,
            score,
            margin: identification.margin,
            vector: output.vector.clone(),
        })
    }
}

fn validate_embedding_identity(
    embedding: &SpeakerEmbedding,
    model_fingerprint: &str,
    model_name: &str,
    model_architecture: &str,
    embedding_dimension: usize,
) -> Result<(), String> {
    if embedding.format != EMBEDDING_FORMAT || embedding.version != EMBEDDING_VERSION {
        return Err(format!(
            "unsupported speaker embedding format/version for '{}'",
            embedding.source
        ));
    }
    if embedding.model_fingerprint != model_fingerprint
        || embedding.model_name != model_name
        || embedding.model_architecture != model_architecture
        || embedding.dimension != embedding_dimension
        || embedding.vector.len() != embedding.dimension
    {
        return Err(format!(
            "speaker embedding '{}' belongs to a different model or dimension",
            embedding.source
        ));
    }
    if embedding.source.is_empty()
        || embedding.source.len() > 16_384
        || !embedding.quality.duration_seconds.is_finite()
        || embedding.quality.duration_seconds <= 0.0
        || !embedding.quality.rms.is_finite()
        || embedding.quality.rms < 0.0
        || !embedding.quality.clipping_fraction.is_finite()
        || !(0.0..=1.0).contains(&embedding.quality.clipping_fraction)
        || !embedding.quality.active_fraction.is_finite()
        || !(0.0..=1.0).contains(&embedding.quality.active_fraction)
        || !embedding.quality.score.is_finite()
        || !(0.0..=1.0).contains(&embedding.quality.score)
        || embedding.vector.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "speaker embedding '{}' has invalid source, vector, or quality metadata",
            embedding.source
        ));
    }
    let mut normalized = embedding.vector.clone();
    normalize_embedding(&mut normalized)?;
    if normalized
        .iter()
        .zip(&embedding.vector)
        .any(|(expected, stored)| (expected - stored).abs() > 1e-3)
    {
        return Err(format!(
            "speaker embedding '{}' is not L2-normalized",
            embedding.source
        ));
    }
    Ok(())
}

fn coalesce_candidates(
    candidates: Vec<SpeakerLearningCandidate>,
) -> Result<Vec<SpeakerLearningCandidate>, String> {
    let mut grouped: BTreeMap<String, Vec<SpeakerLearningCandidate>> = BTreeMap::new();
    for candidate in candidates {
        grouped
            .entry(candidate.speaker_id.clone())
            .or_default()
            .push(candidate);
    }
    let mut combined = Vec::with_capacity(grouped.len());
    for (speaker_id, group) in grouped {
        let first = group
            .first()
            .ok_or_else(|| "cannot combine an empty speaker candidate group".to_string())?;
        let mut vector = vec![0.0f32; first.vector.len()];
        let mut total_weight = 0.0f32;
        let mut total_duration = 0.0f32;
        let mut weighted_quality = 0.0f32;
        let mut weighted_score = 0.0f32;
        let mut minimum_margin: Option<f32> = None;
        for candidate in &group {
            if candidate.model_fingerprint != first.model_fingerprint
                || candidate.vector.len() != vector.len()
            {
                return Err("cannot combine candidates from different speaker models".to_string());
            }
            let weight = (candidate.duration_seconds * candidate.quality_score.max(0.1)).max(0.1);
            for (destination, value) in vector.iter_mut().zip(&candidate.vector) {
                *destination += value * weight;
            }
            total_weight += weight;
            total_duration += candidate.duration_seconds;
            weighted_quality += candidate.quality_score * weight;
            weighted_score += candidate.score * weight;
            if let Some(margin) = candidate.margin {
                minimum_margin = Some(minimum_margin.map_or(margin, |current| current.min(margin)));
            }
        }
        for value in &mut vector {
            *value /= total_weight;
        }
        normalize_embedding(&mut vector)?;
        let source = if group.len() == 1 {
            first.source.clone()
        } else {
            format!("{} (+{} meeting segments)", first.source, group.len() - 1)
        };
        combined.push(SpeakerLearningCandidate {
            format: CANDIDATE_FORMAT.to_string(),
            version: CANDIDATE_VERSION,
            candidate_id: candidate_id(&speaker_id, &vector, &source),
            accepted: false,
            speaker_id,
            model_fingerprint: first.model_fingerprint.clone(),
            source,
            duration_seconds: total_duration,
            quality_score: weighted_quality / total_weight,
            score: weighted_score / total_weight,
            margin: minimum_margin,
            vector,
        });
    }
    Ok(combined)
}

pub(crate) fn write_candidate(
    path: &Path,
    candidate: &SpeakerLearningCandidate,
) -> Result<(), String> {
    let mut line = serde_json::to_vec(candidate)
        .map_err(|error| format!("cannot serialize speaker candidate: {error}"))?;
    if line.len() > MAX_CANDIDATE_LINE_BYTES {
        return Err("speaker candidate exceeds the JSONL line-size limit".to_string());
    }
    line.push(b'\n');
    let existing_bytes = match fs::metadata(path) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => {
            return Err(format!(
                "cannot inspect speaker candidate file '{}': {error}",
                path.display()
            ));
        }
    };
    let resulting_bytes = existing_bytes
        .checked_add(line.len() as u64)
        .ok_or_else(|| "speaker candidate file size overflow".to_string())?;
    if resulting_bytes > MAX_SPEAKER_JSONL_BYTES {
        return Err(format!(
            "speaker candidate file would exceed {MAX_SPEAKER_JSONL_BYTES} bytes"
        ));
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| {
            format!(
                "cannot open speaker candidate file '{}': {error}",
                path.display()
            )
        })?;
    file.write_all(&line).map_err(|error| {
        format!(
            "cannot append speaker candidate file '{}': {error}",
            path.display()
        )
    })?;
    file.flush().map_err(|error| {
        format!(
            "cannot flush speaker candidate file '{}': {error}",
            path.display()
        )
    })
}

fn read_candidates(path: &Path) -> Result<Vec<SpeakerLearningCandidate>, String> {
    let metadata = fs::metadata(path).map_err(|error| {
        format!(
            "cannot read speaker candidate file '{}': {error}",
            path.display()
        )
    })?;
    if !metadata.is_file() {
        return Err(format!(
            "speaker candidate path is not a file: {}",
            path.display()
        ));
    }
    if metadata.len() > MAX_SPEAKER_JSONL_BYTES {
        return Err(format!(
            "speaker candidate file '{}' exceeds {MAX_SPEAKER_JSONL_BYTES} bytes",
            path.display()
        ));
    }
    let file = fs::File::open(path).map_err(|error| {
        format!(
            "cannot open speaker candidate file '{}': {error}",
            path.display()
        )
    })?;
    let mut records = Vec::new();
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    let mut line_number = 0usize;
    loop {
        line.clear();
        let bytes = reader.read_until(b'\n', &mut line).map_err(|error| {
            format!(
                "cannot read speaker candidate file '{}' at line {}: {error}",
                path.display(),
                line_number + 1
            )
        })?;
        if bytes == 0 {
            break;
        }
        line_number += 1;
        if line.len() > MAX_CANDIDATE_LINE_BYTES {
            return Err(format!(
                "speaker candidate line {line_number} exceeds {MAX_CANDIDATE_LINE_BYTES} bytes"
            ));
        }
        let mut end = line.len();
        while end > 0 && matches!(line[end - 1], b'\n' | b'\r') {
            end -= 1;
        }
        let trimmed = &line[..end];
        if trimmed.iter().all(u8::is_ascii_whitespace) {
            return Err(format!(
                "speaker candidate file contains a blank line at {line_number}"
            ));
        }
        let record = serde_json::from_slice(trimmed).map_err(|error| {
            format!("invalid speaker candidate JSON at line {line_number}: {error}")
        })?;
        records.push(record);
        if records.len() > MAX_CANDIDATE_RECORDS {
            return Err(format!(
                "speaker candidate file exceeds {MAX_CANDIDATE_RECORDS} records"
            ));
        }
    }
    if records.is_empty() {
        return Err("speaker candidate file contains no records".to_string());
    }
    Ok(records)
}

fn validate_candidate(
    candidate: &SpeakerLearningCandidate,
    fingerprint: &str,
    dimension: usize,
) -> Result<(), String> {
    if candidate.format != CANDIDATE_FORMAT || candidate.version != CANDIDATE_VERSION {
        return Err(format!(
            "unsupported speaker candidate format/version for '{}'",
            candidate.candidate_id
        ));
    }
    if candidate.candidate_id.trim().is_empty()
        || candidate.speaker_id.trim().is_empty()
        || candidate.model_fingerprint != fingerprint
        || candidate.vector.len() != dimension
        || candidate.vector.iter().any(|value| !value.is_finite())
        || !candidate.duration_seconds.is_finite()
        || candidate.duration_seconds <= 0.0
        || !candidate.quality_score.is_finite()
        || !(0.0..=1.0).contains(&candidate.quality_score)
        || !candidate.score.is_finite()
    {
        return Err(format!(
            "speaker candidate '{}' is invalid or belongs to a different model",
            candidate.candidate_id
        ));
    }
    let mut normalized = candidate.vector.clone();
    normalize_embedding(&mut normalized)?;
    if normalized
        .iter()
        .zip(&candidate.vector)
        .any(|(normalized, original)| (normalized - original).abs() > 1e-3)
    {
        return Err(format!(
            "speaker candidate '{}' embedding is not normalized",
            candidate.candidate_id
        ));
    }
    Ok(())
}

fn identify_vector(
    index: &SpeakerIndex,
    vector: &[f32],
    source: String,
) -> Result<SpeakerIdentificationResult, String> {
    let ranked = index.ranked(vector)?;
    let matches = ranked
        .iter()
        .take(5)
        .map(|item| SpeakerMatch {
            speaker_id: item.speaker_id.clone(),
            score: item.score,
        })
        .collect::<Vec<_>>();
    let top = ranked.first();
    let second_score = ranked.get(1).map(|item| item.score);
    let margin = top
        .zip(second_score)
        .map(|(first, second)| first.score - second);
    let recognized = top.is_some_and(|first| {
        first.score >= index.thresholds.verification
            && margin.is_none_or(|value| value >= index.thresholds.identification_margin)
    });
    Ok(SpeakerIdentificationResult {
        recognized,
        speaker_id: recognized.then(|| {
            top.expect("recognized implies top match")
                .speaker_id
                .clone()
        }),
        score: top.map(|item| item.score),
        threshold: index.thresholds.verification,
        second_score,
        margin,
        required_margin: index.thresholds.identification_margin,
        source,
        matches,
    })
}

fn profile_summaries(index: &SpeakerIndex) -> Vec<SpeakerProfileSummary> {
    index
        .speakers
        .iter()
        .map(|profile| SpeakerProfileSummary {
            speaker_id: profile.speaker_id.clone(),
            observation_count: profile.observations.len(),
            total_duration_seconds: profile
                .observations
                .iter()
                .map(|observation| observation.duration_seconds)
                .sum(),
            observations: profile
                .observations
                .iter()
                .map(observation_summary)
                .collect(),
        })
        .collect()
}

fn observation_summary(observation: &SpeakerObservation) -> SpeakerObservationSummary {
    SpeakerObservationSummary {
        observation_id: observation.id.clone(),
        source: observation.source.clone(),
        duration_seconds: observation.duration_seconds,
        quality_score: observation.quality_score,
        created_unix_ms: observation.created_unix_ms,
    }
}

fn public_quality(quality: EngineSpeakerAudioQuality) -> SpeakerAudioQuality {
    SpeakerAudioQuality {
        duration_seconds: quality.duration_seconds,
        rms: quality.rms,
        clipping_fraction: quality.clipping_fraction,
        active_fraction: quality.active_fraction,
        score: quality.score,
    }
}

fn stored_thresholds(policy: &SpeakerThresholdPolicy) -> StoredThresholds {
    StoredThresholds {
        verification: policy.verification,
        identification_margin: policy.identification_margin,
        enrollment: policy.enrollment,
        auto_learning: policy.auto_learning,
        diarization_cluster: policy.diarization_cluster,
    }
}

fn apply_threshold_overrides(
    thresholds: &mut StoredThresholds,
    overrides: ThresholdOverrides,
) -> Result<(), String> {
    if let Some(value) = overrides.verification {
        validate_score("speaker threshold", value)?;
        thresholds.verification = value;
    }
    if let Some(value) = overrides.identification_margin {
        validate_margin(value)?;
        thresholds.identification_margin = value;
    }
    if thresholds.auto_learning < thresholds.verification {
        return Err(format!(
            "speaker threshold {:.4} exceeds the model auto-learning threshold {:.4}; use a calibrated model/index configuration",
            thresholds.verification, thresholds.auto_learning
        ));
    }
    Ok(())
}

fn validate_score(name: &str, value: f32) -> Result<(), String> {
    if value.is_finite() && (-1.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(format!("{name} must be within -1..=1"))
    }
}

fn validate_margin(value: f32) -> Result<(), String> {
    if value.is_finite() && (0.0..=2.0).contains(&value) {
        Ok(())
    } else {
        Err("speaker margin must be within 0..=2".to_string())
    }
}

fn path_string(path: &Path, kind: &str) -> Result<String, String> {
    path.to_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{kind} path contains non-UTF8 characters"))
}

fn candidate_id(speaker_id: &str, vector: &[f32], source: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in speaker_id.bytes().chain(source.bytes()).chain(
        vector
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes()),
    ) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("candidate-{hash:016x}-{}", now_unix_ms())
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn detect_speech_regions(samples: &[f32], sample_rate: u32) -> Result<Vec<(usize, usize)>, String> {
    if samples.is_empty() {
        return Err("meeting audio is empty".to_string());
    }
    let frame_length = (sample_rate as usize * 30 / 1_000).max(1);
    let hop = (sample_rate as usize * 10 / 1_000).max(1);
    if samples.len() < frame_length {
        return Err("meeting audio is too short for speech segmentation".to_string());
    }
    let mut levels = Vec::new();
    for start in (0..=samples.len() - frame_length).step_by(hop) {
        let frame = &samples[start..start + frame_length];
        let rms =
            (frame.iter().map(|value| value * value).sum::<f32>() / frame.len() as f32).sqrt();
        levels.push(20.0 * rms.max(1e-8).log10());
    }
    let mut ordered = levels.clone();
    ordered.sort_by(f32::total_cmp);
    let noise = ordered[ordered.len() / 5];
    let peak = *ordered.last().expect("non-empty levels");
    if peak < -55.0 {
        return Err("meeting audio contains no usable speech-level signal".to_string());
    }
    let threshold = (noise + 10.0).min(peak - 6.0).max(-50.0);
    let active = levels
        .iter()
        .enumerate()
        .filter_map(|(index, level)| (*level >= threshold).then_some(index))
        .collect::<Vec<_>>();
    if active.is_empty() {
        return Err("meeting audio contains no detected speech".to_string());
    }
    let max_gap_frames = 30usize;
    let padding = sample_rate as usize / 10;
    let minimum = sample_rate as usize / 2;
    let mut regions = Vec::new();
    let mut first = active[0];
    let mut previous = active[0];
    for &frame in &active[1..] {
        if frame - previous > max_gap_frames {
            push_speech_region(
                &mut regions,
                first,
                previous,
                hop,
                frame_length,
                padding,
                minimum,
                samples.len(),
            );
            first = frame;
        }
        previous = frame;
    }
    push_speech_region(
        &mut regions,
        first,
        previous,
        hop,
        frame_length,
        padding,
        minimum,
        samples.len(),
    );
    if regions.is_empty() {
        return Err("meeting audio has no speech region of at least 0.5 seconds".to_string());
    }
    Ok(regions)
}

#[allow(clippy::too_many_arguments)]
fn push_speech_region(
    regions: &mut Vec<(usize, usize)>,
    first_frame: usize,
    last_frame: usize,
    hop: usize,
    frame_length: usize,
    padding: usize,
    minimum: usize,
    sample_count: usize,
) {
    let start = first_frame.saturating_mul(hop).saturating_sub(padding);
    let end = last_frame
        .saturating_mul(hop)
        .saturating_add(frame_length)
        .saturating_add(padding)
        .min(sample_count);
    if end.saturating_sub(start) >= minimum {
        regions.push((start, end));
    }
}

fn split_speech_regions(
    regions: &[(usize, usize)],
    sample_rate: u32,
    minimum_seconds: f32,
) -> Vec<(usize, usize)> {
    let target = (sample_rate as f32 * 2.5).ceil() as usize;
    let minimum = (sample_rate as f32 * minimum_seconds).ceil() as usize;
    let mut chunks = Vec::new();
    for &(region_start, region_end) in regions {
        let mut start = region_start;
        while region_end - start > target + minimum {
            chunks.push((start, start + target));
            start += target;
        }
        if region_end - start >= minimum {
            chunks.push((start, region_end));
        } else if let Some(last) = chunks.last_mut()
            && last.1 == start
        {
            last.1 = region_end;
        }
    }
    chunks
}

fn assign_cluster(
    clusters: &mut Vec<(Vec<f32>, usize)>,
    vector: &[f32],
    threshold: f32,
) -> Result<(String, Option<f32>), String> {
    let mut best: Option<(usize, f32)> = None;
    for (index, (centroid, _)) in clusters.iter().enumerate() {
        let score = cosine_similarity(vector, centroid)?;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((index, score));
        }
    }
    if let Some((index, score)) = best
        && score >= threshold
    {
        let (centroid, count) = &mut clusters[index];
        for (destination, value) in centroid.iter_mut().zip(vector) {
            *destination = (*destination * *count as f32 + *value) / (*count + 1) as f32;
        }
        *count += 1;
        normalize_embedding(centroid)?;
        return Ok((format!("unknown-{}", index + 1), Some(score)));
    }
    clusters.push((vector.to_vec(), 1));
    Ok((
        format!("unknown-{}", clusters.len()),
        best.map(|(_, score)| score),
    ))
}

fn merge_adjacent_segments(
    segments: Vec<SpeakerDiarizationSegment>,
) -> Vec<SpeakerDiarizationSegment> {
    let mut merged: Vec<SpeakerDiarizationSegment> = Vec::new();
    for segment in segments {
        if let Some(previous) = merged.last_mut()
            && previous.speaker_id == segment.speaker_id
            && previous.recognized == segment.recognized
            && segment.start_seconds - previous.end_seconds <= 0.35
        {
            previous.end_seconds = segment.end_seconds;
            previous.score = match (previous.score, segment.score) {
                (Some(left), Some(right)) => Some((left + right) * 0.5),
                (left, right) => left.or(right),
            };
            continue;
        }
        merged.push(segment);
    }
    merged
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::{
        CANDIDATE_FORMAT, CANDIDATE_VERSION, EMBEDDING_FORMAT, EMBEDDING_VERSION,
        SpeakerAudioQuality, SpeakerEmbedding, SpeakerIndexRuntime, SpeakerLearningCandidate,
        detect_speech_regions, identify_vector, read_candidates, read_embedding_inputs,
        split_speech_regions, write_candidate,
    };
    use crate::app::speaker_index::{SpeakerIndex, StoredThresholds};

    fn identification_index() -> SpeakerIndex {
        let mut index = SpeakerIndex::new(
            "fingerprint".to_string(),
            "model".to_string(),
            "speaker_xvector".to_string(),
            2,
            StoredThresholds {
                verification: 0.6,
                identification_margin: 0.05,
                enrollment: 0.4,
                auto_learning: 0.8,
                diarization_cluster: 0.5,
            },
        )
        .unwrap();
        index
            .enroll(
                "alice",
                vec![1.0, 0.0],
                "alice.wav".to_string(),
                2.0,
                1.0,
                false,
            )
            .unwrap();
        index
            .enroll(
                "bob",
                vec![0.8, 0.6],
                "bob.wav".to_string(),
                2.0,
                1.0,
                false,
            )
            .unwrap();
        index
    }

    #[test]
    fn vad_finds_signal_between_silence() {
        let sample_rate = 1_000;
        let mut samples = vec![0.0f32; 500];
        samples.extend((0..1_500).map(|index| (index as f32 * 0.13).sin() * 0.2));
        samples.extend(vec![0.0f32; 500]);
        let regions = detect_speech_regions(&samples, sample_rate).unwrap();
        assert_eq!(regions.len(), 1);
        assert!(regions[0].0 <= 500);
        assert!(regions[0].1 >= 2_000);
    }

    #[test]
    fn long_regions_split_without_short_tail() {
        let chunks = split_speech_regions(&[(0, 8_000)], 1_000, 1.0);
        assert_eq!(chunks, vec![(0, 2_500), (2_500, 5_000), (5_000, 8_000)]);
    }

    #[test]
    fn identification_requires_score_and_runner_up_margin() {
        let index = identification_index();
        let confident = identify_vector(&index, &[1.0, 0.0], "confident.wav".to_string()).unwrap();
        assert!(confident.recognized);
        assert_eq!(confident.speaker_id.as_deref(), Some("alice"));

        let ambiguous = identify_vector(
            &index,
            &[0.948_683_3, 0.316_227_76],
            "ambiguous.wav".to_string(),
        )
        .unwrap();
        assert!(!ambiguous.recognized);
        assert_eq!(ambiguous.speaker_id, None);
        assert!(ambiguous.margin.unwrap().abs() < 1e-5);

        let weak = identify_vector(&index, &[-1.0, 0.0], "weak.wav".to_string()).unwrap();
        assert!(!weak.recognized);
        assert!(weak.score.unwrap() < weak.threshold);
    }

    #[test]
    fn candidate_jsonl_round_trip_starts_unaccepted() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("candidates.jsonl");
        let candidate = SpeakerLearningCandidate {
            format: CANDIDATE_FORMAT.to_string(),
            version: CANDIDATE_VERSION,
            candidate_id: "candidate-1".to_string(),
            accepted: false,
            speaker_id: "alice".to_string(),
            model_fingerprint: "fingerprint".to_string(),
            source: "meeting.wav#1.000-3.000".to_string(),
            duration_seconds: 2.0,
            quality_score: 0.9,
            score: 0.91,
            margin: Some(0.2),
            vector: vec![1.0, 0.0],
        };
        write_candidate(&path, &candidate).unwrap();
        let records = read_candidates(&path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].candidate_id, "candidate-1");
        assert!(!records[0].accepted);
    }

    #[test]
    fn exported_embedding_jsonl_can_be_read_back() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("embedding.jsonl");
        let embedding = SpeakerEmbedding {
            format: EMBEDDING_FORMAT.to_string(),
            version: EMBEDDING_VERSION,
            model_name: "model".to_string(),
            model_architecture: "speaker_xvector".to_string(),
            model_fingerprint: "fingerprint".to_string(),
            dimension: 2,
            source: "voice.wav".to_string(),
            quality: SpeakerAudioQuality {
                duration_seconds: 2.0,
                rms: 0.1,
                clipping_fraction: 0.0,
                active_fraction: 1.0,
                score: 1.0,
            },
            vector: vec![1.0, 0.0],
        };
        std::fs::write(&path, serde_json::to_vec(&embedding).unwrap()).unwrap();
        let records = read_embedding_inputs(&[path.display().to_string()]).unwrap();
        assert_eq!(records, vec![embedding]);
    }

    #[test]
    fn retained_index_runtime_identifies_and_mutates_without_a_model() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("speakers.spkidx");
        identification_index().save_atomic(&path).unwrap();
        let mut runtime = SpeakerIndexRuntime::load(&path).unwrap();
        let alice = SpeakerEmbedding {
            format: EMBEDDING_FORMAT.to_string(),
            version: EMBEDDING_VERSION,
            model_name: "model".to_string(),
            model_architecture: "speaker_xvector".to_string(),
            model_fingerprint: "fingerprint".to_string(),
            dimension: 2,
            source: "later-alice.wav".to_string(),
            quality: SpeakerAudioQuality {
                duration_seconds: 2.0,
                rms: 0.1,
                clipping_fraction: 0.0,
                active_fraction: 1.0,
                score: 1.0,
            },
            vector: vec![0.99, 0.141_067_36],
        };
        let identified = runtime.identify_embedding(&alice).unwrap();
        assert_eq!(identified.speaker_id.as_deref(), Some("alice"));
        let enrolled = runtime.enroll_embedding("alice", &alice, false).unwrap();
        assert_eq!(enrolled.observation_count, 2);
        assert_eq!(runtime.list_profiles()[0].observation_count, 2);
        runtime
            .remove_observation(&enrolled.observation_id)
            .unwrap();
        assert_eq!(runtime.list_profiles()[0].observation_count, 1);
    }
}
