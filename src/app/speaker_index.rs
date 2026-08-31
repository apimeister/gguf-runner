use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::engine::speaker::{cosine_similarity, normalize_embedding};

const SPEAKER_INDEX_FORMAT: &str = "gguf-runner-speaker-index";
const SPEAKER_INDEX_VERSION: u32 = 1;
const MAX_INDEX_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SPEAKERS: usize = 10_000;
const MAX_OBSERVATIONS: usize = 100_000;
const MAX_EMBEDDING_DIM: usize = 8_192;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct StoredThresholds {
    pub(super) verification: f32,
    pub(super) identification_margin: f32,
    pub(super) enrollment: f32,
    pub(super) auto_learning: f32,
    pub(super) diarization_cluster: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct SpeakerObservation {
    pub(super) id: String,
    pub(super) vector: Vec<f32>,
    pub(super) source: String,
    pub(super) duration_seconds: f32,
    pub(super) quality_score: f32,
    pub(super) created_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct SpeakerProfile {
    pub(super) speaker_id: String,
    pub(super) centroid: Vec<f32>,
    pub(super) observations: Vec<SpeakerObservation>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct SpeakerIndex {
    format: String,
    version: u32,
    pub(super) model_fingerprint: String,
    pub(super) model_name: String,
    pub(super) model_architecture: String,
    pub(super) embedding_dimension: usize,
    pub(super) thresholds: StoredThresholds,
    pub(super) speakers: Vec<SpeakerProfile>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RankedSpeaker {
    pub(super) speaker_id: String,
    pub(super) score: f32,
}

impl SpeakerIndex {
    pub(super) fn new(
        model_fingerprint: String,
        model_name: String,
        model_architecture: String,
        embedding_dimension: usize,
        thresholds: StoredThresholds,
    ) -> Result<Self, String> {
        let index = Self {
            format: SPEAKER_INDEX_FORMAT.to_string(),
            version: SPEAKER_INDEX_VERSION,
            model_fingerprint,
            model_name,
            model_architecture,
            embedding_dimension,
            thresholds,
            speakers: Vec::new(),
        };
        index.validate()?;
        Ok(index)
    }

    pub(super) fn load(path: &Path) -> Result<Self, String> {
        let metadata = fs::metadata(path)
            .map_err(|error| format!("cannot read speaker index '{}': {error}", path.display()))?;
        if !metadata.is_file() {
            return Err(format!(
                "speaker index path is not a file: {}",
                path.display()
            ));
        }
        if metadata.len() > MAX_INDEX_BYTES {
            return Err(format!(
                "speaker index '{}' exceeds {} bytes",
                path.display(),
                MAX_INDEX_BYTES
            ));
        }
        let bytes = fs::read(path)
            .map_err(|error| format!("cannot read speaker index '{}': {error}", path.display()))?;
        let index: Self = serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "invalid speaker index JSON in '{}': {error}",
                path.display()
            )
        })?;
        index.validate()?;
        Ok(index)
    }

    pub(super) fn save_atomic(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        if !parent.exists() {
            return Err(format!(
                "speaker index directory does not exist: {}",
                parent.display()
            ));
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("cannot serialize speaker index: {error}"))?;
        if bytes.len() as u64 > MAX_INDEX_BYTES {
            return Err(format!(
                "speaker index would exceed {} bytes",
                MAX_INDEX_BYTES
            ));
        }
        let temporary = temporary_path(path);
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|error| {
                    format!(
                        "cannot create temporary speaker index '{}': {error}",
                        temporary.display()
                    )
                })?;
            file.write_all(&bytes).map_err(|error| {
                format!(
                    "cannot write temporary speaker index '{}': {error}",
                    temporary.display()
                )
            })?;
            file.sync_all().map_err(|error| {
                format!(
                    "cannot sync temporary speaker index '{}': {error}",
                    temporary.display()
                )
            })?;
            fs::rename(&temporary, path).map_err(|error| {
                format!(
                    "cannot replace speaker index '{}' with '{}': {error}",
                    path.display(),
                    temporary.display()
                )
            })
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    pub(super) fn validate_model(
        &self,
        fingerprint: &str,
        embedding_dimension: usize,
    ) -> Result<(), String> {
        if self.model_fingerprint != fingerprint {
            return Err(format!(
                "speaker index model fingerprint mismatch: index={}, loaded={fingerprint}; keep separate indexes per speaker model",
                self.model_fingerprint
            ));
        }
        if self.embedding_dimension != embedding_dimension {
            return Err(format!(
                "speaker index dimension mismatch: index={}, loaded={embedding_dimension}",
                self.embedding_dimension
            ));
        }
        Ok(())
    }

    pub(super) fn enroll(
        &mut self,
        speaker_id: &str,
        vector: Vec<f32>,
        source: String,
        duration_seconds: f32,
        quality_score: f32,
        force: bool,
    ) -> Result<(String, usize), String> {
        validate_speaker_id(speaker_id)?;
        validate_vector(&vector, self.embedding_dimension)?;
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err("speaker observation duration must be positive".to_string());
        }
        if !quality_score.is_finite() || !(0.0..=1.0).contains(&quality_score) {
            return Err("speaker observation quality must be within 0..=1".to_string());
        }
        if self.total_observations() >= MAX_OBSERVATIONS {
            return Err(format!(
                "speaker index reached the {MAX_OBSERVATIONS} observation limit"
            ));
        }
        let profile_index = match self
            .speakers
            .iter()
            .position(|profile| profile.speaker_id == speaker_id)
        {
            Some(index) => index,
            None => {
                if self.speakers.len() >= MAX_SPEAKERS {
                    return Err(format!(
                        "speaker index reached the {MAX_SPEAKERS} speaker limit"
                    ));
                }
                self.speakers.push(SpeakerProfile {
                    speaker_id: speaker_id.to_string(),
                    centroid: vector.clone(),
                    observations: Vec::new(),
                });
                self.speakers.len() - 1
            }
        };
        let profile = &mut self.speakers[profile_index];
        if !profile.observations.is_empty() {
            let score = cosine_similarity(&profile.centroid, &vector)?;
            if score < self.thresholds.enrollment && !force {
                return Err(format!(
                    "speaker enrollment sample for '{speaker_id}' scored {score:.4}, below the index enrollment threshold {:.4}; verify the identity or use --speaker-force-enroll",
                    self.thresholds.enrollment
                ));
            }
        }
        let observation_id = observation_id(speaker_id, &vector);
        if profile
            .observations
            .iter()
            .any(|observation| observation.id == observation_id)
        {
            return Err(format!(
                "speaker observation '{observation_id}' is already enrolled"
            ));
        }
        profile.observations.push(SpeakerObservation {
            id: observation_id.clone(),
            vector,
            source,
            duration_seconds,
            quality_score,
            created_unix_ms: now_unix_ms(),
        });
        rebuild_centroid(profile)?;
        Ok((observation_id, profile.observations.len()))
    }

    pub(super) fn remove_observation(&mut self, observation_id: &str) -> Result<String, String> {
        for profile_index in 0..self.speakers.len() {
            let Some(observation_index) = self.speakers[profile_index]
                .observations
                .iter()
                .position(|observation| observation.id == observation_id)
            else {
                continue;
            };
            let speaker_id = self.speakers[profile_index].speaker_id.clone();
            self.speakers[profile_index]
                .observations
                .remove(observation_index);
            if self.speakers[profile_index].observations.is_empty() {
                self.speakers.remove(profile_index);
            } else {
                rebuild_centroid(&mut self.speakers[profile_index])?;
            }
            return Ok(speaker_id);
        }
        Err(format!("speaker observation not found: {observation_id}"))
    }

    pub(super) fn remove_speaker(&mut self, speaker_id: &str) -> Result<usize, String> {
        let index = self
            .speakers
            .iter()
            .position(|profile| profile.speaker_id == speaker_id)
            .ok_or_else(|| format!("speaker profile not found: {speaker_id}"))?;
        Ok(self.speakers.remove(index).observations.len())
    }

    pub(super) fn ranked(&self, vector: &[f32]) -> Result<Vec<RankedSpeaker>, String> {
        validate_vector(vector, self.embedding_dimension)?;
        let mut ranked = self
            .speakers
            .iter()
            .map(|profile| {
                Ok(RankedSpeaker {
                    speaker_id: profile.speaker_id.clone(),
                    score: cosine_similarity(vector, &profile.centroid)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.speaker_id.cmp(&right.speaker_id))
        });
        Ok(ranked)
    }

    pub(super) fn speaker(&self, speaker_id: &str) -> Option<&SpeakerProfile> {
        self.speakers
            .iter()
            .find(|profile| profile.speaker_id == speaker_id)
    }

    pub(super) fn total_observations(&self) -> usize {
        self.speakers
            .iter()
            .map(|profile| profile.observations.len())
            .sum()
    }

    fn validate(&self) -> Result<(), String> {
        if self.format != SPEAKER_INDEX_FORMAT {
            return Err(format!(
                "unsupported speaker index format '{}'; expected '{SPEAKER_INDEX_FORMAT}'",
                self.format
            ));
        }
        if self.version != SPEAKER_INDEX_VERSION {
            return Err(format!(
                "unsupported speaker index version {}; expected {SPEAKER_INDEX_VERSION}",
                self.version
            ));
        }
        if self.model_fingerprint.trim().is_empty()
            || self.model_name.trim().is_empty()
            || self.model_architecture.trim().is_empty()
        {
            return Err("speaker index model identity is incomplete".to_string());
        }
        if self.embedding_dimension == 0 || self.embedding_dimension > MAX_EMBEDDING_DIM {
            return Err(format!(
                "speaker index embedding dimension {} is outside 1..={MAX_EMBEDDING_DIM}",
                self.embedding_dimension
            ));
        }
        validate_thresholds(&self.thresholds)?;
        if self.speakers.len() > MAX_SPEAKERS {
            return Err(format!(
                "speaker index contains more than {MAX_SPEAKERS} speakers"
            ));
        }
        if self.total_observations() > MAX_OBSERVATIONS {
            return Err(format!(
                "speaker index contains more than {MAX_OBSERVATIONS} observations"
            ));
        }
        let mut speaker_ids = BTreeSet::new();
        let mut observation_ids = BTreeSet::new();
        for (profile_index, profile) in self.speakers.iter().enumerate() {
            validate_speaker_id(&profile.speaker_id)?;
            if !speaker_ids.insert(profile.speaker_id.as_str()) {
                return Err(format!(
                    "speaker index contains duplicate profile '{}'",
                    profile.speaker_id
                ));
            }
            validate_vector(&profile.centroid, self.embedding_dimension)?;
            if profile.observations.is_empty() {
                return Err(format!(
                    "speaker profile {profile_index} has no observations"
                ));
            }
            for observation in &profile.observations {
                if observation.id.trim().is_empty() || observation.id.len() > 256 {
                    return Err("speaker observation has an invalid id".to_string());
                }
                if !observation_ids.insert(observation.id.as_str()) {
                    return Err(format!(
                        "speaker index contains duplicate observation '{}'",
                        observation.id
                    ));
                }
                if observation.source.len() > 16_384 {
                    return Err("speaker observation source is too long".to_string());
                }
                validate_vector(&observation.vector, self.embedding_dimension)?;
                if !observation.duration_seconds.is_finite()
                    || observation.duration_seconds <= 0.0
                    || !observation.quality_score.is_finite()
                    || !(0.0..=1.0).contains(&observation.quality_score)
                {
                    return Err(format!(
                        "speaker observation '{}' has invalid quality metadata",
                        observation.id
                    ));
                }
            }
            let mut rebuilt = profile.clone();
            rebuild_centroid(&mut rebuilt)?;
            if rebuilt
                .centroid
                .iter()
                .zip(&profile.centroid)
                .any(|(expected, stored)| (expected - stored).abs() > 1e-5)
            {
                return Err(format!(
                    "speaker profile '{}' centroid does not match its observations",
                    profile.speaker_id
                ));
            }
        }
        Ok(())
    }
}

fn validate_thresholds(thresholds: &StoredThresholds) -> Result<(), String> {
    for (name, value) in [
        ("verification", thresholds.verification),
        ("enrollment", thresholds.enrollment),
        ("auto learning", thresholds.auto_learning),
        ("diarization cluster", thresholds.diarization_cluster),
    ] {
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(format!(
                "speaker index {name} threshold must be within -1..=1"
            ));
        }
    }
    if !thresholds.identification_margin.is_finite()
        || !(0.0..=2.0).contains(&thresholds.identification_margin)
    {
        return Err("speaker index identification margin must be within 0..=2".to_string());
    }
    if thresholds.auto_learning < thresholds.verification {
        return Err(
            "speaker index auto-learning threshold must be at least verification".to_string(),
        );
    }
    Ok(())
}

fn validate_speaker_id(speaker_id: &str) -> Result<(), String> {
    if speaker_id.trim() != speaker_id || speaker_id.is_empty() || speaker_id.len() > 256 {
        return Err(
            "speaker id must be 1..=256 bytes without leading or trailing whitespace".to_string(),
        );
    }
    if speaker_id.chars().any(char::is_control) {
        return Err("speaker id must not contain control characters".to_string());
    }
    Ok(())
}

fn validate_vector(vector: &[f32], expected_dimension: usize) -> Result<(), String> {
    if vector.len() != expected_dimension {
        return Err(format!(
            "speaker embedding has dimension {}, expected {expected_dimension}",
            vector.len()
        ));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("speaker embedding contains non-finite values".to_string());
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 1e-3 {
        return Err(format!(
            "speaker embedding is not L2-normalized (norm={norm:.6})"
        ));
    }
    Ok(())
}

fn rebuild_centroid(profile: &mut SpeakerProfile) -> Result<(), String> {
    let dimension = profile
        .observations
        .first()
        .ok_or_else(|| "cannot rebuild an empty speaker profile".to_string())?
        .vector
        .len();
    let mut centroid = vec![0.0f32; dimension];
    let mut total_weight = 0.0f32;
    for observation in &profile.observations {
        let weight =
            (observation.duration_seconds.min(30.0) * observation.quality_score.max(0.1)).max(0.1);
        for (destination, value) in centroid.iter_mut().zip(&observation.vector) {
            *destination += value * weight;
        }
        total_weight += weight;
    }
    for value in &mut centroid {
        *value /= total_weight;
    }
    normalize_embedding(&mut centroid)?;
    profile.centroid = centroid;
    Ok(())
}

fn observation_id(speaker_id: &str, vector: &[f32]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in speaker_id.bytes().chain(
        vector
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes()),
    ) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("obs-{hash:016x}")
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn temporary_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("speakers.spkidx");
    parent.join(format!(
        ".{name}.tmp-{}-{}",
        std::process::id(),
        now_unix_ms()
    ))
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::{SpeakerIndex, StoredThresholds};

    fn thresholds() -> StoredThresholds {
        StoredThresholds {
            verification: 0.6,
            identification_margin: 0.05,
            enrollment: 0.4,
            auto_learning: 0.8,
            diarization_cluster: 0.5,
        }
    }

    fn index() -> SpeakerIndex {
        SpeakerIndex::new(
            "fingerprint".to_string(),
            "model".to_string(),
            "speaker_xvector".to_string(),
            2,
            thresholds(),
        )
        .unwrap()
    }

    #[test]
    fn enrollment_is_additive_and_rebuilds_centroid() {
        let mut index = index();
        index
            .enroll(
                "alice",
                vec![1.0, 0.0],
                "a.wav".to_string(),
                2.0,
                1.0,
                false,
            )
            .unwrap();
        index
            .enroll(
                "alice",
                vec![0.8, 0.6],
                "b.wav".to_string(),
                2.0,
                1.0,
                false,
            )
            .unwrap();
        let profile = index.speaker("alice").unwrap();
        assert_eq!(profile.observations.len(), 2);
        assert!(profile.centroid[0] > profile.centroid[1]);
        assert!((profile.centroid.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn low_similarity_addition_requires_force() {
        let mut index = index();
        index
            .enroll(
                "alice",
                vec![1.0, 0.0],
                "a.wav".to_string(),
                2.0,
                1.0,
                false,
            )
            .unwrap();
        assert!(
            index
                .enroll(
                    "alice",
                    vec![0.0, 1.0],
                    "b.wav".to_string(),
                    2.0,
                    1.0,
                    false
                )
                .is_err()
        );
        assert!(
            index
                .enroll("alice", vec![0.0, 1.0], "b.wav".to_string(), 2.0, 1.0, true)
                .is_ok()
        );
    }

    #[test]
    fn index_round_trip_preserves_observations() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("speakers.spkidx");
        let mut index = index();
        index
            .enroll(
                "alice",
                vec![1.0, 0.0],
                "a.wav".to_string(),
                2.0,
                0.9,
                false,
            )
            .unwrap();
        index.save_atomic(&path).unwrap();
        let loaded = SpeakerIndex::load(&path).unwrap();
        assert_eq!(loaded, index);
    }

    #[test]
    fn observation_removal_is_recoverable() {
        let mut index = index();
        let (id, _) = index
            .enroll(
                "alice",
                vec![1.0, 0.0],
                "a.wav".to_string(),
                2.0,
                1.0,
                false,
            )
            .unwrap();
        assert_eq!(index.remove_observation(&id).unwrap(), "alice");
        assert!(index.speakers.is_empty());
    }

    #[test]
    fn validation_rejects_a_centroid_unrelated_to_observations() {
        let mut index = index();
        index
            .enroll(
                "alice",
                vec![1.0, 0.0],
                "a.wav".to_string(),
                2.0,
                1.0,
                false,
            )
            .unwrap();
        index.speakers[0].centroid = vec![0.0, 1.0];
        assert!(index.validate().unwrap_err().contains("centroid"));
    }

    #[test]
    fn validation_rejects_duplicate_speaker_profiles() {
        let mut index = index();
        index
            .enroll(
                "alice",
                vec![1.0, 0.0],
                "a.wav".to_string(),
                2.0,
                1.0,
                false,
            )
            .unwrap();
        index.speakers.push(index.speakers[0].clone());
        assert!(index.validate().unwrap_err().contains("duplicate profile"));
    }
}
