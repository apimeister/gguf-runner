mod decode;
mod features;

pub(crate) use decode::{
    decode_audio_file, load_audio_chunk_samples, prepare_audios_for_multimodal,
    probe_audios_for_multimodal,
};
pub(crate) use features::extract_whisper_log_mel_windows;

#[derive(Debug)]
pub(crate) struct DecodedAudio {
    pub(crate) source_sample_rate: u32,
    pub(crate) source_channels: u16,
    pub(crate) sample_rate: u32,
    pub(crate) samples_mono_f32: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedAudioChunk {
    pub(crate) start_sample: usize,
    pub(crate) sample_count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedAudioFeatureWindow {
    pub(crate) start_frame: usize,
    pub(crate) valid_frames: usize,
    pub(crate) padded_frames: usize,
    pub(crate) mel_bins: usize,
    /// Mel-major storage: `[mel_bins][padded_frames]`.
    pub(crate) data_mel_major: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PreparedAudioFeatureWindowPlan {
    pub(crate) start_frame: usize,
    pub(crate) valid_frames: usize,
    pub(crate) padded_frames: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedAudioInputPlan {
    pub(crate) path: String,
    pub(crate) source_sample_rate: u32,
    pub(crate) source_channels: u16,
    pub(crate) target_sample_rate: u32,
    pub(crate) target_sample_count: usize,
    pub(crate) feature_windows: Vec<PreparedAudioFeatureWindowPlan>,
}

#[derive(Debug)]
pub(crate) struct PreparedAudioTensor {
    pub(crate) path: String,
    pub(crate) source_sample_rate: u32,
    pub(crate) source_channels: u16,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
    pub(crate) total_samples: usize,
    pub(crate) chunks: Vec<PreparedAudioChunk>,
    pub(crate) feature_windows: Vec<PreparedAudioFeatureWindow>,
    pub(crate) data_mono_f32: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AudioDecodeConfig {
    pub(crate) target_sample_rate: u32,
    pub(crate) max_output_samples: usize,
    pub(crate) chunk_size_samples: usize,
    pub(crate) max_file_bytes: usize,
    pub(crate) max_channels: u16,
    pub(crate) max_source_sample_rate: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WhisperLogMelConfig {
    pub(crate) sample_rate: u32,
    pub(crate) fft_length: usize,
    pub(crate) window_length: usize,
    pub(crate) hop_length: usize,
    pub(crate) mel_bins: usize,
    pub(crate) mel_floor: f32,
    pub(crate) max_window_frames: usize,
    pub(crate) frame_chunk_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AudioPreprocessConfig {
    pub(crate) decode: AudioDecodeConfig,
    pub(crate) features: WhisperLogMelConfig,
}
