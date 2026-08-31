use crate::app::batch::JsonlBatchManifest;
use crate::engine::types::AudioTranscriptionResult;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AudioBatchRecord {
    id: String,
    path: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    context: String,
}

impl AudioBatchRecord {
    fn validate(&self, line_number: usize) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err(format!(
                "audio batch manifest line {line_number}: `id` must not be empty"
            ));
        }
        if self.path.trim().is_empty() {
            return Err(format!(
                "audio batch manifest line {line_number}: `path` must not be empty"
            ));
        }
        if self
            .language
            .as_deref()
            .is_some_and(|language| language.trim().is_empty())
        {
            return Err(format!(
                "audio batch manifest line {line_number}: `language` must not be empty when present"
            ));
        }
        Ok(())
    }
}

pub(crate) type AudioBatchManifest = JsonlBatchManifest;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AudioBatchSummary {
    pub(crate) succeeded: usize,
    pub(crate) failed: usize,
}

#[derive(Serialize)]
struct AudioBatchOutput<'a> {
    id: &'a str,
    path: &'a str,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

pub(crate) fn open_and_validate_manifest(path: &Path) -> Result<AudioBatchManifest, String> {
    JsonlBatchManifest::open_and_validate::<AudioBatchRecord, _>(
        path,
        "audio batch",
        AudioBatchRecord::validate,
    )
}

pub(crate) fn run_manifest<W, F>(
    manifest: &mut AudioBatchManifest,
    writer: &mut W,
    mut transcribe: F,
) -> Result<AudioBatchSummary, String>
where
    W: Write,
    F: FnMut(&Path, &str, Option<&str>) -> Result<AudioTranscriptionResult, String>,
{
    let mut summary = AudioBatchSummary::default();
    let base_dir = manifest.base_dir().to_path_buf();
    manifest.visit::<AudioBatchRecord, _, _>(
        AudioBatchRecord::validate,
        |record, _line_number| {
            let input_path = Path::new(&record.path);
            let resolved_path = if input_path.is_absolute() {
                input_path.to_path_buf()
            } else {
                base_dir.join(input_path)
            };

            match transcribe(&resolved_path, &record.context, record.language.as_deref()) {
                Ok(result) => {
                    summary.succeeded += 1;
                    write_output(
                        writer,
                        &AudioBatchOutput {
                            id: &record.id,
                            path: &record.path,
                            status: "ok",
                            language: (!result.language.is_empty())
                                .then_some(result.language.as_str()),
                            text: Some(&result.transcript),
                            error: None,
                        },
                    )?;
                }
                Err(error) => {
                    summary.failed += 1;
                    write_output(
                        writer,
                        &AudioBatchOutput {
                            id: &record.id,
                            path: &record.path,
                            status: "error",
                            language: None,
                            text: None,
                            error: Some(&error),
                        },
                    )?;
                }
            }
            Ok(())
        },
    )?;
    Ok(summary)
}

fn write_output<W: Write>(writer: &mut W, output: &AudioBatchOutput<'_>) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, output)
        .map_err(|err| format!("serialize audio batch result: {err}"))?;
    writer
        .write_all(b"\n")
        .map_err(|err| format!("write audio batch result: {err}"))?;
    writer
        .flush()
        .map_err(|err| format!("flush audio batch result: {err}"))
}

#[cfg(test)]
mod tests {
    use super::{open_and_validate_manifest, run_manifest};
    use crate::engine::types::AudioTranscriptionResult;
    use std::fs;
    use std::io::{self, Write};
    use tempfile::tempdir;

    #[derive(Default)]
    struct FlushCountingWriter {
        bytes: Vec<u8>,
        flushes: usize,
    }

    impl Write for FlushCountingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    #[test]
    fn validates_the_entire_manifest_before_processing() {
        let temp = tempdir().expect("tempdir");
        let manifest_path = temp.path().join("batch.jsonl");
        fs::write(
            &manifest_path,
            concat!(
                "{\"id\":\"first\",\"path\":\"first.wav\"}\n",
                "{\"id\":\"second\",\"path\":42}\n"
            ),
        )
        .expect("write manifest");

        let error = open_and_validate_manifest(&manifest_path).expect_err("invalid manifest");
        assert!(error.contains("line 2"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_unknown_fields_and_empty_required_values() {
        let temp = tempdir().expect("tempdir");
        let unknown_path = temp.path().join("unknown.jsonl");
        fs::write(
            &unknown_path,
            "{\"id\":\"one\",\"path\":\"one.wav\",\"extra\":true}\n",
        )
        .expect("write manifest");
        assert!(open_and_validate_manifest(&unknown_path).is_err());

        let empty_path = temp.path().join("empty.jsonl");
        fs::write(&empty_path, "{\"id\":\" \",\"path\":\"one.wav\"}\n").expect("write manifest");
        let error = open_and_validate_manifest(&empty_path).expect_err("empty id");
        assert!(error.contains("`id` must not be empty"));
    }

    #[test]
    fn rejects_a_record_longer_than_the_line_limit() {
        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("huge.jsonl");
        let padding = "x".repeat(crate::app::batch::MAX_MANIFEST_LINE_BYTES);
        fs::write(
            &path,
            format!("{{\"id\":\"one\",\"path\":\"one.wav\",\"context\":\"{padding}\"}}\n"),
        )
        .expect("write manifest");

        let error = open_and_validate_manifest(&path).expect_err("oversized line");

        assert!(error.contains("exceeds"), "unexpected error: {error}");
    }

    #[test]
    fn preserves_order_resolves_relative_paths_and_flushes_each_result() {
        let temp = tempdir().expect("tempdir");
        let manifest_path = temp.path().join("batch.jsonl");
        fs::write(
            &manifest_path,
            concat!(
                "{\"id\":\"good\",\"path\":\"audio/good.wav\",\"language\":\"English\",\"context\":\"Everlock\"}\n",
                "{\"id\":\"bad\",\"path\":\"audio/bad.wav\"}\n"
            ),
        )
        .expect("write manifest");
        let mut manifest = open_and_validate_manifest(&manifest_path).expect("valid manifest");
        let mut writer = FlushCountingWriter::default();
        let mut calls = Vec::new();

        let summary = run_manifest(&mut manifest, &mut writer, |path, context, language| {
            calls.push((
                path.to_path_buf(),
                context.to_string(),
                language.map(str::to_string),
            ));
            if path.ends_with("bad.wav") {
                Err("decode failed".to_string())
            } else {
                Ok(AudioTranscriptionResult {
                    raw_output: "language English<asr_text>Hello".to_string(),
                    language: "English".to_string(),
                    transcript: "Hello".to_string(),
                })
            }
        })
        .expect("run manifest");

        assert_eq!(summary.succeeded, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(writer.flushes, 2);
        assert_eq!(calls[0].0, temp.path().join("audio/good.wav"));
        assert_eq!(calls[0].1, "Everlock");
        assert_eq!(calls[0].2.as_deref(), Some("English"));
        assert_eq!(calls[1].0, temp.path().join("audio/bad.wav"));

        let output = String::from_utf8(writer.bytes).expect("utf8 output");
        let lines = output.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0],
            "{\"id\":\"good\",\"path\":\"audio/good.wav\",\"status\":\"ok\",\"language\":\"English\",\"text\":\"Hello\"}"
        );
        assert_eq!(
            lines[1],
            "{\"id\":\"bad\",\"path\":\"audio/bad.wav\",\"status\":\"error\",\"error\":\"decode failed\"}"
        );
    }
}
