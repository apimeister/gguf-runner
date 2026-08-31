use crate::app::batch::JsonlBatchManifest;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) const IMAGE_BATCH_CHUNK_SIZE: usize = 4;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageBatchRecord {
    id: String,
    path: String,
    prompt: String,
    #[serde(default)]
    system_prompt: Option<String>,
}

impl ImageBatchRecord {
    fn validate(&self, line_number: usize) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err(format!(
                "image batch manifest line {line_number}: `id` must not be empty"
            ));
        }
        if self.path.trim().is_empty() {
            return Err(format!(
                "image batch manifest line {line_number}: `path` must not be empty"
            ));
        }
        if self.prompt.trim().is_empty() {
            return Err(format!(
                "image batch manifest line {line_number}: `prompt` must not be empty"
            ));
        }
        Ok(())
    }
}

pub(crate) type ImageBatchManifest = JsonlBatchManifest;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ImageBatchSummary {
    pub(crate) succeeded: usize,
    pub(crate) failed: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ImageBatchJob {
    pub(crate) id: String,
    pub(crate) path: String,
    pub(crate) resolved_path: PathBuf,
    pub(crate) prompt: String,
    pub(crate) system_prompt: Option<String>,
}

#[derive(Serialize)]
struct ImageBatchOutput<'a> {
    id: &'a str,
    path: &'a str,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

pub(crate) fn open_and_validate_manifest(path: &Path) -> Result<ImageBatchManifest, String> {
    JsonlBatchManifest::open_and_validate::<ImageBatchRecord, _>(
        path,
        "image batch",
        ImageBatchRecord::validate,
    )
}

pub(crate) fn run_manifest<W, F>(
    manifest: &mut ImageBatchManifest,
    writer: &mut W,
    chunk_size: usize,
    mut generate: F,
) -> Result<ImageBatchSummary, String>
where
    W: Write,
    F: FnMut(&[ImageBatchJob]) -> Vec<Result<String, String>>,
{
    if chunk_size == 0 {
        return Err("image batch chunk size must be greater than zero".to_string());
    }
    let mut summary = ImageBatchSummary::default();
    let base_dir = manifest.base_dir().to_path_buf();
    let mut batch = Vec::with_capacity(chunk_size);
    manifest.visit::<ImageBatchRecord, _, _>(
        ImageBatchRecord::validate,
        |record, _line_number| {
            let input_path = Path::new(&record.path);
            let resolved_path = if input_path.is_absolute() {
                input_path.to_path_buf()
            } else {
                base_dir.join(input_path)
            };

            batch.push(ImageBatchJob {
                id: record.id,
                path: record.path,
                resolved_path,
                prompt: record.prompt,
                system_prompt: record.system_prompt,
            });
            if batch.len() == chunk_size {
                process_batch(&mut batch, writer, &mut summary, &mut generate)?;
            }
            Ok(())
        },
    )?;
    process_batch(&mut batch, writer, &mut summary, &mut generate)?;
    Ok(summary)
}

fn process_batch<W, F>(
    batch: &mut Vec<ImageBatchJob>,
    writer: &mut W,
    summary: &mut ImageBatchSummary,
    generate: &mut F,
) -> Result<(), String>
where
    W: Write,
    F: FnMut(&[ImageBatchJob]) -> Vec<Result<String, String>>,
{
    if batch.is_empty() {
        return Ok(());
    }
    let results = generate(batch);
    if results.len() != batch.len() {
        return Err(format!(
            "image batch generator returned {} result(s) for {} record(s)",
            results.len(),
            batch.len()
        ));
    }
    for (record, result) in batch.iter().zip(results) {
        match result {
            Ok(text) => {
                summary.succeeded += 1;
                write_output(
                    writer,
                    &ImageBatchOutput {
                        id: &record.id,
                        path: &record.path,
                        status: "ok",
                        text: Some(&text),
                        error: None,
                    },
                )?;
            }
            Err(error) => {
                summary.failed += 1;
                write_output(
                    writer,
                    &ImageBatchOutput {
                        id: &record.id,
                        path: &record.path,
                        status: "error",
                        text: None,
                        error: Some(&error),
                    },
                )?;
            }
        }
    }
    batch.clear();
    Ok(())
}

fn write_output<W: Write>(writer: &mut W, output: &ImageBatchOutput<'_>) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, output)
        .map_err(|err| format!("serialize image batch result: {err}"))?;
    writer
        .write_all(b"\n")
        .map_err(|err| format!("write image batch result: {err}"))?;
    writer
        .flush()
        .map_err(|err| format!("flush image batch result: {err}"))
}

#[cfg(test)]
mod tests {
    use super::{open_and_validate_manifest, run_manifest};
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
                "{\"id\":\"first\",\"path\":\"first.jpg\",\"prompt\":\"Describe\"}\n",
                "{\"id\":\"second\",\"path\":42,\"prompt\":\"Describe\"}\n"
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
            "{\"id\":\"one\",\"path\":\"one.jpg\",\"prompt\":\"Describe\",\"extra\":true}\n",
        )
        .expect("write manifest");
        assert!(open_and_validate_manifest(&unknown_path).is_err());

        let empty_path = temp.path().join("empty.jsonl");
        fs::write(
            &empty_path,
            "{\"id\":\"one\",\"path\":\"one.jpg\",\"prompt\":\" \"}\n",
        )
        .expect("write manifest");
        let error = open_and_validate_manifest(&empty_path).expect_err("empty prompt");
        assert!(error.contains("`prompt` must not be empty"));
    }

    #[test]
    fn preserves_order_resolves_paths_flushes_and_continues_after_errors() {
        let temp = tempdir().expect("tempdir");
        let manifest_path = temp.path().join("batch.jsonl");
        fs::write(
            &manifest_path,
            concat!(
                "{\"id\":\"good\",\"path\":\"images/good.jpg\",\"prompt\":\"Describe\",\"system_prompt\":\"Be brief\"}\n",
                "{\"id\":\"bad\",\"path\":\"images/bad.jpg\",\"prompt\":\"Read\"}\n",
                "{\"id\":\"last\",\"path\":\"images/last.jpg\",\"prompt\":\"Classify\"}\n"
            ),
        )
        .expect("write manifest");
        let mut manifest = open_and_validate_manifest(&manifest_path).expect("valid manifest");
        let mut writer = FlushCountingWriter::default();
        let mut calls = Vec::new();
        let mut chunk_lengths = Vec::new();

        let summary = run_manifest(&mut manifest, &mut writer, 2, |jobs| {
            chunk_lengths.push(jobs.len());
            jobs.iter()
                .map(|job| {
                    calls.push((
                        job.resolved_path.clone(),
                        job.prompt.clone(),
                        job.system_prompt.clone(),
                    ));
                    if job.resolved_path.ends_with("bad.jpg") {
                        Err("decode failed".to_string())
                    } else {
                        Ok(format!("answer for {}", job.prompt))
                    }
                })
                .collect()
        })
        .expect("run manifest");

        assert_eq!(summary.succeeded, 2);
        assert_eq!(summary.failed, 1);
        assert_eq!(writer.flushes, 3);
        assert_eq!(calls[0].0, temp.path().join("images/good.jpg"));
        assert_eq!(calls[0].1, "Describe");
        assert_eq!(calls[0].2.as_deref(), Some("Be brief"));
        assert_eq!(calls[1].2, None);
        assert_eq!(chunk_lengths, vec![2, 1]);

        let output = String::from_utf8(writer.bytes).expect("utf8 output");
        let lines = output.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("\"id\":\"good\""));
        assert!(lines[0].contains("\"status\":\"ok\""));
        assert!(lines[1].contains("\"id\":\"bad\""));
        assert!(lines[1].contains("\"status\":\"error\""));
        assert!(lines[2].contains("\"id\":\"last\""));
        assert!(lines[2].contains("\"status\":\"ok\""));
    }

    #[test]
    fn rejects_zero_sized_chunks() {
        let temp = tempdir().expect("tempdir");
        let manifest_path = temp.path().join("batch.jsonl");
        fs::write(
            &manifest_path,
            "{\"id\":\"one\",\"path\":\"one.jpg\",\"prompt\":\"Describe\"}\n",
        )
        .expect("write manifest");
        let mut manifest = open_and_validate_manifest(&manifest_path).expect("valid manifest");
        let mut writer = FlushCountingWriter::default();

        let error = run_manifest(&mut manifest, &mut writer, 0, |_| Vec::new())
            .expect_err("zero chunk size");
        assert!(error.contains("greater than zero"));
        assert!(writer.bytes.is_empty());
    }

    #[test]
    fn rejects_a_generator_result_count_mismatch() {
        let temp = tempdir().expect("tempdir");
        let manifest_path = temp.path().join("batch.jsonl");
        fs::write(
            &manifest_path,
            concat!(
                "{\"id\":\"one\",\"path\":\"one.jpg\",\"prompt\":\"Describe\"}\n",
                "{\"id\":\"two\",\"path\":\"two.jpg\",\"prompt\":\"Describe\"}\n"
            ),
        )
        .expect("write manifest");
        let mut manifest = open_and_validate_manifest(&manifest_path).expect("valid manifest");
        let mut writer = FlushCountingWriter::default();

        let error = run_manifest(&mut manifest, &mut writer, 2, |_| {
            vec![Ok("only one result".to_string())]
        })
        .expect_err("result count mismatch");
        assert!(error.contains("1 result(s) for 2 record(s)"));
        assert!(writer.bytes.is_empty());
    }
}
