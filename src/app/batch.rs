use serde::de::DeserializeOwned;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub(crate) const MAX_MANIFEST_LINE_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub(crate) struct JsonlBatchManifest {
    file: File,
    base_dir: PathBuf,
    record_count: usize,
    kind: &'static str,
}

impl JsonlBatchManifest {
    pub(crate) fn open_and_validate<T, F>(
        path: &Path,
        kind: &'static str,
        mut validate: F,
    ) -> Result<Self, String>
    where
        T: DeserializeOwned,
        F: FnMut(&T, usize) -> Result<(), String>,
    {
        let mut file = File::open(path)
            .map_err(|err| format!("open {kind} manifest '{}': {err}", path.display()))?;
        let record_count = {
            let mut reader = BufReader::new(&mut file);
            visit_records::<T, _, _>(&mut reader, kind, |record, line_number| {
                validate(&record, line_number)
            })?
        };
        if record_count == 0 {
            return Err(format!(
                "{kind} manifest '{}' contains no records",
                path.display()
            ));
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|err| format!("rewind {kind} manifest '{}': {err}", path.display()))?;

        let manifest_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let base_dir = if manifest_dir.is_absolute() {
            manifest_dir.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|err| format!("resolve current directory for {kind}: {err}"))?
                .join(manifest_dir)
        };

        Ok(Self {
            file,
            base_dir,
            record_count,
            kind,
        })
    }

    pub(crate) fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub(crate) fn visit<T, V, F>(&mut self, mut validate: V, visit: F) -> Result<(), String>
    where
        T: DeserializeOwned,
        V: FnMut(&T, usize) -> Result<(), String>,
        F: FnMut(T, usize) -> Result<(), String>,
    {
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|err| format!("rewind {} manifest: {err}", self.kind))?;
        let mut reader = BufReader::new(&mut self.file);
        let mut visit = visit;
        let processed = visit_records::<T, _, _>(&mut reader, self.kind, |record, line_number| {
            validate(&record, line_number)?;
            visit(record, line_number)
        })?;
        if processed != self.record_count {
            return Err(format!(
                "{} manifest changed after validation: expected {} records, found {processed}",
                self.kind, self.record_count
            ));
        }
        Ok(())
    }
}

fn visit_records<T, R, F>(reader: &mut R, kind: &str, mut visit: F) -> Result<usize, String>
where
    T: DeserializeOwned,
    R: BufRead,
    F: FnMut(T, usize) -> Result<(), String>,
{
    let mut line = String::new();
    let mut line_number = 0usize;
    let mut record_count = 0usize;
    loop {
        line.clear();
        // Read at most one byte past the limit so a manifest without newlines
        // cannot pull an unbounded line into memory before the check below.
        let bytes = reader
            .by_ref()
            .take(MAX_MANIFEST_LINE_BYTES as u64 + 1)
            .read_line(&mut line)
            .map_err(|err| format!("read {kind} manifest line {}: {err}", line_number + 1))?;
        if bytes == 0 {
            break;
        }
        line_number += 1;
        if bytes > MAX_MANIFEST_LINE_BYTES {
            return Err(format!(
                "{kind} manifest line {line_number} exceeds {MAX_MANIFEST_LINE_BYTES} bytes"
            ));
        }
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<T>(&line)
            .map_err(|err| format!("invalid {kind} manifest line {line_number}: {err}"))?;
        visit(record, line_number)?;
        record_count += 1;
    }
    Ok(record_count)
}
