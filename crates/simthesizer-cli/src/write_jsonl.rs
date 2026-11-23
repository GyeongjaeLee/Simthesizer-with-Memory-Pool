use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::Result;
use tempfile::NamedTempFile;

use simthesizer_core::types::RequestResult;

/// Result rows in completion order in a temporary file beside the output;
/// `finish` copies them sorted by request ID. Memory holds one index entry per
/// row, and a run that never finishes leaves no result file.
pub struct ResultsWriter {
    file: NamedTempFile,
    writer: BufWriter<File>,
    /// (request ID, byte offset, byte length) per row.
    rows: Vec<(u64, u64, u64)>,
    offset: u64,
    line: Vec<u8>,
}

impl ResultsWriter {
    pub fn new(path: &Path) -> Result<Self> {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let file = NamedTempFile::new_in(parent)?;
        let writer = BufWriter::new(file.as_file().try_clone()?);
        Ok(Self {
            file,
            writer,
            rows: Vec::new(),
            offset: 0,
            line: Vec::new(),
        })
    }

    pub fn push(&mut self, row: &RequestResult) -> Result<()> {
        self.line.clear();
        serde_json::to_writer(&mut self.line, row)?;
        self.line.push(b'\n');
        self.writer.write_all(&self.line)?;
        let len = self.line.len() as u64;
        self.rows.push((row.request_id, self.offset, len));
        self.offset += len;
        Ok(())
    }

    pub fn finish(mut self, path: &Path) -> Result<()> {
        self.writer.flush()?;
        self.rows.sort_by_key(|&(id, _, _)| id);
        let source = self.file.as_file();
        let mut out = BufWriter::new(File::create(path)?);
        for &(_, offset, len) in &self.rows {
            self.line.resize(len as usize, 0);
            source.read_exact_at(&mut self.line, offset)?;
            out.write_all(&self.line)?;
        }
        out.flush()?;
        Ok(())
    }
}
