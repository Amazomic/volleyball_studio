use super::*;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

pub(super) fn directory(root: &Path, identity: &Value) -> Result<PathBuf, String> {
    storage::safe_folder(root)?;
    let parent = root.join("phase1_checkpoints");
    let ensure = |path: &Path| match fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => storage::safe_folder(path),
        Err(e) => Err(e.to_string()),
    };
    ensure(&parent)?;
    let key = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(identity).map_err(|e| e.to_string())?)
    );
    let path = parent.join(key);
    ensure(&path)?;
    let manifest = path.join("identity.json");
    match storage::read_json(&manifest).map_err(|e| format!("{e:?}"))? {
        Some(saved) if saved != *identity => {
            return Err("Phase 1 checkpoint identity mismatch".into())
        }
        Some(_) => (),
        None => storage::write_json_atomic(&manifest, identity).map_err(|e| format!("{e:?}"))?,
    }
    Ok(path)
}

pub(super) fn model_digest(repo: &Path, file: &str) -> Result<String, String> {
    use std::io::Read;
    let mut input = File::open(repo.join("models").join(file))
        .map_err(|e| format!("Open checkpoint model {file}: {e}"))?;
    let mut hash = Sha256::new();
    let mut chunk = vec![0u8; 1024 * 1024];
    loop {
        let len = input.read(&mut chunk).map_err(|e| e.to_string())?;
        if len == 0 {
            break;
        }
        hash.update(&chunk[..len]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

const MAX_CHECKPOINT_ROW: usize = storage::MAX_JSONL_ROW_BYTES * 2 + 4096;
const MAX_CHECKPOINT_FILE: u64 = 64 * 1024 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedWindow {
    first_frame: u64,
    payload: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SavedDetection {
    frame_index: u64,
    class_name: String,
    score: f32,
    bbox_xyxy_norm: [f32; 4],
    query_index: Option<usize>,
    touch_probability: Option<f32>,
    head_point_norm: Option<[f32; 2]>,
}

impl SavedDetection {
    pub(super) fn into_raw(self, class: &'static str) -> Result<RawDetection, String> {
        if self.class_name != class {
            return Err("Detector checkpoint class mismatch".into());
        }
        Ok(RawDetection {
            frame_index: self.frame_index,
            class_name: class,
            score: self.score,
            bbox_xyxy_norm: self.bbox_xyxy_norm,
            query_index: self.query_index,
            touch_probability: self.touch_probability,
            head_point_norm: self.head_point_norm,
        })
    }
}

pub(super) struct DetectionJournal {
    file: File,
    reader: Option<BufReader<File>>,
    class: &'static str,
    next_frame: u64,
    valid_bytes: u64,
    pending: bool,
    replayed: Option<Vec<RawDetection>>,
}

impl DetectionJournal {
    pub(super) fn open(folder: &Path, name: &str, class: &'static str) -> Result<Self, String> {
        storage::safe_folder(folder)?;
        let path = folder.join(name);
        let file = match fs::symlink_metadata(&path) {
            Ok(_) => {
                storage::safe_file(folder, name, MAX_CHECKPOINT_FILE)?;
                OpenOptions::new().write(true).read(true).open(&path)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => OpenOptions::new()
                .write(true)
                .read(true)
                .create_new(true)
                .open(&path),
            Err(e) => return Err(format!("Inspect detector checkpoint: {e}")),
        }
        .map_err(|e| format!("Open detector checkpoint: {e}"))?;
        let reader = BufReader::new(File::open(&path).map_err(|e| e.to_string())?);
        Ok(Self {
            file,
            reader: Some(reader),
            class,
            next_frame: 0,
            valid_bytes: 0,
            pending: false,
            replayed: None,
        })
    }

    fn read_line(reader: &mut BufReader<File>) -> Result<Option<Vec<u8>>, String> {
        let mut line = Vec::new();
        loop {
            let chunk = reader.fill_buf().map_err(|e| e.to_string())?;
            if chunk.is_empty() {
                return Ok(None);
            }
            let count = chunk
                .iter()
                .position(|&b| b == b'\n')
                .map_or(chunk.len(), |pos| pos + 1);
            if line.len() + count > MAX_CHECKPOINT_ROW {
                return Err("Detector checkpoint row exceeds limit".into());
            }
            let complete = chunk[count - 1] == b'\n';
            line.extend_from_slice(&chunk[..count]);
            reader.consume(count);
            if complete {
                return Ok(Some(line));
            }
        }
    }

    fn validate(&self, first: u64, records: &[RawDetection]) -> Result<(), String> {
        let end = first
            .checked_add(9)
            .ok_or("Detector checkpoint frame overflow")?;
        if records.iter().any(|row| {
            row.class_name != self.class
                || !(first..end).contains(&row.frame_index)
                || !row.score.is_finite()
                || !row.bbox_xyxy_norm.iter().all(|n| n.is_finite())
                || row.touch_probability.is_some_and(|n| !n.is_finite())
                || row
                    .head_point_norm
                    .is_some_and(|p| !p.iter().all(|n| n.is_finite()))
        }) {
            return Err("Detector checkpoint contains invalid observations".into());
        }
        Ok(())
    }

    pub(super) fn cached(&mut self, first_frame: u64) -> Result<Option<Vec<RawDetection>>, String> {
        if first_frame != self.next_frame || self.pending {
            return Err("Detector checkpoint windows are not contiguous".into());
        }
        let line = self
            .reader
            .as_mut()
            .map(Self::read_line)
            .transpose()?
            .flatten();
        let records = if let Some(line) = line {
            let saved: SavedWindow = serde_json::from_slice(&line)
                .map_err(|e| format!("Corrupt detector checkpoint: {e}"))?;
            if saved.first_frame != first_frame
                || saved.sha256 != format!("{:x}", Sha256::digest(saved.payload.as_bytes()))
            {
                return Err("Detector checkpoint window or checksum mismatch".into());
            }
            let rows: Vec<SavedDetection> = serde_json::from_str(&saved.payload)
                .map_err(|e| format!("Invalid detector checkpoint observations: {e}"))?;
            let records = rows
                .into_iter()
                .map(|row| row.into_raw(self.class))
                .collect::<Result<Vec<_>, String>>()?;
            self.validate(first_frame, &records)?;
            self.valid_bytes += line.len() as u64;
            Some(records)
        } else {
            if self.reader.take().is_some() {
                let len = self.file.metadata().map_err(|e| e.to_string())?.len();
                if len != self.valid_bytes {
                    self.file
                        .set_len(self.valid_bytes)
                        .map_err(|e| format!("Discard incomplete detector window: {e}"))?;
                    self.file.sync_all().map_err(|e| e.to_string())?;
                }
                self.file
                    .seek(SeekFrom::Start(self.valid_bytes))
                    .map_err(|e| e.to_string())?;
            }
            None
        };
        self.pending = true;
        self.replayed = records.clone();
        Ok(records)
    }

    pub(super) fn commit(
        &mut self,
        first_frame: u64,
        records: &[RawDetection],
    ) -> Result<(), String> {
        if !self.pending || first_frame != self.next_frame {
            return Err("Detector checkpoint commit has no matching window".into());
        }
        self.validate(first_frame, records)?;
        if let Some(cached) = self.replayed.take() {
            if cached != records {
                return Err("Replayed detector observations changed".into());
            }
        } else {
            let payload = serde_json::to_string(records).map_err(|e| e.to_string())?;
            let saved = SavedWindow {
                first_frame,
                sha256: format!("{:x}", Sha256::digest(payload.as_bytes())),
                payload,
            };
            let mut line = serde_json::to_vec(&saved).map_err(|e| e.to_string())?;
            line.push(b'\n');
            if line.len() > MAX_CHECKPOINT_ROW
                || self.valid_bytes + line.len() as u64 > MAX_CHECKPOINT_FILE
            {
                return Err("Detector checkpoint exceeds storage limit".into());
            }
            self.file
                .write_all(&line)
                .map_err(|e| format!("Write detector checkpoint: {e}"))?;
            self.file
                .sync_data()
                .map_err(|e| format!("Sync detector checkpoint: {e}"))?;
            self.valid_bytes += line.len() as u64;
        }
        self.next_frame = first_frame
            .checked_add(9)
            .ok_or("Detector checkpoint frame overflow")?;
        self.pending = false;
        Ok(())
    }

    pub(super) fn finish(&mut self) -> Result<(), String> {
        if self.pending && self.replayed.is_some() {
            return Err("Detector checkpoint has an unconsumed complete window".into());
        }
        if let Some(reader) = self.reader.as_mut() {
            if Self::read_line(reader)?.is_some() {
                return Err("Detector checkpoint extends beyond the decoded video".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_replays_completed_windows_including_empty_and_discards_torn_tail() {
        let folder =
            std::env::temp_dir().join(format!("volley-detector-checkpoint-{}", Uuid::new_v4()));
        fs::create_dir(&folder).unwrap();
        let name = "players.jsonl.partial";
        let record = RawDetection {
            frame_index: 4,
            class_name: "player",
            score: 0.7,
            bbox_xyxy_norm: [0.1, 0.2, 0.5, 0.7],
            query_index: Some(0),
            touch_probability: None,
            head_point_norm: None,
        };
        {
            let mut journal = DetectionJournal::open(&folder, name, "player").unwrap();
            assert!(journal.cached(0).unwrap().is_none());
            journal.commit(0, std::slice::from_ref(&record)).unwrap();
            assert!(journal.cached(9).unwrap().is_none());
            journal.commit(9, &[]).unwrap();
        }
        let file = folder.join(name);
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(&file)
            .unwrap()
            .write_all(b"{\"first_frame\":18")
            .unwrap();
        {
            let mut journal = DetectionJournal::open(&folder, name, "player").unwrap();
            assert_eq!(journal.cached(0).unwrap(), Some(vec![record.clone()]));
            journal.commit(0, std::slice::from_ref(&record)).unwrap();
            assert_eq!(journal.cached(9).unwrap(), Some(vec![]));
            journal.commit(9, &[]).unwrap();
            assert!(journal.cached(18).unwrap().is_none());
            journal.commit(18, &[]).unwrap();
        }
        {
            let mut journal = DetectionJournal::open(&folder, name, "player").unwrap();
            assert_eq!(journal.cached(0).unwrap(), Some(vec![record.clone()]));
            journal.commit(0, std::slice::from_ref(&record)).unwrap();
            assert_eq!(journal.cached(9).unwrap(), Some(vec![]));
            journal.commit(9, &[]).unwrap();
            assert_eq!(journal.cached(18).unwrap(), Some(vec![]));
        }
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn checkpoint_rejects_complete_corrupt_rows_and_noncontiguous_frames() {
        let folder =
            std::env::temp_dir().join(format!("volley-detector-checkpoint-{}", Uuid::new_v4()));
        fs::create_dir(&folder).unwrap();
        let name = "players.jsonl.partial";
        fs::write(folder.join(name), b"{}\n").unwrap();
        {
            let mut journal = DetectionJournal::open(&folder, name, "player").unwrap();
            assert!(journal.cached(0).is_err());
        }
        fs::remove_file(folder.join(name)).unwrap();
        {
            let mut journal = DetectionJournal::open(&folder, name, "player").unwrap();
            assert!(journal.cached(9).is_err());
        }
        fs::remove_dir_all(folder).unwrap();
    }
}
