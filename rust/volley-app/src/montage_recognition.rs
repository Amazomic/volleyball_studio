use crate::{
    montage_material::MaterialBinding,
    native_rally_evidence::{inspect_project, read_manifest},
    projects,
    storage::{self, ApiError},
    AppState,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use tokio::sync::watch;
use volley_media::montage::{OriginalClip, SelectedClip};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

const MAX_TRACK_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_ROW_BYTES: usize = 512 * 1024;

#[derive(Clone)]
pub(crate) struct ClipSource {
    file: PathBuf,
    sha256: String,
    frames: u64,
    replay_id: String,
    rally_run_id: String,
    video: String,
}

pub(crate) fn resolve_sources(
    state: &AppState,
    project_id: &str,
    selected: &[SelectedClip],
    clips: &[OriginalClip],
    materials: &[MaterialBinding],
) -> Result<Vec<ClipSource>, ApiError> {
    let mut sources = Vec::with_capacity(clips.len());
    for (index, clip) in clips.iter().enumerate() {
        let original = selected.get(index).ok_or_else(|| {
            storage::conflict("Монтаж не соответствует исходному набору фрагментов")
        })?;
        let mut matches = materials.iter().filter(|material| {
            material.video == original.video
                && material
                    .intervals
                    .iter()
                    .any(|(start, end)| clip.start_frame >= *start && clip.end_frame <= *end)
        });
        let material = matches.next().ok_or_else(|| {
            storage::bad_request(
                "Для экспорта распознавания требуется сохранённая CPU-версия каждого фрагмента",
            )
        })?;
        if matches.any(|other| other.rally_run_id != material.rally_run_id) {
            return Err(storage::conflict(
                "Для фрагмента обнаружено несколько несовместимых CPU-версий",
            ));
        }
        let run = inspect_project(state, project_id, &material.rally_run_id)?;
        let source = run
            .project_source
            .canonicalize()
            .map_err(|_| storage::conflict("Исходное видео CPU-анализа недоступно"))?;
        if run.video != original.video || source != clip.source || clip.end_frame >= run.frames {
            return Err(storage::conflict(
                "Выбранный фрагмент не соответствует проверенному CPU-анализу",
            ));
        }
        let dir = projects::project_root(state)
            .join(project_id)
            .join("native_cpu_replays")
            .join(&run.replay_id);
        let manifest = read_manifest(&dir, "tracking").map_err(storage::bad_request)?;
        if manifest["schema"] != "volley-native-cpu-tracking/1"
            || manifest["state"] != "provisional_tracklets_rebuilt"
            || manifest["file"] != "tracked_frames.jsonl"
            || manifest["project_id"] != project_id
            || manifest["video"] != run.video
            || manifest["replay_id"] != run.replay_id
            || manifest["frames"] != run.frames
            || manifest["fps"] != 30
            || manifest["sha256"] != run.replay_sha
            || manifest["original_source"] != material.original_source
        {
            return Err(storage::conflict(
                "Источник распознавания не соответствует сохранённому CPU-анализу",
            ));
        }
        let file = storage::safe_file(&dir, "tracked_frames.jsonl", MAX_TRACK_BYTES)
            .map_err(storage::bad_request)?;
        sources.push(ClipSource {
            file,
            sha256: run.replay_sha,
            frames: run.frames,
            replay_id: run.replay_id,
            rally_run_id: run.run_id,
            video: run.video,
        });
    }
    Ok(sources)
}

fn check_cancel(cancel: &watch::Receiver<bool>) -> Result<(), String> {
    if *cancel.borrow() {
        Err("cancelled".into())
    } else {
        Ok(())
    }
}

fn export_frames(
    clips: &[OriginalClip],
    sources: &[ClipSource],
    paths: &[PathBuf],
    cancel: &watch::Receiver<bool>,
) -> Result<(), String> {
    let mut writers = paths
        .iter()
        .map(|path| File::create(path).map(BufWriter::new))
        .collect::<io::Result<Vec<_>>>()
        .map_err(|error| format!("Create recognition evidence: {error}"))?;
    let mut seen = HashSet::new();
    let starts = clips
        .iter()
        .scan(0u64, |position, clip| {
            let start = *position;
            *position += clip.end_frame - clip.start_frame + 1;
            Some(start)
        })
        .collect::<Vec<_>>();
    for source in sources {
        if !seen.insert(source.file.clone()) {
            continue;
        }
        let indices = sources
            .iter()
            .enumerate()
            .filter_map(|(index, other)| (other.file == source.file).then_some(index))
            .collect::<Vec<_>>();
        if indices.iter().any(|&index| {
            sources[index].sha256 != source.sha256 || sources[index].frames != source.frames
        }) {
            return Err("Conflicting CPU evidence for montage clips".into());
        }
        let file = File::open(&source.file)
            .map_err(|error| format!("Open recognition evidence: {error}"))?;
        let mut input = BufReader::new(file);
        let mut digest = Sha256::new();
        let mut line = Vec::new();
        for frame in 0..source.frames {
            if frame.is_multiple_of(300) {
                check_cancel(cancel)?;
            }
            line.clear();
            input
                .read_until(b'\n', &mut line)
                .map_err(|error| format!("Read recognition evidence: {error}"))?;
            if line.is_empty() || line.len() > MAX_ROW_BYTES {
                return Err("CPU recognition evidence was truncated or oversized".into());
            }
            digest.update(&line);
            let relevant = indices
                .iter()
                .copied()
                .filter(|&index| {
                    clips[index].start_frame <= frame && frame <= clips[index].end_frame
                })
                .collect::<Vec<_>>();
            if relevant.is_empty() {
                continue;
            }
            let row: Value = serde_json::from_slice(&line)
                .map_err(|error| format!("Read selected CPU frame: {error}"))?;
            if row["frame_index"].as_u64() != Some(frame)
                || !row["tracked_players"].is_array()
                || !row["ball_candidates"].is_array()
                || row["track_ids_provisional"] != true
            {
                return Err("Selected recognition frame has invalid tracking data".into());
            }
            for index in relevant {
                let clip = &clips[index];
                let output_frame = starts[index] + frame - clip.start_frame;
                let record = json!({
                    "clip_index":index,
                    "output_frame_30fps":output_frame,
                    "output_time_sec":output_frame as f64 / 30.0,
                    "source_frame":frame,
                    "source_time_sec":clip.analysis_offset_sec + frame as f64 / 30.0,
                    "ball":row["selected_ball"],
                    "ball_candidates":row["ball_candidates"],
                    "players":row["tracked_players"],
                    "identities_provisional":true
                });
                serde_json::to_writer(&mut writers[index], &record)
                    .map_err(|error| format!("Write recognition frame: {error}"))?;
                writers[index]
                    .write_all(b"\n")
                    .map_err(|error| format!("Write recognition line: {error}"))?;
            }
        }
        if input
            .read_until(b'\n', &mut line)
            .map_err(|error| error.to_string())?
            != 0
            || format!("{:x}", digest.finalize()) != source.sha256
        {
            return Err("CPU recognition evidence checksum or frame count changed".into());
        }
    }
    for mut writer in writers {
        writer
            .flush()
            .map_err(|error| format!("Flush recognition evidence: {error}"))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("Sync recognition evidence: {error}"))?;
    }
    Ok(())
}

pub(crate) fn build_archive(
    root: &Path,
    clips: &[OriginalClip],
    sources: &[ClipSource],
    receipt: &Path,
    cancel: &watch::Receiver<bool>,
) -> Result<(), String> {
    if clips.is_empty() || sources.len() != clips.len() {
        return Err("Montage recognition source count differs from clips".into());
    }
    check_cancel(cancel)?;
    let temp = root.join("_render_temp").join("recognition");
    fs::create_dir_all(&temp).map_err(|error| format!("Create archive staging: {error}"))?;
    let paths = (0..clips.len())
        .map(|i| temp.join(format!("clip_{i:04}.jsonl")))
        .collect::<Vec<_>>();
    export_frames(clips, sources, &paths, cancel)?;
    check_cancel(cancel)?;
    let entries = clips
        .iter()
        .zip(sources)
        .enumerate()
        .scan(0u64, |offset, (index, (clip, source))| {
            let frames = clip.end_frame - clip.start_frame + 1;
            let row = json!({
                "clip_index":index,
                "video":source.video,
                "rally_run_id":source.rally_run_id,
                "replay_id":source.replay_id,
                "source_replay_sha256":source.sha256,
                "source_frame_start":clip.start_frame,
                "source_frame_end":clip.end_frame,
                "source_time_start_sec":clip.analysis_offset_sec + clip.start_frame as f64 / 30.0,
                "output_frame_start_30fps":*offset,
                "output_time_start_sec":*offset as f64 / 30.0,
                "frames":frames,
                "data_file":format!("recognition/clip_{index:04}.jsonl")
            });
            *offset += frames;
            Some(row)
        })
        .collect::<Vec<_>>();
    let manifest = json!({
        "schema":"volley-montage-recognition/1",
        "coordinate_system":"normalized_original_display_v1",
        "timeline_fps":30,
        "identity_policy":"provisional_track_ids_not_verified_athletes",
        "pose_policy":"players[].pose_contains_measured_or_interpolated_coco17_joints_when_available",
        "ball_policy":"ball_is_selected_vballnet_observation_null_when_absent",
        "video_file":"montage.mp4",
        "clips":entries
    });
    let partial = root.join("montage.zip.partial");
    let result = (|| -> Result<(), String> {
        let output =
            File::create(&partial).map_err(|error| format!("Create montage archive: {error}"))?;
        let mut zip = ZipWriter::new(BufWriter::new(output));
        let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        let compressed =
            SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        zip.start_file("montage.mp4", stored)
            .map_err(|error| error.to_string())?;
        io::copy(
            &mut File::open(root.join("montage.mp4")).map_err(|error| error.to_string())?,
            &mut zip,
        )
        .map_err(|error| format!("Pack montage MP4: {error}"))?;
        check_cancel(cancel)?;
        zip.start_file("selection.json", compressed)
            .map_err(|error| error.to_string())?;
        io::copy(
            &mut File::open(receipt).map_err(|error| error.to_string())?,
            &mut zip,
        )
        .map_err(|error| format!("Pack montage receipt: {error}"))?;
        zip.start_file("recognition/manifest.json", compressed)
            .map_err(|error| error.to_string())?;
        serde_json::to_writer_pretty(&mut zip, &manifest)
            .map_err(|error| format!("Pack recognition manifest: {error}"))?;
        for (index, path) in paths.iter().enumerate() {
            check_cancel(cancel)?;
            zip.start_file(format!("recognition/clip_{index:04}.jsonl"), compressed)
                .map_err(|error| error.to_string())?;
            io::copy(
                &mut File::open(path).map_err(|error| error.to_string())?,
                &mut zip,
            )
            .map_err(|error| format!("Pack recognition frames: {error}"))?;
        }
        let writer = zip
            .finish()
            .map_err(|error| format!("Finish archive: {error}"))?;
        writer
            .into_inner()
            .map_err(|error| format!("Flush montage ZIP: {error}"))?
            .sync_all()
            .map_err(|error| format!("Sync montage ZIP: {error}"))?;
        check_cancel(cancel)?;
        fs::rename(&partial, root.join("montage.zip"))
            .map_err(|error| format!("Publish montage ZIP: {error}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(partial);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn archive_keeps_ball_tracks_and_pose_joints_in_output_time() {
        let root =
            std::env::temp_dir().join(format!("montage-recognition-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("_render_temp")).unwrap();
        let original = root.join("original.mp4");
        fs::write(&original, b"original").unwrap();
        fs::write(root.join("montage.mp4"), b"rendered montage video").unwrap();
        fs::write(root.join("selection.json"), b"{}").unwrap();
        let source = root.join("tracked_frames.jsonl");
        let rows = (0..6).map(|frame| {
            let record = json!({
                "frame_index":frame,
                "selected_ball":{"source_model":"vballnet","detection":{"bbox_xyxy_norm":[0.1,0.2,0.3,0.4]}},
                "ball_candidates":[],
                "tracked_players":[{"track_id":7,"pose":[[0.2,0.5,0.99]],"measurement_source":"rtmpose_rgb"}],
                "track_ids_provisional":true
            });
            format!("{}\n",serde_json::to_string(&record).unwrap())
        }).collect::<String>();
        fs::write(&source, rows.as_bytes()).unwrap();
        let hash = format!("{:x}", Sha256::digest(rows.as_bytes()));
        let example = OriginalClip {
            source: original,
            start_frame: 2,
            end_frame: 3,
            analysis_offset_sec: 0.0,
            native_fps: 30.0,
            native_rate: "30/1".into(),
            native_width: 1920,
            native_height: 1080,
        };
        let clips = vec![
            example.clone(),
            OriginalClip {
                start_frame: 4,
                end_frame: 4,
                ..example
            },
        ];
        let sources = clips
            .iter()
            .map(|_| ClipSource {
                file: source.clone(),
                sha256: hash.clone(),
                frames: 6,
                replay_id: "a".repeat(32),
                rally_run_id: "b".repeat(32),
                video: "sample.mp4".into(),
            })
            .collect::<Vec<_>>();
        let (_tx, rx) = watch::channel(false);
        build_archive(&root, &clips, &sources, &root.join("selection.json"), &rx).unwrap();
        let zip = File::open(root.join("montage.zip")).unwrap();
        let mut zip = zip::ZipArchive::new(zip).unwrap();
        let mut manifest = String::new();
        zip.by_name("recognition/manifest.json")
            .unwrap()
            .read_to_string(&mut manifest)
            .unwrap();
        let manifest: Value = serde_json::from_str(&manifest).unwrap();
        assert_eq!(manifest["clips"][1]["output_frame_start_30fps"], 3 - 1);
        let mut frames = String::new();
        zip.by_name("recognition/clip_0001.jsonl")
            .unwrap()
            .read_to_string(&mut frames)
            .unwrap();
        let frame: Value = serde_json::from_str(frames.trim()).unwrap();
        assert_eq!(frame["source_frame"], 4);
        assert_eq!(frame["output_frame_30fps"], 2);
        assert_eq!(frame["ball"]["source_model"], "vballnet");
        assert_eq!(frame["players"][0]["pose"][0][2], 0.99);
        assert_eq!(frame["identities_provisional"], true);
        let mut video = String::new();
        zip.by_name("montage.mp4")
            .unwrap()
            .read_to_string(&mut video)
            .unwrap();
        assert_eq!(video, "rendered montage video");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn recognition_rejects_changed_cpu_snapshot() {
        let root = std::env::temp_dir().join(format!(
            "montage-recognition-invalid-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).unwrap();
        let file = root.join("tracked_frames.jsonl");
        fs::write(&file,b"{\"frame_index\":0,\"tracked_players\":[],\"ball_candidates\":[],\"track_ids_provisional\":true}\n").unwrap();
        let clip = OriginalClip {
            source: file.clone(),
            start_frame: 0,
            end_frame: 0,
            analysis_offset_sec: 0.0,
            native_fps: 30.0,
            native_rate: "30/1".into(),
            native_width: 100,
            native_height: 100,
        };
        let source = ClipSource {
            file,
            sha256: "0".repeat(64),
            frames: 1,
            replay_id: "a".into(),
            rally_run_id: "b".into(),
            video: "v".into(),
        };
        let (_tx, rx) = watch::channel(false);
        let err = export_frames(&[clip], &[source], &[root.join("out.jsonl")], &rx).unwrap_err();
        assert!(err.contains("checksum"));
        fs::remove_dir_all(root).unwrap();
    }
}
