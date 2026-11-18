//! On-demand browser-safe fragmented MP4 playback. Never modifies source evidence.
use crate::{
    playback, projects,
    storage::{self, ApiError},
    AppState,
};
use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, Response},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::{Duration, UNIX_EPOCH},
};
use tokio::{fs, process::Command, sync::Semaphore};
use uuid::Uuid;

pub const FRAGMENT_SECONDS: u64 = 18;
static TRANSCODES: OnceLock<Semaphore> = OnceLock::new();

fn cached_file(root: &Path, source: &Path, index: u64) -> Result<PathBuf, ApiError> {
    let metadata = source
        .metadata()
        .map_err(|error| storage::internal(format!("Video metadata failed: {error}")))?;
    let modified = metadata
        .modified()
        .map_err(|error| storage::internal(format!("Video modified time failed: {error}")))?
        .duration_since(UNIX_EPOCH)
        .map_err(|error| storage::internal(format!("Video modified time invalid: {error}")))?;
    let identity = format!(
        "{}:{}:{}:{}",
        source.display(),
        metadata.len(),
        modified.as_nanos(),
        FRAGMENT_SECONDS
    );
    let key = format!("{:x}", Sha256::digest(identity.as_bytes()));
    Ok(root
        .join("browser-fragments-v1")
        .join(key)
        .join(format!("{index:07}.mp4")))
}

async fn cache_ready(path: &Path) -> bool {
    fs::metadata(path)
        .await
        .is_ok_and(|metadata| metadata.len() > 1024)
}

async fn transcode(source: &Path, output: &Path, index: u64) -> Result<(), ApiError> {
    let part = output.with_extension(format!("{}.partial.mp4", Uuid::new_v4()));
    let start = (index * FRAGMENT_SECONDS).to_string();
    let duration = FRAGMENT_SECONDS.to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            &start,
            "-i",
        ])
        .arg(source)
        .args([
            "-t",
            &duration,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "scale=1280:1280:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "26",
            "-profile:v",
            "baseline",
            "-level:v",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "60",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "-y",
        ])
        .arg(&part)
        .kill_on_drop(true);
    let result = tokio::time::timeout(Duration::from_secs(120), command.output()).await;
    let success = match result {
        Ok(Ok(output)) if output.status.success() => true,
        Ok(Ok(output)) => {
            let message = String::from_utf8_lossy(&output.stderr);
            let _ = fs::remove_file(&part).await;
            return Err(storage::internal(format!(
                "Fragment encoding failed: {}",
                message.chars().take(500).collect::<String>()
            )));
        }
        Ok(Err(error)) => {
            let _ = fs::remove_file(&part).await;
            return Err(storage::internal(format!("Cannot launch FFmpeg: {error}")));
        }
        Err(_) => false,
    };
    if !success {
        let _ = fs::remove_file(&part).await;
        return Err(storage::internal("Fragment encoding timed out"));
    }
    if !cache_ready(&part).await {
        let _ = fs::remove_file(&part).await;
        return Err(storage::internal(
            "Fragment encoding produced no playable MP4",
        ));
    }
    if let Err(error) = fs::rename(&part, output).await {
        let _ = fs::remove_file(&part).await;
        return Err(storage::internal(format!(
            "Fragment cache publish failed: {error}"
        )));
    }
    Ok(())
}

/// A single timeline-aligned fMP4 fragment, loaded only if the browser needs it.
pub async fn video_fragment(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
    headers: HeaderMap,
) -> Result<Response<Body>, ApiError> {
    let project_id = query
        .get("project_id")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| storage::bad_request("Missing project_id"))?;
    let video = query
        .get("video")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| storage::bad_request("Missing video"))?;
    let index = query
        .get("index")
        .ok_or_else(|| storage::bad_request("Missing fragment index"))?
        .parse::<u64>()
        .map_err(|_| storage::bad_request("Invalid fragment index"))?;
    let project: Value = projects::existing_project(&state, project_id)?;
    if !project["videos"]
        .as_array()
        .is_some_and(|rows| rows.iter().any(|value| value.as_str() == Some(video)))
    {
        return Err(storage::bad_request("Video does not belong to project"));
    }
    let source = projects::canonical_source(&state, video)?;
    let info = tokio::task::spawn_blocking({
        let source = source.clone();
        move || volley_media::probe(&source).map_err(storage::bad_request)
    })
    .await
    .map_err(|error| storage::internal(format!("Video probe task failed: {error}")))??;
    if index > u64::MAX / FRAGMENT_SECONDS || (index * FRAGMENT_SECONDS) as f64 >= info.duration_sec
    {
        return Err(storage::bad_request("Fragment is outside video duration"));
    }
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Browser fragment cache requires writable storage",
        ));
    }
    let output = cached_file(&state.data_root, &source, index)?;
    if !cache_ready(&output).await {
        let _permit = TRANSCODES
            .get_or_init(|| Semaphore::new(1))
            .acquire()
            .await
            .map_err(|error| storage::internal(format!("Fragment queue closed: {error}")))?;
        // One shared encoder slot avoids duplicate fragments and GPU job contention.
        if !cache_ready(&output).await {
            let folder = output
                .parent()
                .ok_or_else(|| storage::internal("Invalid fragment cache directory"))?;
            fs::create_dir_all(folder).await.map_err(|error| {
                storage::internal(format!("Fragment cache creation failed: {error}"))
            })?;
            transcode(&source, &output, index).await?;
        }
    }
    playback::serve_mp4(output, &headers).await
}

#[cfg(test)]
mod tests {
    use super::{cached_file, FRAGMENT_SECONDS};
    use std::{fs, time::SystemTime};

    #[test]
    fn fragments_are_partitioned_by_source_and_index() {
        let root =
            std::env::temp_dir().join(format!("volley-fragment-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let first = root.join("a.mp4");
        let second = root.join("b.mp4");
        fs::write(&first, b"file A").unwrap();
        fs::write(&second, b"file B").unwrap();
        let a0 = cached_file(&root, &first, 0).unwrap();
        let a1 = cached_file(&root, &first, 1).unwrap();
        let b0 = cached_file(&root, &second, 0).unwrap();
        assert_ne!(a0, a1);
        assert_ne!(a0, b0);
        assert_eq!(FRAGMENT_SECONDS, 18);
        assert!(SystemTime::now()
            .duration_since(fs::metadata(&first).unwrap().modified().unwrap())
            .is_ok());
        let _ = fs::remove_dir_all(root);
    }
}
