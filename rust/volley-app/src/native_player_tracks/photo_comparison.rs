use super::*;
use tokio::io::AsyncWriteExt;

struct PhotoInput {
    player_id: String,
    project_id: String,
    video: String,
    replay_id: String,
    track_id: u64,
    image_sha256: String,
    bytes: Vec<u8>,
}

fn exclude_cross_athlete_reference_collisions(photos: &mut Vec<PhotoInput>) {
    let mut owners = BTreeMap::<String, Option<String>>::new();
    for photo in photos.iter() {
        owners
            .entry(photo.image_sha256.clone())
            .and_modify(|owner| {
                if owner.as_deref() != Some(photo.player_id.as_str()) {
                    *owner = None;
                }
            })
            .or_insert_with(|| Some(photo.player_id.clone()));
    }
    photos.retain(|photo| {
        owners
            .get(&photo.image_sha256)
            .is_some_and(|owner| owner.as_deref() == Some(photo.player_id.as_str()))
    });
}

pub(super) struct Snapshot {
    ctx: Inputs,
    identity: Value,
    pub(super) identity_revision: String,
    pub(super) albums: Vec<(String, String)>,
    photos: Vec<PhotoInput>,
    vectors: BTreeMap<u64, Vec<f32>>,
    pub(super) model: String,
}

pub(super) fn snapshot(state: &AppState, req: &ReidReviewRequest) -> Result<Snapshot, ApiError> {
    let ctx = source_for_rallies(
        state,
        &req.project_id,
        &req.video,
        &req.replay_id,
        req.rally_run_id.as_deref(),
    )?;
    if ctx.source_sha != req.expected_source_sha256 {
        return Err(storage::conflict(
            "CPU replay changed; refresh player gallery",
        ));
    }
    let raw = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    let identity = validated(raw.as_deref(), &ctx).map_err(storage::bad_request)?;
    let model = volley_inference::reid_openvino::fingerprint(&state.repository_root)
        .map_err(storage::bad_request)?;
    let assigned = identity["assignments"]
        .as_object()
        .ok_or_else(|| storage::bad_request("Invalid assignments"))?;
    let excluded = identity["excluded_ids"]
        .as_array()
        .ok_or_else(|| storage::bad_request("Invalid excluded IDs"))?
        .iter()
        .filter_map(Value::as_u64)
        .collect::<BTreeSet<_>>();
    let eligible = ctx
        .tracks
        .keys()
        .filter(|id| !assigned.contains_key(&id.to_string()) && !excluded.contains(id))
        .collect::<Vec<_>>();
    if eligible.len() > 1024 {
        return Err(storage::bad_request(
            "Comparison is limited to 1024 Track IDs",
        ));
    }
    let mut albums = Vec::new();
    let mut photos = Vec::new();
    let mut total_bytes = 0usize;
    for player_id in ctx.roster.keys() {
        let folder = checked_reference_folder(state, player_id, false)?;
        let raw =
            versioned_store::read_optional_bounded(&folder.join("index.json"), MAX_REFERENCE_INDEX)
                .map_err(storage::bad_request)?;
        let doc = references_doc(raw.as_deref(), player_id)?;
        albums.push((player_id.clone(), revision(raw.as_deref())));
        let mut seen = BTreeSet::new();
        for row in doc["photos"].as_array().into_iter().flatten() {
            let sha = row["image_sha256"]
                .as_str()
                .ok_or_else(|| storage::bad_request("Invalid photo checksum"))?;
            if !seen.insert(sha.to_owned()) {
                continue;
            }
            if photos.len() >= 256 {
                return Err(storage::bad_request(
                    "Too many reference photos for one comparison (maximum 256)",
                ));
            }
            let id = row["id"]
                .as_str()
                .ok_or_else(|| storage::bad_request("Invalid reference ID"))?;
            let bytes = versioned_store::read_optional_bounded(
                &folder.join(format!("ref-{id}.jpg")),
                MAX_PHOTO_BYTES as u64,
            )
            .map_err(storage::bad_request)?
            .ok_or_else(|| storage::conflict("Reference photo missing"))?;
            total_bytes = total_bytes.saturating_add(bytes.len());
            if total_bytes > 64 * 1024 * 1024 {
                return Err(storage::bad_request(
                    "Reference photos exceed 64 MiB for one comparison",
                ));
            }
            if bytes.len() < 100
                || !bytes.starts_with(&[0xff, 0xd8])
                || !bytes.ends_with(&[0xff, 0xd9])
                || format!("{:x}", Sha256::digest(&bytes)) != sha
            {
                return Err(storage::conflict("Reference photo checksum mismatch"));
            }
            let field = |name: &str| {
                row[name]
                    .as_str()
                    .filter(|v| !v.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| storage::bad_request("Reference photo provenance missing"))
            };
            photos.push(PhotoInput {
                player_id: player_id.clone(),
                project_id: field("project_id")?,
                video: field("video")?,
                replay_id: field("replay_id")?,
                track_id: row["track_id"]
                    .as_u64()
                    .ok_or_else(|| storage::bad_request("Invalid reference track"))?,
                image_sha256: sha.to_owned(),
                bytes,
            });
        }
    }
    exclude_cross_athlete_reference_collisions(&mut photos);
    Ok(Snapshot {
        ctx,
        identity,
        identity_revision: revision(raw.as_deref()),
        albums,
        photos,
        vectors: BTreeMap::new(),
        model,
    })
}

async fn decode(bytes: &[u8]) -> Result<Vec<f32>, ApiError> {
    let mut command = tokio::process::Command::new("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vf",
            "scale=128:256:flags=bilinear,format=bgr24",
            "-frames:v",
            "1",
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|e| storage::bad_request(format!("FFmpeg photo decoder: {e}")))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| storage::internal("Photo decoder input unavailable"))?;
    let data = bytes.to_vec();
    let writer = tokio::spawn(async move { stdin.write_all(&data).await });
    let result = tokio::time::timeout(std::time::Duration::from_secs(35), child.wait_with_output())
        .await
        .map_err(|_| storage::bad_request("Photo decoding timed out"))?
        .map_err(|e| storage::bad_request(format!("Photo decoder failed: {e}")))?;
    writer
        .await
        .map_err(|e| storage::internal(format!("Photo decoder input: {e}")))?
        .map_err(|e| storage::bad_request(format!("Photo decoder input: {e}")))?;
    if !result.status.success()
        || result.stdout.len() != volley_inference::reid_openvino::PIXELS * 3
    {
        return Err(storage::bad_request("Cannot decode saved reference photo"));
    }
    volley_inference::reid_openvino::bgr24_to_nchw(&result.stdout).map_err(storage::bad_request)
}

/// Only a competition for the same athlete needs the extra video crops:
/// independent tracklets in different rallies already have one clear owner.
pub(super) fn contested_photo_tracks(ctx: &Inputs, matches: &Value) -> BTreeSet<u64> {
    let mut owner_tracks = BTreeMap::<&str, Vec<u64>>::new();
    for (id, proposals) in matches.as_object().into_iter().flat_map(|m| m.iter()) {
        let (Ok(track), Some(owner)) = (
            id.parse::<u64>(),
            proposals
                .as_array()
                .and_then(|rows| rows.first())
                .and_then(|row| row["player_id"].as_str()),
        ) else {
            continue;
        };
        if ctx.tracks.contains_key(&track) {
            owner_tracks.entry(owner).or_default().push(track);
        }
    }
    let mut contested = BTreeSet::new();
    for group in owner_tracks.values() {
        for (index, id) in group.iter().enumerate() {
            for other in group.iter().skip(index + 1) {
                if tracks_co_visible(&ctx.tracks[id], &ctx.tracks[other]) {
                    contested.insert(*id);
                    contested.insert(*other);
                }
            }
        }
    }
    contested
}

/// Choose an independently measured crop well away from the best photo.
/// The index only stores measured (never interpolated) samples inside rallies.
fn independent_photo_sample(track: &Aggregate) -> Option<&PhotoSample> {
    track
        .photo_samples
        .iter()
        .filter(|sample| sample.frame.abs_diff(track.best_frame) >= 60)
        .max_by_key(|sample| sample.frame.abs_diff(track.best_frame))
}

async fn decode_track_crop(
    video: &Path,
    frame: u64,
    [x, y, w, h]: [u32; 4],
) -> Result<Vec<f32>, ApiError> {
    let sec = format!("{:.6}", frame as f64 / 30.0);
    let filter = format!(
        "crop={w}:{h}:{x}:{y},scale={}:{}:flags=bilinear,format=bgr24",
        volley_inference::reid_openvino::WIDTH,
        volley_inference::reid_openvino::HEIGHT
    );
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(35),
        tokio::process::Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                &sec,
                "-i",
            ])
            .arg(video)
            .args([
                "-vf",
                &filter,
                "-frames:v",
                "1",
                "-an",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| storage::bad_request("Independent Track ID photo timed out"))?
    .map_err(|e| storage::bad_request(format!("FFmpeg independent Track ID photo: {e}")))?;
    if !output.status.success()
        || output.stdout.len() != volley_inference::reid_openvino::PIXELS * 3
    {
        return Err(storage::bad_request(
            "Independent Track ID photo could not be decoded",
        ));
    }
    volley_inference::reid_openvino::bgr24_to_nchw(&output.stdout).map_err(storage::bad_request)
}

pub(super) async fn compare(
    state: Arc<AppState>,
    headers: HeaderMap,
    req: ReidReviewRequest,
) -> Result<Value, ApiError> {
    if req.rally_run_id.as_deref().is_some_and(|id| !opaque_id(id)) {
        return Err(storage::bad_request(
            "Invalid rally run for photo comparison",
        ));
    }
    let cloned = state.clone();
    let query = req.clone();
    let mut snap = tokio::task::spawn_blocking(move || snapshot(&cloned, &query))
        .await
        .map_err(|e| storage::internal(format!("Photo comparison source: {e}")))??;
    if snap.photos.is_empty() {
        return Ok(json!({
            "schema":"volley-native-player-photo-comparison/1",
            "project_id":snap.ctx.project_id,"replay_id":snap.ctx.replay_id,
            "source_replay_sha256":snap.ctx.source_sha,
            "player_matches":{},"photo_count":0,
            "identity_revision":snap.identity_revision,
            "album_revisions":snap.albums,
            "model_fingerprint":snap.model,
            "auto_assign_enabled":false,"confirmed":false
        }));
    }
    let assigned = snap.identity["assignments"]
        .as_object()
        .ok_or_else(|| storage::bad_request("Invalid assignments"))?;
    let excluded = snap.identity["excluded_ids"]
        .as_array()
        .ok_or_else(|| storage::bad_request("Invalid excluded IDs"))?
        .iter()
        .filter_map(Value::as_u64)
        .collect::<BTreeSet<_>>();
    let ids = snap
        .ctx
        .tracks
        .keys()
        .filter(|id| !assigned.contains_key(&id.to_string()) && !excluded.contains(id))
        .copied()
        .collect::<Vec<_>>();
    if ids.is_empty() {
        return Ok(
            json!({"schema":"volley-native-player-photo-comparison/1","project_id":snap.ctx.project_id,"replay_id":snap.ctx.replay_id,"source_replay_sha256":snap.ctx.source_sha,"player_matches":{},"photo_count":0,"identity_revision":snap.identity_revision,"album_revisions":snap.albums,"model_fingerprint":snap.model,"auto_assign_enabled":false,"confirmed":false}),
        );
    }
    for chunk in ids.chunks(8) {
        let Json(result) = embeddings(
            State(state.clone()),
            headers.clone(),
            Json(EmbeddingRequest {
                project_id: req.project_id.clone(),
                video: req.video.clone(),
                replay_id: req.replay_id.clone(),
                rally_run_id: req.rally_run_id.clone(),
                expected_source_sha256: req.expected_source_sha256.clone(),
                track_ids: chunk.to_vec(),
                // Source crop provenance is now part of the SHA/model-bound
                // cache. Subsequent comparisons reuse exactly the same crop.
                transient: false,
            }),
        )
        .await?;
        if result["model_fingerprint"] != snap.model {
            return Err(storage::conflict("ReID model changed during comparison"));
        }
        for row in result["features"]
            .as_array()
            .ok_or_else(|| storage::internal("Track features missing"))?
        {
            let id = row["track_id"]
                .as_u64()
                .ok_or_else(|| storage::internal("Track feature ID missing"))?;
            let vector: Vec<f32> = serde_json::from_value(row["vector"].clone())
                .map_err(|_| storage::internal("Invalid track feature"))?;
            if !chunk.contains(&id) || snap.vectors.insert(id, vector).is_some() {
                return Err(storage::internal("Unexpected track feature"));
            }
        }
        if chunk.iter().any(|id| !snap.vectors.contains_key(id)) {
            return Err(storage::internal("Incomplete track features"));
        }
    }
    let _permit = state
        .reid_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("ReID gate: {e}")))?;
    let _preview = state
        .preview_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("Photo decoder gate: {e}")))?;
    let mut inputs = Vec::new();
    for photo in &snap.photos {
        inputs.push(decode(&photo.bytes).await?);
    }
    let library = std::env::var("VOLLEY_OPENVINO_C_LIB")
        .map_err(|_| storage::bad_request("Native OpenVINO C runtime is not configured"))?;
    let repo = state.repository_root.clone();
    let vectors = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, ApiError> {
        let mut encoder =
            volley_inference::reid_openvino::Encoder::load(&repo, Path::new(&library))
                .map_err(storage::bad_request)?;
        inputs
            .iter()
            .map(|input| encoder.infer(input).map_err(storage::bad_request))
            .collect()
    })
    .await
    .map_err(|e| storage::internal(format!("Photo ReID worker: {e}")))??;
    let check_state = state.clone();
    let check_req = req.clone();
    let current = tokio::task::spawn_blocking(move || snapshot(&check_state, &check_req))
        .await
        .map_err(|e| storage::internal(format!("Photo comparison recheck: {e}")))??;
    if current.identity_revision != snap.identity_revision
        || current.albums != snap.albums
        || current.model != snap.model
        || current.ctx.roster != snap.ctx.roster
        || current.ctx.tracks != snap.ctx.tracks
    {
        return Err(storage::conflict(
            "Player assignments, photo album or ReID model changed during comparison",
        ));
    }
    let refs = snap
        .photos
        .drain(..)
        .zip(vectors)
        .map(|(p, vector)| PhotoReference {
            player_id: p.player_id,
            project_id: p.project_id,
            video: p.video,
            replay_id: p.replay_id,
            track_id: p.track_id,
            vector,
        })
        .collect::<Vec<_>>();
    let first_pass = photo_match_candidates(&snap.ctx, &snap.identity, &snap.vectors, &refs)
        .map_err(storage::bad_request)?;
    let contested = contested_photo_tracks(&snap.ctx, &first_pass);
    let mut independently_verified = BTreeMap::<u64, Vec<f32>>::new();
    if !contested.is_empty() {
        let plan = preparation::plan_for(&state, &req.project_id, &req.video)?;
        let source = plan
            .ready_path()
            .ok_or_else(|| storage::bad_request("Prepared Track ID video missing"))?;
        let video = volley_media::probe(source).map_err(storage::bad_request)?;
        let mut crops = Vec::<(u64, Vec<f32>)>::new();
        for id in contested.iter().take(128) {
            let Some(track) = snap.ctx.tracks.get(id) else {
                continue;
            };
            let Some(sample) = independent_photo_sample(track) else {
                continue;
            };
            let Ok(crop) = crop_box(sample.bbox, video.width, video.height) else {
                continue;
            };
            // Supplemental evidence is best effort: an inaccessible or too
            // blurry late crop should not abort the entire CPU postprocessing.
            if let Ok(input) = decode_track_crop(source, sample.frame, crop).await {
                crops.push((*id, input));
            }
        }
        if !crops.is_empty() {
            let repo = state.repository_root.clone();
            let library = std::env::var("VOLLEY_OPENVINO_C_LIB")
                .map_err(|_| storage::bad_request("Native OpenVINO C runtime is not configured"))?;
            independently_verified = tokio::task::spawn_blocking(
                move || -> Result<BTreeMap<u64, Vec<f32>>, ApiError> {
                    let mut encoder =
                        volley_inference::reid_openvino::Encoder::load(&repo, Path::new(&library))
                            .map_err(storage::bad_request)?;
                    let mut compared = BTreeMap::new();
                    for (id, input) in crops {
                        compared.insert(id, encoder.infer(&input).map_err(storage::bad_request)?);
                    }
                    Ok(compared)
                },
            )
            .await
            .map_err(|e| storage::internal(format!("Second ReID crop worker: {e}")))??;
        }
    }
    let matches = if independently_verified.is_empty() {
        first_pass
    } else {
        photo_match_candidates_with_samples(
            &snap.ctx,
            &snap.identity,
            &snap.vectors,
            &independently_verified,
            &refs,
        )
        .map_err(storage::bad_request)?
    };
    Ok(
        json!({"schema":"volley-native-player-photo-comparison/1", "project_id":snap.ctx.project_id,
        "replay_id":snap.ctx.replay_id,"source_replay_sha256":snap.ctx.source_sha,
        "player_matches":matches,"photo_count":refs.len(),
        "independent_photo_checks":independently_verified.len(),
        "identity_revision":snap.identity_revision,"album_revisions":snap.albums,"model_fingerprint":snap.model,"auto_assign_enabled":false,"confirmed":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_image_shared_by_two_athletes_cannot_identify_either() {
        let photo = |player_id: &str, sha: &str| PhotoInput {
            player_id: player_id.to_owned(),
            project_id: "project".into(),
            video: "match.mp4".into(),
            replay_id: "replay".into(),
            track_id: 1,
            image_sha256: sha.to_owned(),
            bytes: vec![],
        };
        let mut refs = vec![
            photo("first", "same"),
            photo("second", "same"),
            photo("first", "good-first"),
            photo("second", "good-second"),
        ];
        exclude_cross_athlete_reference_collisions(&mut refs);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].image_sha256, "good-first");
        assert_eq!(refs[1].image_sha256, "good-second");
    }

    #[tokio::test]
    async fn saved_jpeg_decodes_to_openvino_input() {
        let image = tokio::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=red:s=16x32",
                "-frames:v",
                "1",
                "-c:v",
                "mjpeg",
                "-f",
                "image2pipe",
                "pipe:1",
            ])
            .output()
            .await
            .unwrap();
        assert!(image.status.success());
        let input = decode(&image.stdout).await.unwrap();
        assert_eq!(input.len(), volley_inference::reid_openvino::PIXELS * 3);
        assert!(input[2 * volley_inference::reid_openvino::PIXELS] > input[0]);
    }
}
