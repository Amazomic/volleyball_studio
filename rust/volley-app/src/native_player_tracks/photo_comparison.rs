use super::*;
use std::sync::{Mutex, OnceLock};
use tokio::io::AsyncWriteExt;

// Allow full multi-angle albums from several players without rejecting the
// comparison. The 64 MiB source-data bound still applies.
const MAX_COMPARISON_REFERENCE_PHOTOS: usize = 1024;
const MAX_CACHED_REFERENCE_FEATURES: usize = 2048;
type ReferenceFeatureCache = BTreeMap<(String, String), Vec<f32>>;
static REFERENCE_FEATURES: OnceLock<Mutex<ReferenceFeatureCache>> = OnceLock::new();

fn reference_features() -> &'static Mutex<ReferenceFeatureCache> {
    REFERENCE_FEATURES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

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
            if photos.len() >= MAX_COMPARISON_REFERENCE_PHOTOS {
                return Err(storage::bad_request(
                    "Too many reference photos for one comparison (maximum 1024)",
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
fn independent_photo_samples(track: &Aggregate) -> Vec<PhotoSample> {
    let mut candidates = track.photo_samples.clone();
    candidates.sort_by_key(|sample| sample.frame);
    candidates.dedup_by_key(|sample| sample.frame);
    let duration = track
        .last
        .saturating_sub(track.first)
        .saturating_add(1)
        .max(1);
    let best_part = track
        .best_frame
        .saturating_sub(track.first)
        .saturating_mul(3)
        / duration;
    let mut used = BTreeSet::from([track.best_frame]);
    let mut selected = Vec::with_capacity(2);
    for part in 0..3_u64 {
        if part == best_part {
            continue;
        }
        let target = track.first + duration.saturating_mul(part * 2 + 1) / 6;
        if let Some(sample) = candidates
            .iter()
            .filter(|sample| {
                sample.frame >= track.first
                    && sample.frame <= track.last
                    && !used.contains(&sample.frame)
                    && sample.frame.saturating_sub(track.first).saturating_mul(3) / duration == part
            })
            .min_by_key(|sample| sample.frame.abs_diff(target))
        {
            selected.push(sample.clone());
            used.insert(sample.frame);
        }
    }
    while selected.len() < 2 {
        let Some(sample) = candidates
            .iter()
            .filter(|sample| {
                sample.frame >= track.first
                    && sample.frame <= track.last
                    && !used.contains(&sample.frame)
            })
            .max_by_key(|sample| {
                used.iter()
                    .map(|frame| sample.frame.abs_diff(*frame))
                    .min()
                    .unwrap_or(0)
            })
        else {
            break;
        };
        selected.push(sample.clone());
        used.insert(sample.frame);
    }
    selected
}

type IndependentCropCache = BTreeMap<String, Vec<f32>>;

fn independent_features() -> &'static Mutex<IndependentCropCache> {
    static CACHE: OnceLock<Mutex<IndependentCropCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
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
    // Reuse features for identical saved images and model fingerprints across
    // explicit refresh and auto-link; neither user data nor cache schema changes.
    let mut photo_vectors = vec![None; snap.photos.len()];
    {
        let cache = reference_features()
            .lock()
            .map_err(|_| storage::internal("Reference ReID cache lock poisoned"))?;
        for (index, photo) in snap.photos.iter().enumerate() {
            photo_vectors[index] = cache
                .get(&(snap.model.clone(), photo.image_sha256.clone()))
                .cloned();
        }
    }
    let mut missing_indices = Vec::new();
    let mut inputs = Vec::new();
    for (index, photo) in snap.photos.iter().enumerate() {
        if photo_vectors[index].is_none() {
            missing_indices.push(index);
            inputs.push(decode(&photo.bytes).await?);
        }
    }
    if !inputs.is_empty() {
        let library = std::env::var("VOLLEY_OPENVINO_C_LIB")
            .map_err(|_| storage::bad_request("Native OpenVINO C runtime is not configured"))?;
        let repo = state.repository_root.clone();
        let inferred = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, ApiError> {
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
        let mut cache = reference_features()
            .lock()
            .map_err(|_| storage::internal("Reference ReID cache lock poisoned"))?;
        if cache.len() + inferred.len() > MAX_CACHED_REFERENCE_FEATURES {
            cache.clear();
        }
        for (index, vector) in missing_indices.into_iter().zip(inferred) {
            cache.insert(
                (snap.model.clone(), snap.photos[index].image_sha256.clone()),
                vector.clone(),
            );
            photo_vectors[index] = Some(vector);
        }
    }
    let vectors = photo_vectors
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| storage::internal("Reference ReID vectors incomplete"))?;
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
    let mut independently_verified = BTreeMap::<u64, BTreeMap<usize, Vec<f32>>>::new();
    let mut cached_inputs = Vec::<(u64, usize, String, u64, [u32; 4])>::new();
    let mut source_for_sampling = None;
    if first_pass.as_object().is_some_and(|rows| !rows.is_empty()) {
        let plan = preparation::plan_for(&state, &req.project_id, &req.video)?;
        let source = plan
            .ready_path()
            .ok_or_else(|| storage::bad_request("Prepared Track ID video missing"))?;
        let source_stamp = inference_stream::source_stamp(source).map_err(storage::bad_request)?;
        let video = volley_media::probe(source).map_err(storage::bad_request)?;
        let cache_prefix = format!(
            "{}:{:?}:{}:{}",
            snap.model,
            source_stamp,
            snap.ctx.source_sha,
            source.display()
        );
        let cache = independent_features()
            .lock()
            .map_err(|_| storage::internal("Independent ReID cache lock poisoned"))?;
        for (&id, track) in &snap.ctx.tracks {
            let high_enough = first_pass[id.to_string()].as_array().is_some_and(|rows| {
                rows.iter().any(|row| {
                    row["cosine_similarity"]
                        .as_f64()
                        .is_some_and(|score| score >= 0.40)
                })
            });
            if !high_enough {
                continue;
            }
            let samples = independent_photo_samples(track);
            if samples.len() != 2 {
                continue;
            }
            for (index, sample) in samples.into_iter().enumerate() {
                let Ok(crop) = crop_box(sample.bbox, video.width, video.height) else {
                    continue;
                };
                let key = format!("{cache_prefix}:{id}:{}:{crop:?}", sample.frame);
                if let Some(vector) = cache.get(&key) {
                    independently_verified
                        .entry(id)
                        .or_default()
                        .insert(index, vector.clone());
                } else {
                    cached_inputs.push((id, index, key, sample.frame, crop));
                }
            }
        }
        drop(cache);
        source_for_sampling = Some(source.to_path_buf());
    }
    if let Some(source) = source_for_sampling {
        let mut set = tokio::task::JoinSet::new();
        let mut decoded = Vec::new();
        for (id, index, key, frame, crop) in cached_inputs {
            let path = source.clone();
            set.spawn(async move {
                (
                    id,
                    index,
                    key,
                    decode_track_crop(&path, frame, crop).await.ok(),
                )
            });
            if set.len() >= 4 {
                if let Some(Ok((id, index, key, Some(input)))) = set.join_next().await {
                    decoded.push((id, index, key, input));
                }
            }
        }
        while let Some(result) = set.join_next().await {
            if let Ok((id, index, key, Some(input))) = result {
                decoded.push((id, index, key, input));
            }
        }
        if !decoded.is_empty() {
            let repo = state.repository_root.clone();
            let library = std::env::var("VOLLEY_OPENVINO_C_LIB")
                .map_err(|_| storage::bad_request("Native OpenVINO C runtime is not configured"))?;
            let inferred = tokio::task::spawn_blocking(
                move || -> Result<Vec<(u64, usize, String, Vec<f32>)>, ApiError> {
                    let mut encoder =
                        volley_inference::reid_openvino::Encoder::load(&repo, Path::new(&library))
                            .map_err(storage::bad_request)?;
                    decoded
                        .into_iter()
                        .map(|(id, index, key, input)| {
                            Ok((
                                id,
                                index,
                                key,
                                encoder.infer(&input).map_err(storage::bad_request)?,
                            ))
                        })
                        .collect()
                },
            )
            .await
            .map_err(|e| storage::internal(format!("ReID independent crop worker: {e}")))??;
            let mut cache = independent_features()
                .lock()
                .map_err(|_| storage::internal("Independent ReID cache lock poisoned"))?;
            if cache.len().saturating_add(inferred.len()) > 8192 {
                cache.clear();
            }
            for (id, index, key, vector) in inferred {
                cache.insert(key, vector.clone());
                independently_verified
                    .entry(id)
                    .or_default()
                    .insert(index, vector);
            }
        }
    }
    let multi_samples = independently_verified
        .into_iter()
        .filter_map(|(id, samples)| {
            if samples.len() == 2 {
                Some((id, samples.into_values().collect::<Vec<_>>()))
            } else {
                None
            }
        })
        .collect::<BTreeMap<_, _>>();
    let matches = photo_match_candidates_with_samples(
        &snap.ctx,
        &snap.identity,
        &snap.vectors,
        &multi_samples,
        &refs,
    )
    .map_err(storage::bad_request)?;
    Ok(json!({
        "schema":"volley-native-player-photo-comparison/1",
        "project_id":snap.ctx.project_id,
        "replay_id":snap.ctx.replay_id,
        "source_replay_sha256":snap.ctx.source_sha,
        "player_matches":matches,
        "photo_count":refs.len(),
        "independent_photo_checks":multi_samples.len() * 2,
        "multiframe_track_count":multi_samples.len(),
        "photo_sample_policy":"three_observations_v1",
        "identity_revision":snap.identity_revision,
        "album_revisions":snap.albums,
        "model_fingerprint":snap.model,
        "auto_assign_enabled":false,
        "confirmed":false
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_real_observations_are_distributed_over_track_duration() {
        let track = Aggregate {
            first: 0,
            last: 95,
            best_frame: 49,
            best_box: [0.1, 0.2, 0.6, 0.8],
            photo_samples: [0, 16, 33, 49, 71, 86, 95]
                .into_iter()
                .map(|frame| PhotoSample {
                    frame,
                    bbox: [0.1, 0.2, 0.6, 0.8],
                })
                .collect(),
            ..Aggregate::default()
        };
        let samples = independent_photo_samples(&track);
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.frame)
                .collect::<Vec<_>>(),
            vec![16, 86]
        );
        let short = Aggregate {
            photo_samples: vec![PhotoSample {
                frame: 49,
                bbox: [0.1, 0.2, 0.6, 0.8],
            }],
            ..track
        };
        assert!(independent_photo_samples(&short).is_empty());
    }

    #[test]
    fn full_multi_athlete_photo_albums_fit_in_one_comparison() {
        // Six existing athlete albums with 46+64+48+32+55+64 photos
        // previously failed at the old hard cap of 256.
        assert!(MAX_COMPARISON_REFERENCE_PHOTOS >= 46 + 64 + 48 + 32 + 55 + 64);
        assert!(MAX_COMPARISON_REFERENCE_PHOTOS >= 8 * MAX_PLAYER_PHOTOS);
        assert!(MAX_CACHED_REFERENCE_FEATURES >= MAX_COMPARISON_REFERENCE_PHOTOS);
    }

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
