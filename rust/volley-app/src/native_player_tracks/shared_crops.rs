use super::*;
use volley_inference::reid_openvino::{bgr24_to_nchw, HEIGHT, WIDTH};

const CROP_BYTES: usize = WIDTH * HEIGHT * 3;
pub(crate) const DEFAULT_REID_BUFFER_MIB: u32 = 512;
pub(crate) const MAX_REID_BUFFER_MIB: u32 = 4096;
type CropKey = (u64, [u32; 4]);
type CropPlan = BTreeMap<u64, Vec<[f64; 4]>>;

#[derive(Clone)]
pub(crate) struct SharedReidCrops(Arc<Mutex<Buffer>>);

#[derive(Default)]
struct Buffer {
    limit_mib: u32,
    capacity: usize,
    source: Option<(PathBuf, (u64, u128), u64)>,
    crops: BTreeMap<CropKey, Vec<u8>>,
    hits: usize,
    misses: usize,
}

impl SharedReidCrops {
    pub(crate) fn new(limit_mib: u32) -> Result<Self, String> {
        if limit_mib > MAX_REID_BUFFER_MIB {
            return Err("ReID crop buffer limit exceeds 4096 MiB".into());
        }
        let capacity = (u64::from(limit_mib) * 1024 * 1024 / CROP_BYTES as u64) as usize;
        Ok(Self(Arc::new(Mutex::new(Buffer {
            limit_mib,
            capacity,
            ..Buffer::default()
        }))))
    }

    pub(crate) fn bind(&self, source: &Path, offset: f64) -> Result<(), String> {
        let stamp = inference_stream::source_stamp(source)?;
        let mut buffer = self
            .0
            .lock()
            .map_err(|_| "ReID crop buffer lock poisoned")?;
        if buffer.source.is_some() {
            return Err("ReID crop buffer already bound".into());
        }
        buffer.source = Some((source.to_path_buf(), stamp, offset.to_bits()));
        Ok(())
    }

    pub(crate) fn stats(&self) -> Value {
        self.0.lock().map_or(Value::Null, |buffer| {
            json!({"bound":buffer.source.is_some(),"captured":buffer.crops.len(),"hits":buffer.hits,"misses":buffer.misses,
                "bytes":buffer.crops.len()*CROP_BYTES,"capacity":buffer.capacity,"limit_mib":buffer.limit_mib})
        })
    }

    pub(crate) fn capture(
        &self,
        frame: u64,
        rgb: &[u8],
        width: usize,
        height: usize,
        boxes: &[[f64; 4]],
    ) -> Result<(), String> {
        let mut buffer = self
            .0
            .lock()
            .map_err(|_| "ReID crop buffer lock poisoned")?;
        for &bbox in boxes {
            if buffer.crops.len() >= buffer.capacity {
                break;
            }
            let Ok(crop) = crop_box(bbox, width as u32, height as u32) else {
                continue;
            };
            if let std::collections::btree_map::Entry::Vacant(entry) =
                buffer.crops.entry((frame, crop))
            {
                entry.insert(rgb_crop_bgr(rgb, width, height, crop)?);
            }
        }
        Ok(())
    }

    pub(super) fn get(
        &self,
        source: &Path,
        stamp: (u64, u128),
        offset: f64,
        frame: u64,
        crop: [u32; 4],
    ) -> Option<Vec<f32>> {
        let mut buffer = self.0.lock().ok()?;
        let valid = buffer.source.as_ref().is_some_and(|(path, saved, start)| {
            path == source && *saved == stamp && *start == offset.to_bits()
        });
        let result = valid
            .then(|| buffer.crops.get(&(frame, crop)))
            .flatten()
            .and_then(|bytes| bgr24_to_nchw(bytes).ok());
        if result.is_some() {
            buffer.hits += 1;
        } else {
            buffer.misses += 1;
        }
        result
    }
}

pub(crate) fn plan_shared_reid_crops(
    tracked: &Path,
    sha: &str,
    frames: u64,
    rally: (u64, u64),
    offset: u64,
) -> Result<CropPlan, String> {
    let windows = crate::native_rally_evidence::RallyWindows {
        intervals: vec![rally],
        contexts: BTreeMap::new(),
    };
    let (scoped, full) = source_index::scan_with_scope(tracked, sha, frames, Some(&windows))?;
    let mut result = CropPlan::new();
    for track in scoped
        .values()
        .chain(full.iter().flat_map(|tracks| tracks.values()))
    {
        for (frame, bbox) in std::iter::once((track.best_frame, track.best_box)).chain(
            track
                .photo_samples
                .iter()
                .map(|sample| (sample.frame, sample.bbox)),
        ) {
            let boxes = result.entry(offset + frame).or_default();
            if !boxes.contains(&bbox) {
                boxes.push(bbox);
            }
        }
    }
    Ok(result)
}

fn weights(input: usize, output: usize) -> Vec<Vec<(usize, f32)>> {
    let ratio = input as f64 / output as f64;
    let radius = ratio.max(1.0);
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * ratio - 0.5;
            let left = ((center - radius).floor() as isize).max(0) as usize;
            let right = ((center + radius).ceil() as usize).min(input - 1);
            let mut taps = (left..=right)
                .filter_map(|index| {
                    let weight = (1.0 - (index as f64 - center).abs() / radius).max(0.0) as f32;
                    (weight > 0.0).then_some((index, weight))
                })
                .collect::<Vec<_>>();
            let sum = taps.iter().map(|(_, weight)| weight).sum::<f32>();
            for (_, weight) in &mut taps {
                *weight /= sum;
            }
            taps
        })
        .collect()
}

fn rgb_crop_bgr(
    rgb: &[u8],
    width: usize,
    height: usize,
    crop: [u32; 4],
) -> Result<Vec<u8>, String> {
    let [x, y, w, h] = crop.map(|n| n as usize);
    if width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(rgb.len())
        || w == 0
        || h == 0
        || x > width
        || y > height
        || w > width - x
        || h > height - y
    {
        return Err("Invalid original RGB ReID crop".into());
    }
    let horizontal = weights(w, WIDTH);
    let vertical = weights(h, HEIGHT);
    let mut intermediate = vec![0.0_f32; h * WIDTH * 3];
    for row in 0..h {
        for (column, taps) in horizontal.iter().enumerate() {
            let target = (row * WIDTH + column) * 3;
            for &(sx, weight) in taps {
                let source = ((y + row) * width + x + sx) * 3;
                for channel in 0..3 {
                    intermediate[target + channel] += rgb[source + 2 - channel] as f32 * weight;
                }
            }
        }
    }
    let mut bgr = vec![0_u8; CROP_BYTES];
    for (row, taps) in vertical.iter().enumerate() {
        for column in 0..WIDTH * 3 {
            let value = taps
                .iter()
                .map(|&(sy, weight)| intermediate[sy * WIDTH * 3 + column] * weight)
                .sum::<f32>();
            bgr[row * WIDTH * 3 + column] = value.round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(bgr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_crops_cover_verified_samples_and_exclude_interpolations() {
        let root = std::env::temp_dir().join(format!("volley-reid-plan-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let path = root.join("tracked.jsonl");
        let mut raw = Vec::new();
        for frame in 0..18_u64 {
            let bbox = [0.15 + frame as f64 / 1000.0, 0.2, 0.45, 0.75];
            let player = json!({"track_id":7,"is_verified_athlete":false,
                "detection":{"frame_index":frame,"class_name":"player","score":if frame == 3 {0.99} else {0.8},
                "bbox_xyxy_norm":bbox,"query_index":0}});
            let row = json!({"frame_index":frame,"selected_ball":null,"tracked_players":[player],
                "interpolated_track_ids":if frame == 3 {vec![7]} else {vec![]},
                "track_ids_provisional":true,"confirmed_athlete_identities":false,"touches_confirmed":false});
            serde_json::to_writer(&mut raw, &row).unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let plan = plan_shared_reid_crops(&path, &sha, 18, (4, 15), 100).unwrap();
        let windows = crate::native_rally_evidence::RallyWindows {
            intervals: vec![(4, 15)],
            contexts: BTreeMap::new(),
        };
        let (tracks, _) = source_index::scan_with_scope(&path, &sha, 18, Some(&windows)).unwrap();
        let track = &tracks[&7];
        assert!(plan[&(100 + track.best_frame)].contains(&track.best_box));
        for photo in photo_comparison::independent_photo_samples(track) {
            assert!(plan[&(100 + photo.frame)].contains(&photo.bbox));
        }
        assert!(!plan.contains_key(&103));
        assert!(plan_shared_reid_crops(&path, &"0".repeat(64), 18, (4, 15), 100).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn original_stream_crop_matches_legacy_model_input_on_synthetic_video() {
        use std::{process::Command, sync::atomic::AtomicBool};
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let root =
            std::env::temp_dir().join(format!("volley-reid-stream-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let source = root.join("source.mp4");
        let created = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x256:rate=30",
                "-frames:v",
                "12",
                "-c:v",
                "libx264",
                "-crf",
                "12",
                "-pix_fmt",
                "yuv420p",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            created.status.success(),
            "{}",
            String::from_utf8_lossy(&created.stderr)
        );
        let stamp = inference_stream::source_stamp(&source).unwrap();
        let shared = SharedReidCrops::new(DEFAULT_REID_BUFFER_MIB).unwrap();
        shared.bind(&source, 0.0).unwrap();
        let bbox = [0.2, 0.1, 0.8, 0.9];
        let crop = crop_box(bbox, 320, 256).unwrap();
        let mut seen = 0;
        volley_media::raw_decoder::decode_original_frames(
            &source,
            12,
            &AtomicBool::new(false),
            |index, rgb, w, h| {
                assert_eq!((w, h), (320, 256));
                if index == 3 || index == 6 {
                    shared.capture(index, rgb, w, h, &[bbox])?;
                    seen += 1;
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(seen, 2);
        let input = shared.get(&source, stamp, 0.0, 3, crop).unwrap();
        let another = shared.get(&source, stamp, 0.0, 6, crop).unwrap();
        assert_ne!(input, another);
        let [x, y, w, h] = crop;
        let legacy = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "0.100000",
                "-i",
            ])
            .arg(&source)
            .args([
                "-vf",
                &format!("crop={w}:{h}:{x}:{y},scale={WIDTH}:{HEIGHT}:flags=bilinear,format=bgr24"),
                "-frames:v",
                "1",
                "-an",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(
            legacy.status.success(),
            "{}",
            String::from_utf8_lossy(&legacy.stderr)
        );
        let old = bgr24_to_nchw(&legacy.stdout).unwrap();
        let mae = input
            .iter()
            .zip(&old)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / input.len() as f64;
        println!(
            "Synthetic original RGB crop vs FFmpeg legacy mean absolute pixel delta: {mae:.3}"
        );
        assert!(mae < 12.0, "Unexpected RGB crop drift {mae}");
        if let Ok(library) = std::env::var("VOLLEY_TEST_OPENVINO_C_LIB") {
            let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let mut encoder =
                volley_inference::reid_openvino::Encoder::load(&repo, Path::new(&library)).unwrap();
            let a = encoder.infer(&input).unwrap();
            let b = encoder.infer(&old).unwrap();
            let cosine = a
                .iter()
                .zip(&b)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>();
            println!("Synthetic legacy/shared ReID cosine: {cosine:.6}");
            assert!(cosine > 0.98, "Shared crop ReID descriptor drift {cosine}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shared_crop_preserves_original_geometry_and_bgr_channels() {
        let mut rgb = vec![0_u8; 192 * 288 * 3];
        for y in 16..272 {
            for x in 32..160 {
                let i = (y * 192 + x) * 3;
                rgb[i..i + 3].copy_from_slice(&[x as u8, y as u8, 197]);
            }
        }
        let crop = rgb_crop_bgr(&rgb, 192, 288, [32, 16, 128, 256]).unwrap();
        assert_eq!(&crop[..3], &[197, 16, 32]);
        assert_eq!(&crop[crop.len() - 3..], &[197, 15, 159]);
        assert!(rgb_crop_bgr(&rgb, 192, 288, [160, 16, 128, 256]).is_err());
    }

    #[test]
    fn shared_crop_downsamples_without_aliasing_or_channel_swap() {
        let rgb: Vec<u8> = (0..512 * 512)
            .flat_map(|i| {
                let v = if (i / 512 + i % 512) % 2 == 0 { 0 } else { 255 };
                [v, 52, 179]
            })
            .collect();
        let crop = rgb_crop_bgr(&rgb, 512, 512, [0, 0, 512, 512]).unwrap();
        for px in crop.as_chunks::<3>().0 {
            assert_eq!(px[0], 179);
            assert_eq!(px[1], 52);
            assert!((120..=135).contains(&px[2]));
        }
    }

    #[test]
    fn shared_crop_buffer_capacity_is_configurable() {
        for (mib, expected) in [
            (0, 0),
            (1, 10),
            (DEFAULT_REID_BUFFER_MIB, 5461),
            (528, 5632),
            (4096, 43690),
        ] {
            let stats = SharedReidCrops::new(mib).unwrap().stats();
            let capacity = stats["capacity"].as_u64().unwrap() as usize;
            assert_eq!(capacity, expected);
            assert_eq!(stats["limit_mib"], mib);
            assert_eq!(stats["bytes"], 0);
            assert!(capacity as u64 * CROP_BYTES as u64 <= u64::from(mib) * 1024 * 1024);
            assert!((capacity as u64 + 1) * CROP_BYTES as u64 > u64::from(mib) * 1024 * 1024);
        }
        assert!(SharedReidCrops::new(4097).is_err());
    }

    #[test]
    fn shared_crop_buffer_stops_at_configured_capacity() {
        let rgb = [25, 100, 200].repeat(64 * 64);
        let bbox = [0.25, 0.25, 0.75, 0.75];
        for (mib, expected) in [(0, 0), (1, 10)] {
            let shared = SharedReidCrops::new(mib).unwrap();
            for frame in 0..12 {
                shared.capture(frame, &rgb, 64, 64, &[bbox]).unwrap();
            }
            assert_eq!(shared.stats()["captured"], expected);
        }
    }

    #[test]
    fn shared_crop_lookup_rejects_other_source_time_and_box() {
        let shared = SharedReidCrops::new(DEFAULT_REID_BUFFER_MIB).unwrap();
        let source = PathBuf::from("source.mp4");
        let stamp = (100, 123);
        shared.0.lock().unwrap().source = Some((source.clone(), stamp, 2.0_f64.to_bits()));
        let rgb = [25, 100, 200].repeat(64 * 64);
        let bbox = [0.25, 0.25, 0.75, 0.75];
        let crop = crop_box(bbox, 64, 64).unwrap();
        shared.capture(9, &rgb, 64, 64, &[bbox, bbox]).unwrap();
        let input = shared.get(&source, stamp, 2.0, 9, crop).unwrap();
        assert_eq!(input[0], 200.0);
        assert_eq!(input[WIDTH * HEIGHT], 100.0);
        assert!(shared.get(&source, stamp, 2.0, 10, crop).is_none());
        assert!(shared.get(&source, stamp, 0.0, 9, crop).is_none());
        assert!(shared.get(&source, (100, 124), 2.0, 9, crop).is_none());
        assert!(shared
            .get(Path::new("other.mp4"), stamp, 2.0, 9, crop)
            .is_none());
        assert!(shared.get(&source, stamp, 2.0, 9, [1, 1, 32, 32]).is_none());
        assert_eq!(shared.stats()["captured"], 1);
        assert_eq!(shared.stats()["hits"], 1);
        assert_eq!(shared.stats()["misses"], 5);
    }
}
