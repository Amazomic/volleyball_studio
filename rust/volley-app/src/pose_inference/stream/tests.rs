#[test]
fn shared_pose_sampling_includes_every_emitted_sample_after_bounded_lookahead() {
    for frames in [
        8u64, 9, 10, 17, 18, 19, 27, 28, 36, 40, 48, 49, 50, 71, 72, 73,
    ] {
        for stride in [1usize, 2, 3, 5, 9, 10, 30] {
            let emitted = frames / 9 * 9;
            let expected: Vec<_> = (0..emitted)
                .filter(|frame| frame % stride as u64 == 0)
                .collect();
            let mut pending = VecDeque::new();
            let mut actual = Vec::new();
            for first in (0..emitted).step_by(9) {
                for idx in first..first + 9 {
                    if idx % stride as u64 == 0 {
                        pending.push_back(sample(idx));
                    }
                }
                actual.extend(ready(&mut pending, first + 8).into_iter().map(|v| v.index));
            }
            actual.extend(ready(&mut pending, u64::MAX).into_iter().map(|v| v.index));
            assert_eq!(
                actual, expected,
                "Pose sample loss at {frames} frames / stride {stride}"
            );
        }
    }
}

// RTMPose streaming timing, manifest and cancellation regression tests.
use super::*;

fn sample(index: u64) -> PendingFrame {
    PendingFrame {
        index,
        pixels: FramePixels::Host {
            rgb: Arc::new(Vec::new()),
            offset: 0,
        },
        width: 1920,
        height: 1080,
        boxes: Vec::new(),
    }
}

#[test]
fn pose_window_requires_thirty_future_frames_or_terminal_drain() {
    let mut queue = VecDeque::from([sample(0), sample(10), sample(20), sample(30), sample(40)]);
    assert!(ready(&mut queue, 29).is_empty());
    assert_eq!(
        ready(&mut queue, 30)
            .iter()
            .map(|p| p.index)
            .collect::<Vec<_>>(),
        vec![0]
    );
    assert!(ready(&mut queue, 39).is_empty());
    assert_eq!(
        ready(&mut queue, 50)
            .iter()
            .map(|p| p.index)
            .collect::<Vec<_>>(),
        vec![10, 20]
    );
    assert_eq!(
        ready(&mut queue, u64::MAX)
            .iter()
            .map(|p| p.index)
            .collect::<Vec<_>>(),
        vec![30, 40]
    );
    assert!(queue.is_empty());
}

fn fake_pending() -> (PendingPose, Value) {
    let id = Uuid::new_v4().simple().to_string();
    let folder = std::env::temp_dir().join(format!("volley-pose-publish-{id}"));
    fs::create_dir(&folder).unwrap();
    let record = b"{\"frame\":0,\"players\":[]}\n";
    fs::write(folder.join(POSE_FILE), record).unwrap();
    let sha = format!("{:x}", Sha256::digest(record));
    let original = json!({"source_sha256":"abc"});
    let source = json!({"bytes":123,"mtime_ns":900});
    let pending = PendingPose {
        artifact: PoseArtifact {
            folder,
            committed: false,
        },
        run_id: id,
        checksum: sha,
        model_sha: MODEL_SHA256.to_string(),
        sample_stride: 10,
        emitted_frames: 18,
        decoded_frames: 22,
        native_width: 1920,
        native_height: 1080,
        sampled: 1,
        people: 0,
        observed_crops: 0,
        interpolated_crops: 0,
        cuda_pixels: false,
        original_source: original.clone(),
        source_identity: source.clone(),
        project_id: "1234abcdef01".into(),
        video: "fixture.mp4".into(),
    };
    let fusion = json!({
        "schema":"volley-native-gpu-fusion/1",
        "state":"raw_candidates_fused",
        "fusion_id":Uuid::new_v4().simple().to_string(),
        "project_id":"1234abcdef01",
        "video":"fixture.mp4",
        "source_identity":source,
        "original_source":original,
        "emitted_frames":18,
        "decoded_frames":22,
        "player_sampling_period_frames":10,
        "sha256":"a".repeat(64),
    });
    (pending, fusion)
}

#[test]
fn streamed_pose_manifest_is_published_only_after_verified_fusion() {
    let (pending, fusion) = fake_pending();
    let folder = pending.artifact.folder.clone();
    let cancelled = AtomicBool::new(false);
    let manifest = pending.publish(&fusion, &cancelled).unwrap();
    assert_eq!(manifest["schema"], "volley-native-gpu-pose/1");
    assert_eq!(manifest["source_fusion_sha256"], fusion["sha256"]);
    assert_eq!(manifest["decoded_frames"], 22);
    assert_eq!(manifest["sampled_frames"], 1);
    assert_eq!(manifest["observed_pose_crops"], 0);
    assert_eq!(manifest["interpolated_pose_crops"], 0);
    assert!(manifest.get("interpolate_pose").is_none());
    assert_eq!(
        manifest["sha256"],
        format!(
            "{:x}",
            Sha256::digest(fs::read(folder.join(POSE_FILE)).unwrap())
        )
    );
    assert_eq!(
        crate::storage::read_json(&folder.join("manifest.json"))
            .unwrap()
            .unwrap(),
        manifest
    );
    fs::remove_dir_all(folder).unwrap();
}

#[test]
fn mismatched_fusion_and_cancelled_pose_are_not_published() {
    let (pending, mut fusion) = fake_pending();
    let folder = pending.artifact.folder.clone();
    fusion["source_identity"]["mtime_ns"] = json!(901);
    let cancelled = AtomicBool::new(false);
    assert!(pending
        .publish(&fusion, &cancelled)
        .unwrap_err()
        .contains("fusion identity"));
    assert!(!folder.exists());
    let (pending, fusion) = fake_pending();
    let folder = pending.artifact.folder.clone();
    cancelled.store(true, Ordering::Release);
    assert_eq!(
        pending.publish(&fusion, &cancelled).unwrap_err(),
        "cancelled"
    );
    assert!(!folder.exists());
}

#[test]
fn incomplete_pose_folder_never_survives_cancellation_or_failed_fusion() {
    let root = std::env::temp_dir().join(format!("volley-pose-clean-{}", Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    let artifact = PoseArtifact {
        folder: root.clone(),
        committed: false,
    };
    drop(artifact);
    assert!(!root.exists());
}
