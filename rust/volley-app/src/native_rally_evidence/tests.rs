use super::*;

#[test]
fn unlinked_ball_trajectory_touch_remains_a_valid_unknown_candidate() {
    let event: NativeMotion = serde_json::from_value(json!({
        "frame_index":651,
        "provisional_track_id":null,
        "probability":1.0,
        "ball_x_px":955.0,
        "ball_y_px":485.0,
        "motion_frame_index":651,
        "direction_change_deg":94.0,
        "speed_before_px_frame":20.0,
        "speed_after_px_frame":31.0,
        "velocity_residual_px_frame":31.0,
        "technique":"unknown_touch",
        "physical_motion_supported":true,
        "athlete_identity_verified":false,
        "ball_occluded_at_contact":true,
        "inferred_body_reach_supported":false
    }))
    .unwrap();
    assert!(event.valid(700, 1920, 1080));
    let result = finalize_touch(&event, true, None, &mut BTreeMap::new()).unwrap();
    assert!(!result.attributed);
    assert!(result.kind.is_none());
}

#[cfg(test)]
#[test]
fn second_strike_by_same_player_is_not_a_duplicate_when_time_separated() {
    assert!(adjacent_same_track(Some((3, 100)), 3, 104));
    assert!(!adjacent_same_track(Some((3, 100)), 3, 130));
    assert!(!adjacent_same_track(Some((3, 100)), 4, 103));
}
#[cfg(test)]
#[test]
fn physical_surface_events_are_never_validated_as_athlete_touches() {
    let floor: NativeSurfaceEvent = serde_json::from_value(json!({
        "frame_index":20,"surface":"floor","outcome":"out",
        "evidence":"nine_observed_ball_frames_vertical_reversal","provisional":true
    }))
    .unwrap();
    assert!(floor.valid(10, 30));
    assert!(!floor.valid(21, 30));
    let mut not_provisional = floor.clone();
    not_provisional.provisional = false;
    assert!(!not_provisional.valid(10, 30));
    let mut fake = floor;
    fake.surface = "player".into();
    assert!(!fake.valid(10, 30));
}
#[test]
fn rally_windows_include_only_detected_play_not_inter_rally_frames() {
    let scope = RallyWindows {
        intervals: vec![(20, 45), (110, 185), (290, 310)],
        contexts: BTreeMap::new(),
    };
    for frame in [20, 40, 45, 110, 185, 290, 310] {
        assert!(scope.contains(frame));
    }
    for frame in [0, 19, 46, 109, 186, 289, 311, 400] {
        assert!(!scope.contains(frame));
    }
    assert!(!RallyWindows {
        intervals: vec![],
        contexts: BTreeMap::new()
    }
    .contains(100));
}

#[test]
fn motion_court_context_roundtrip_is_source_bounded() {
    let mut row = json!({
        "frame_index":115, "provisional_track_id":3,
        "probability":0.93, "ball_x_px":150.0, "ball_y_px":100.0,
        "motion_frame_index":115, "direction_change_deg":100.0,
        "speed_before_px_frame":4.0, "speed_after_px_frame":11.0,
        "velocity_residual_px_frame":8.0, "technique":"unknown_touch",
        "action_prediction":{
            "kind":"block","method":"court_geometry_v1",
            "evidence_score":0.86,"depth_ratio":-0.12,"posture_ratio":0.4
        },
        "court":{
            "x_m":-0.45,"y_m":1.0,"side":"near",
            "near_net":true,"in_serve_zone":false
        },
        "physical_motion_supported":true, "athlete_identity_verified":false
    });
    let event: NativeMotion = serde_json::from_value(row.clone()).unwrap();
    assert!(event.valid(400, 640, 360));
    assert_eq!(
        event.court.unwrap().side,
        volley_core::court_geometry::CourtSide::Near
    );
    row["court"]["x_m"] = json!(-20.0);
    let malformed: NativeMotion = serde_json::from_value(row).unwrap();
    assert!(!malformed.valid(400, 640, 360));
}

#[test]
fn pose_and_motion_are_required_to_name_a_touch() {
    let row = json!({
        "frame_index":40,"provisional_track_id":3,"probability":0.93,
        "ball_x_px":150.0,"ball_y_px":100.0,"motion_frame_index":40,
        "direction_change_deg":90.0,"speed_before_px_frame":4.0,
        "speed_after_px_frame":12.0,"velocity_residual_px_frame":8.0,
        "technique":"unknown_touch","physical_motion_supported":true,
        "athlete_identity_verified":false,"body_contact_supported":true,
        "contact_landmark":{"joint_indices":[9,9],"point_px":[150.0,100.0],
            "distance_px":0.0,"normalized_distance":0.0},
        "action_prediction":{"kind":"attack","evidence_score":0.85,
            "method":"contact_geometry_v1","depth_ratio":-0.1,"posture_ratio":0.4}
    });
    let mut event: NativeMotion = serde_json::from_value(row.clone()).unwrap();
    let mut last = BTreeMap::new();
    let result = finalize_touch(&event, true, None, &mut last).unwrap();
    assert!(result.attributed && result.pose_supported);
    assert!(result.kind.is_none());
    event.action_prediction.as_mut().unwrap().method =
        volley_core::action_prediction::PredictionMethod::PoseKeypointsMotionV1;
    assert_eq!(
        finalize_touch(&event, true, None, &mut BTreeMap::new())
            .unwrap()
            .kind,
        Some(ActionKind::Attack)
    );
    event.action_prediction.as_mut().unwrap().method =
        volley_core::action_prediction::PredictionMethod::CourtGeometryV1;
    assert!(finalize_touch(&event, true, None, &mut BTreeMap::new())
        .unwrap()
        .kind
        .is_none());
    event.action_prediction.as_mut().unwrap().method =
        volley_core::action_prediction::PredictionMethod::PoseKeypointsV1;
    assert!(finalize_touch(&event, true, None, &mut BTreeMap::new())
        .unwrap()
        .kind
        .is_none());
    event.action_prediction.as_mut().unwrap().kind = ActionKind::LowTouch;
    assert!(finalize_touch(&event, true, None, &mut BTreeMap::new())
        .unwrap()
        .kind
        .is_none());
}

fn fixture() -> (ValidatedRun, PathBuf) {
    let root =
        std::env::temp_dir().join(format!("rally-evidence-preview-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let rally_file = root.join("rallies.jsonl");
    let touch_file = root.join("touches.jsonl");
    let replay_id = "0123456789abcdef0123456789abcdef";
    let touch_id = "abcdef0123456789abcdef0123456789";
    let bounds = [(30, 80), (95, 155), (310, 380)];
    let mut rows = Vec::new();
    for (start, end) in bounds {
        let row = json!({
            "candidate_id":expected_id(replay_id,touch_id,start,end),
            "detected_start_frame":start,"detected_end_frame":end,
            "ball_observations":end-start+1,"ball_coverage":1.0,
            "motion_supported_touch_proposals":2,
            "first_touch_frame":start+10,"last_touch_frame":end-10,
            "max_touch_probability":0.9,"classification":"provisional_rally",
            "authoritative_rally":false,"athlete_identities_verified":false,
            "serve_verified":false,"reviewed":false
        });
        rows.extend_from_slice(serde_json::to_string(&row).unwrap().as_bytes());
        rows.push(b'\n');
    }
    let mut events = Vec::new();
    for (index, frame) in [40, 70, 115, 145, 330, 370].into_iter().enumerate() {
        let row = json!({
            "frame_index":frame,"probability":0.93,
            "provisional_track_id": if index % 2 == 0 { 3 } else { 4 },
            "ball_x_px":150.0,"ball_y_px":100.0,
            "motion_frame_index":frame,
            "direction_change_deg":100.0,
            "speed_before_px_frame":4.0,"speed_after_px_frame":11.0,
            "velocity_residual_px_frame":8.0,
            "technique":"unknown_touch",
            "action_prediction":(frame==115).then(|| json!({
                "kind":"dig","evidence_score":0.86,"method":"pose_keypoints_motion_v1",
                "depth_ratio":0.35,"posture_ratio":0.4
            })),
            "physical_motion_supported":true,"athlete_identity_verified":false,
            "body_contact_supported":true,
            "contact_landmark":{"joint_indices":[9,9],"point_px":[150.0,100.0],
                "distance_px":0.0,"normalized_distance":0.0},
        });
        events.extend_from_slice(serde_json::to_string(&row).unwrap().as_bytes());
        events.push(b'\n');
    }
    fs::write(&rally_file, &rows).unwrap();
    fs::write(&touch_file, &events).unwrap();
    let run = ValidatedRun {
        project_id: "1234abcdef01".into(),
        video: "match.mp4".into(),
        run_id: "0123456789abcdef0123456789abcdef".into(),
        project_source: root.join("source.mp4"),
        rally_file,
        touch_file,
        replay_id: replay_id.into(),
        replay_sha: "a".repeat(64),
        touch_id: touch_id.into(),
        rally_sha: format!("{:x}", Sha256::digest(rows)),
        touch_sha: format!("{:x}", Sha256::digest(events)),
        frames: 480,
        source_duration_sec: 16.0,
        analysis_range: None,
        width: 640,
        height: 360,
        context: 45,
        expected_rallies: 3,
        expected_touches: 6,
        min_ball_score: 0.35,
    };
    (run, root)
}
#[test]
fn primary_gpu_rallies_are_visible_in_preview_even_without_cpu_touch_events() {
    let (mut run, root) = fixture();
    let mut rows = fs::read_to_string(&run.rally_file)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    for row in &mut rows {
        row["method"] = json!("primary_gpu_rally_mask_v1");
        row["motion_supported_touch_proposals"] = json!(0);
        row["ball_observations"] = json!(0);
        row["ball_coverage"] = json!(0.0);
        row["ball_coverage_not_recomputed"] = json!(true);
        row["max_touch_probability"] = json!(0.0);
        row["first_touch_frame"] = row["detected_start_frame"].clone();
        row["last_touch_frame"] = row["detected_end_frame"].clone();
    }
    let bytes = rows
        .iter()
        .map(|row| format!("{}\n", serde_json::to_string(row).unwrap()))
        .collect::<String>();
    fs::write(&run.rally_file, &bytes).unwrap();
    fs::write(&run.touch_file, b"").unwrap();
    run.rally_sha = format!("{:x}", Sha256::digest(bytes.as_bytes()));
    run.touch_sha = format!("{:x}", Sha256::digest(b""));
    run.expected_touches = 0;
    let timeline = generate(&run, Choice::Timeline).unwrap().value;
    let visible = timeline["rallies"].as_array().unwrap();
    assert_eq!(visible.len(), 3);
    assert_eq!(visible[0]["start_frame"], 30);
    assert_eq!(visible[0]["end_frame"], 80);
    assert_eq!(visible[1]["start_frame"], 95);
    assert_eq!(visible[1]["end_frame"], 155);
    assert!(timeline["surface_contact_candidates"]
        .as_array()
        .unwrap()
        .is_empty());
    rows[1]["surface_event_candidates"] = json!([
        {"frame_index": 140, "surface": "floor", "outcome": "line_uncertain", "position": [0.3, 0.6], "evidence": "trajectory_ground_contact_v3", "provisional": true},
        {"frame_index": 150, "surface": "floor", "outcome": "out", "position": [0.2, 0.7], "evidence": "trajectory_ground_contact_v3", "provisional": true}
    ]);
    let bytes = rows
        .iter()
        .map(|row| format!("{}\n", serde_json::to_string(row).unwrap()))
        .collect::<String>();
    fs::write(&run.rally_file, &bytes).unwrap();
    run.rally_sha = format!("{:x}", Sha256::digest(bytes.as_bytes()));
    let timeline = generate(&run, Choice::Timeline).unwrap().value;
    let floor = timeline["surface_contact_candidates"].as_array().unwrap();
    assert_eq!(floor.len(), 2);
    assert_eq!(floor[0]["outcome"], "line_uncertain");
    assert_eq!(floor[1]["outcome"], "out");
    assert_eq!(floor[0]["position"], json!([0.3, 0.6]));
    assert!(timeline["touch_annotations"].as_array().unwrap().is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn verified_timeline_accepts_only_versioned_ball_rallies_without_touches() {
    let (mut run, root) = fixture();
    let saved = fs::read_to_string(&run.rally_file).unwrap();
    let mut rows = saved
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    rows[0]["motion_supported_touch_proposals"] = json!(0);
    rows[0]["max_touch_probability"] = json!(0.0);
    rows[0]["first_touch_frame"] = json!(30);
    rows[0]["last_touch_frame"] = json!(30);
    let original_touches = fs::read_to_string(&run.touch_file).unwrap();
    let remaining_touches = original_touches
        .lines()
        .filter(|line| {
            let frame: Value = serde_json::from_str(line).unwrap();
            frame["frame_index"].as_u64().unwrap() >= 95
        })
        .map(|line| format!("{line}\n"))
        .collect::<String>();
    fs::write(&run.touch_file, &remaining_touches).unwrap();
    run.touch_sha = format!("{:x}", Sha256::digest(remaining_touches.as_bytes()));
    run.expected_touches = 4;
    let store = |rows: &[Value], run: &mut ValidatedRun| {
        let bytes = rows
            .iter()
            .map(|row| format!("{}\n", serde_json::to_string(row).unwrap()))
            .collect::<String>();
        fs::write(&run.rally_file, &bytes).unwrap();
        run.rally_sha = format!("{:x}", Sha256::digest(bytes.as_bytes()));
    };
    store(&rows, &mut run);
    assert!(
        generate(&run, Choice::Timeline).is_err(),
        "Legacy rally proposals still require at least one motion-supported touch"
    );
    rows[0]["method"] = json!("coherent_vballnet_motion_v2");
    store(&rows, &mut run);
    let result = generate(&run, Choice::Timeline).unwrap();
    let verified = result.value["rallies"].as_array().unwrap();
    assert_eq!(verified.len(), 3);
    assert_eq!(verified[0]["start_frame"], 30);
    assert_eq!(verified[0]["end_frame"], 80);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn predicted_action_does_not_bypass_source_track_or_score_validation() {
    let (run, root) = fixture();
    let saved = fs::read_to_string(&run.touch_file).expect("touch fixture");
    let raw = saved.lines().nth(2).expect("predicted row");
    let mut event: NativeMotion = serde_json::from_str(raw).expect("typed prediction");
    assert!(event.valid(run.frames, run.width, run.height));
    event.provisional_track_id = None;
    assert!(!event.valid(run.frames, run.width, run.height));
    event.provisional_track_id = Some(3);
    event
        .action_prediction
        .as_mut()
        .expect("prediction")
        .evidence_score = 1.5;
    assert!(!event.valid(run.frames, run.width, run.height));
    fs::remove_dir_all(root).expect("remove test fixture");
}
#[test]
fn timeline_exposes_initial_serves_without_adding_physical_contacts() {
    let (run, root) = fixture();
    let proposals = json!([
        {"rally_start_frame":30,"trajectory_start_frame":30,
         "provisional_track_id":42,"kind":"serve",
         "identity_verified":false,"source":"rally_trajectory_start_v1"},
        {"rally_start_frame":95,"trajectory_start_frame":96,
         "provisional_track_id":null,"kind":"serve",
         "identity_verified":false,"source":"rally_trajectory_start_v1"}
    ]);
    let preliminary = serde_json::to_vec(&json!({
        "initial_serve_proposals":proposals,
    }))
    .unwrap();
    fs::write(root.join("preliminary_rallies.json"), &preliminary).unwrap();
    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec(&json!({
            "initial_serve_proposal_count":2,
            "preliminary_rally_sha256":format!("{:x}",Sha256::digest(&preliminary))
        }))
        .unwrap(),
    )
    .unwrap();
    let timeline = generate(&run, Choice::Timeline).unwrap();
    assert_eq!(timeline.value["serve_proposals"], proposals);
    assert_eq!(timeline.value["raw_touch_candidates"], 6);
    assert_eq!(
        timeline.value["touch_annotations"]
            .as_array()
            .unwrap()
            .len(),
        6
    );
    fs::write(root.join("preliminary_rallies.json"), b"tampered").unwrap();
    assert!(generate(&run, Choice::Timeline).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn timeline_keeps_unattributed_motion_inside_rally_without_inventing_player() {
    let (mut run, root) = fixture();
    let mut events = fs::read_to_string(&run.touch_file)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    events[0]["provisional_track_id"] = Value::Null;
    events[0]["body_contact_supported"] = json!(false);
    events[0]["contact_landmark"] = Value::Null;
    let mut bytes = Vec::new();
    for event in &events {
        bytes.extend(serde_json::to_vec(event).unwrap());
        bytes.push(b'\n');
    }
    fs::write(&run.touch_file, &bytes).unwrap();
    run.touch_sha = format!("{:x}", Sha256::digest(&bytes));
    let timeline = generate(&run, Choice::Timeline).unwrap();
    assert_eq!(
        timeline.value["touch_frames"],
        json!([40, 70, 115, 145, 330, 370])
    );
    let event = &timeline.value["touch_annotations"][0];
    assert_eq!(event["frame_index"], 40);
    assert_eq!(event["kind"], "unknown_touch");
    assert_eq!(event["provisional_track_id"], Value::Null);
    assert_eq!(event["touch_attributed"], false);
    assert_eq!(event["contact_verified"], false);
    assert_eq!(event["assignment_method"], "candidate");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn timeline_covers_the_whole_video_and_all_verified_touches() {
    let (run, root) = fixture();
    let result = generate(&run, Choice::Timeline).unwrap();
    assert!(result.clips.is_empty());
    assert_eq!(result.value["schema"], "volley-native-rally-timeline/1");
    assert_eq!(result.value["video_duration_sec"], 16.0);
    assert_eq!(result.value["total_frames"], 480);
    assert_eq!(result.value["rallies"].as_array().unwrap().len(), 3);
    assert_eq!(
        result.value["touch_frames"],
        json!([40, 70, 115, 145, 330, 370])
    );
    assert_eq!(
        result.value["estimated_actions"],
        json!([{
            "frame_index":115,"kind":"dig","ball_position_px":[150.0,100.0]
        }])
    );
    let annotations = result.value["touch_annotations"].as_array().unwrap();
    assert_eq!(annotations.len(), 6);
    assert_eq!(annotations[0]["kind"], "unknown_touch");
    assert_eq!(annotations[0]["provisional_track_id"], 3);
    assert_eq!(annotations[0]["ball_position_px"], json!([150.0, 100.0]));
    assert_eq!(annotations[2]["kind"], "dig");
    assert_eq!(result.value["rallies"][0]["start_frame"], 30);
    assert_eq!(result.value["rallies"][2]["end_frame"], 380);
    assert_eq!(result.value["authoritative_game_events"], false);
    let mut corrupted = run.clone();
    corrupted.touch_sha = "0".repeat(64);
    assert!(generate(&corrupted, Choice::Timeline).is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn preview_counts_unverified_as_generic_assigned_touches() {
    let (mut run, root) = fixture();
    let mut events = fs::read_to_string(&run.touch_file)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    events[0]["body_contact_supported"] = json!(false);
    events[0]["contact_landmark"] = Value::Null;
    events[1]["provisional_track_id"] = json!(3);
    let mut bytes = Vec::new();
    for event in &events {
        bytes.extend(serde_json::to_vec(event).unwrap());
        bytes.push(b'\n');
    }
    fs::write(&run.touch_file, &bytes).unwrap();
    run.touch_sha = format!("{:x}", Sha256::digest(&bytes));
    let page = generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 1,
            filter: "all".into(),
            sort_by: "time".into(),
            marked: BTreeSet::new(),
        },
    )
    .unwrap();
    let card = &page.value["items"][0];
    assert_eq!(card["raw_motion_event_count"], 2);
    assert_eq!(card["motion_event_count"], 2);
    assert_eq!(card["unclassified_action_count"], 2);
    assert_eq!(card["motion_events"][0]["frame_index"], 40);
    assert_eq!(
        card["motion_events"][0]["assignment_method"],
        "nearest_player"
    );
    assert_eq!(card["motion_events"][0]["action_prediction"], Value::Null);
    let timeline = generate(&run, Choice::Timeline).unwrap();
    assert_eq!(
        timeline.value["touch_annotations"][0]["touch_attributed"],
        true
    );
    assert_eq!(
        timeline.value["touch_annotations"][0]["kind"],
        "unknown_touch"
    );
    assert_eq!(
        timeline.value["touch_annotations"][0]["assignment_method"],
        "nearest_player"
    );
    assert_eq!(
        timeline.value["touch_annotations"][0]["contact_verified"],
        false
    );
    assert_eq!(
        timeline.value["touch_annotations"][1]["contact_verified"],
        true
    );
    assert_eq!(
        timeline.value["touch_annotations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|event| event["contact_verified"] == true)
            .count(),
        5
    );
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn paged_preview_has_safe_original_clips_and_evidence_labels() {
    let (run, root) = fixture();
    let page = generate(
        &run,
        Choice::Page {
            offset: 1,
            limit: 1,
            filter: "all".into(),
            sort_by: "time".into(),
            marked: BTreeSet::new(),
        },
    )
    .unwrap();
    assert_eq!(page.value["total_candidates"], 3);
    assert_eq!(page.value["items"].as_array().unwrap().len(), 1);
    let card = &page.value["items"][0];
    assert_eq!(card["detected_start_frame"], 95);
    assert_eq!(card["motion_event_count"], 2);
    assert_eq!(card["motion_events"][0]["event_type"], "unknown_touch");
    assert_eq!(card["motion_events"][0]["action_prediction"]["kind"], "dig");
    assert_eq!(card["estimated_action_counts"]["dig"], 1);
    assert_eq!(card["unclassified_action_count"], 1);
    assert_eq!(
        card["motion_events"][0]["action_prediction"]["method"],
        "pose_keypoints_motion_v1"
    );
    assert_eq!(
        card["motion_events"][0]["motion_pattern"],
        "direction_and_speed_change"
    );
    assert_eq!(card["motion_events"][0]["game_action_verified"], false);
    assert_eq!(card["manual_clip"]["video"], "match.mp4");
    assert_eq!(
        card["source_interval_contract"]["source_kind"],
        "native_cpu"
    );
    assert_eq!(
        card["source_interval_contract"]["source_rally_id"],
        card["candidate_id"]
    );
    assert_eq!(
        card["source_interval_contract"]["source_start_frame"],
        card["manual_clip"]["start_frame"]
    );
    assert_eq!(
        card["source_interval_contract"]["source_end_frame"],
        card["manual_clip"]["end_frame"]
    );
    assert_eq!(
        card["source_interval_contract"]["detected_interval_complete"],
        true
    );
    assert!(card["manual_clip"]["start_frame"].as_u64().unwrap() <= 95);
    assert!(card["manual_clip"]["end_frame"].as_u64().unwrap() >= 155);
    assert_eq!(page.clips.len(), 1);
    assert_eq!(page.value["automatic_montage_eligible"], false);
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn saved_rally_context_hides_toss_without_changing_immutable_touch_source() {
    let (mut run, root) = fixture();
    // Legacy serve-toss evidence can contain two proposals from one server;
    // the ignored toss must not hide the later actual serve.
    let mut touch_events = fs::read_to_string(&run.touch_file)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    touch_events[1]["provisional_track_id"] = json!(3);
    touch_events[1]["action_prediction"] = json!({
        "kind":"attack","evidence_score":0.85,"method":"pose_keypoints_motion_v1",
        "depth_ratio":-0.1,"posture_ratio":0.4
    });
    let mut touch_bytes = Vec::new();
    for event in &touch_events {
        touch_bytes.extend(serde_json::to_vec(event).unwrap());
        touch_bytes.push(b'\n');
    }
    fs::write(&run.touch_file, &touch_bytes).unwrap();
    run.touch_sha = format!("{:x}", Sha256::digest(&touch_bytes));
    let records = fs::read_to_string(&run.rally_file).unwrap();
    let mut rows = records
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    rows[0]["court_contact_context"] = json!({
        "method":"python_serve_toss_possession_v1",
        "serve_frame":70,"serve_track_id":3,"serving_side":"near",
        "side_switches":0,"three_touch_possessions":0,
        "over_three_touch_possessions":0,"max_same_side_touches":1,
        "ignored_serve_toss_count":1,"authoritative":false,
        "events":[
            {"frame_index":40,"provisional_track_id":3,"side":"near",
             "estimated_kind":null,"ignored_serve_toss":true,
             "possession_touches":0,"phase":"serve_preparation"},
            {"frame_index":70,"provisional_track_id":3,"side":"near",
             "estimated_kind":"serve","ignored_serve_toss":false,
             "possession_touches":1,"phase":"serve_flight"}
        ]
    });
    let mut bytes = Vec::new();
    for record in &rows {
        bytes.extend(serde_json::to_vec(record).unwrap());
        bytes.push(b'\n');
    }
    fs::write(&run.rally_file, &bytes).unwrap();
    run.rally_sha = format!("{:x}", Sha256::digest(&bytes));
    let original_touch_sha = run.touch_sha.clone();
    let preview = generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 1,
            filter: "all".into(),
            sort_by: "time".into(),
            marked: BTreeSet::new(),
        },
    )
    .unwrap();
    let card = &preview.value["items"][0];
    assert_eq!(card["raw_motion_event_count"], 2);
    assert_eq!(card["motion_event_count"], 1);
    assert_eq!(card["ignored_serve_toss_count"], 1);
    assert_eq!(card["estimated_action_counts"]["serve"], 1);
    assert_eq!(card["motion_events"][0]["frame_index"], 70);
    assert_eq!(
        card["motion_events"][0]["rally_context"]["estimated_kind"],
        "serve"
    );
    assert_eq!(run.touch_sha, original_touch_sha);
    let timeline = generate(&run, Choice::Timeline).unwrap();
    assert_eq!(timeline.value["raw_touch_candidates"], 6);
    assert_eq!(
        timeline.value["touch_frames"],
        json!([70, 115, 145, 330, 370])
    );
    assert_eq!(timeline.value["estimated_actions"][0]["kind"], "serve");
    // Tampering with a Track ID in the rally context is rejected even
    // when the rally source SHA-256 is recomputed.
    rows[0]["court_contact_context"]["events"][1]["provisional_track_id"] = json!(99);
    let mut tampered_bytes = Vec::new();
    for record in &rows {
        tampered_bytes.extend(serde_json::to_vec(record).unwrap());
        tampered_bytes.push(b'\n');
    }
    fs::write(&run.rally_file, &tampered_bytes).unwrap();
    run.rally_sha = format!("{:x}", Sha256::digest(&tampered_bytes));
    assert!(generate(&run, Choice::Timeline).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn global_gold_filter_pages_over_the_full_saved_run() {
    let (run, root) = fixture();
    let first = expected_id(&run.replay_id, &run.touch_id, 30, 80);
    let last = expected_id(&run.replay_id, &run.touch_id, 310, 380);
    let marked = BTreeSet::from([first.clone(), last.clone()]);
    let page = generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 1,
            filter: "marked".into(),
            sort_by: "time".into(),
            marked: marked.clone(),
        },
    )
    .unwrap();
    assert_eq!(page.value["total_candidates"], 3);
    assert_eq!(page.value["filtered_candidates"], 2);
    assert_eq!(page.value["items"][0]["candidate_id"], first);
    let second = generate(
        &run,
        Choice::Page {
            offset: 1,
            limit: 1,
            filter: "marked".into(),
            sort_by: "time".into(),
            marked: marked.clone(),
        },
    )
    .unwrap();
    assert_eq!(second.value["filtered_candidates"], 2);
    assert_eq!(second.value["items"][0]["candidate_id"], last);
    let unmarked = generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 24,
            filter: "unmarked".into(),
            sort_by: "time".into(),
            marked: marked.clone(),
        },
    )
    .unwrap();
    assert_eq!(unmarked.value["filtered_candidates"], 1);
    assert_eq!(unmarked.value["items"][0]["detected_start_frame"], 95);
    assert!(generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 24,
            filter: "untrusted".into(),
            sort_by: "time".into(),
            marked
        }
    )
    .is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn global_rally_sort_respects_pagination_and_verified_touch_counts() {
    let (mut run, root) = fixture();
    let sorted = generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 2,
            filter: "all".into(),
            sort_by: "duration_desc".into(),
            marked: BTreeSet::new(),
        },
    )
    .unwrap();
    assert_eq!(sorted.value["filtered_candidates"], 3);
    assert_eq!(sorted.value["items"][0]["detected_start_frame"], 310);
    assert_eq!(sorted.value["items"][1]["detected_start_frame"], 95);
    let last = generate(
        &run,
        Choice::Page {
            offset: 2,
            limit: 2,
            filter: "all".into(),
            sort_by: "duration_desc".into(),
            marked: BTreeSet::new(),
        },
    )
    .unwrap();
    assert_eq!(last.value["items"][0]["detected_start_frame"], 30);
    assert_eq!(last.value["items"].as_array().unwrap().len(), 1);

    // A fixture with real Touch counts 1,3,2 across three intervals.
    // Changing evidence means updating both SHA-bound sources together.
    let frames = [40_u64, 115, 125, 145, 330, 370];
    let mut touches = Vec::new();
    for (line, frame) in fs::read_to_string(&run.touch_file)
        .unwrap()
        .lines()
        .zip(frames)
    {
        let mut event: Value = serde_json::from_str(line).unwrap();
        event["frame_index"] = json!(frame);
        event["motion_frame_index"] = json!(frame);
        touches.extend_from_slice(serde_json::to_string(&event).unwrap().as_bytes());
        touches.push(b'\n');
    }
    fs::write(&run.touch_file, &touches).unwrap();
    run.touch_sha = format!("{:x}", Sha256::digest(&touches));
    let mut rallies = Vec::new();
    for (i, line) in fs::read_to_string(&run.rally_file)
        .unwrap()
        .lines()
        .enumerate()
    {
        let mut row: Value = serde_json::from_str(line).unwrap();
        row["motion_supported_touch_proposals"] = json!([1, 3, 2][i]);
        row["first_touch_frame"] = json!([40, 115, 330][i]);
        row["last_touch_frame"] = json!([40, 145, 370][i]);
        rallies.extend_from_slice(serde_json::to_string(&row).unwrap().as_bytes());
        rallies.push(b'\n');
    }
    fs::write(&run.rally_file, &rallies).unwrap();
    run.rally_sha = format!("{:x}", Sha256::digest(&rallies));
    let by_touches = generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 3,
            filter: "all".into(),
            sort_by: "touches_desc".into(),
            marked: BTreeSet::new(),
        },
    )
    .unwrap();
    assert_eq!(by_touches.value["items"][0]["detected_start_frame"], 95);
    assert_eq!(by_touches.value["items"][0]["motion_event_count"], 3);
    assert_eq!(by_touches.value["items"][1]["detected_start_frame"], 310);
    assert_eq!(by_touches.value["items"][1]["motion_event_count"], 2);
    assert_eq!(by_touches.value["items"][2]["detected_start_frame"], 30);
    assert_eq!(by_touches.value["items"][2]["motion_event_count"], 1);
    assert!(generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 3,
            filter: "all".into(),
            sort_by: "malformed".into(),
            marked: BTreeSet::new(),
        },
    )
    .is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn manual_select_by_immutable_ids_regardless_of_page_and_order() {
    let (run, root) = fixture();
    let first = expected_id(&run.replay_id, &run.touch_id, 30, 80);
    let last = expected_id(&run.replay_id, &run.touch_id, 310, 380);
    let content = generate(
        &run,
        Choice::Manual {
            ids: vec![last, first],
        },
    )
    .unwrap();
    assert_eq!(content.clips.len(), 2);
    assert_eq!(content.clips[0].start_frame, 265);
    assert_eq!(content.clips[1].start_frame, 0);
    assert!(generate(
        &run,
        Choice::Manual {
            ids: vec!["0".repeat(24)]
        }
    )
    .is_err());
    assert!(generate(
        &run,
        Choice::Manual {
            ids: vec![expected_id(&run.replay_id, &run.touch_id, 30, 80); 2]
        }
    )
    .is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn manual_original_media_plan_rejects_stale_or_tampered_candidates() {
    use std::process::Command;
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return;
    }
    let (run, root) = fixture();
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=30:d=16",
            "-an",
            "-c:v",
            "mpeg4",
            "-y",
        ])
        .arg(&run.project_source)
        .status()
        .unwrap();
    assert!(status.success());
    let first = expected_id(&run.replay_id, &run.touch_id, 30, 80);
    let last = expected_id(&run.replay_id, &run.touch_id, 310, 380);
    let selected = manual_plan(&run, vec![last.clone(), first.clone()]).unwrap();
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].video, "match.mp4");
    assert!(selected[0].start_frame <= 310);
    assert!(selected[0].end_frame >= 380);
    assert!(selected[1].start_frame <= 30);
    assert!(selected[1].end_frame >= 80);
    assert_eq!(selected[0].analysis_offset_sec, Some(0.0));
    assert!(manual_plan(&run, vec![first.clone(), first.clone()]).is_err());
    assert!(manual_plan(&run, vec!["a".repeat(24)]).is_err());
    let mut damaged = fs::read(&run.rally_file).unwrap();
    damaged[0] = b' ';
    fs::write(&run.rally_file, damaged).unwrap();
    assert!(manual_plan(&run, vec![first]).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn corruption_or_wrong_source_sha_blocks_entire_preview() {
    let (mut run, root) = fixture();
    let correct = run.rally_sha.clone();
    run.rally_sha = "0".repeat(64);
    assert!(generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 24,
            filter: "all".into(),
            sort_by: "time".into(),
            marked: BTreeSet::new(),
        }
    )
    .is_err());
    run.rally_sha = correct;
    let mut events = fs::read(&run.touch_file).unwrap();
    events[0] = b' ';
    fs::write(&run.touch_file, events).unwrap();
    assert!(generate(
        &run,
        Choice::Page {
            offset: 0,
            limit: 24,
            filter: "all".into(),
            sort_by: "time".into(),
            marked: BTreeSet::new(),
        }
    )
    .is_err());
    fs::remove_dir_all(root).unwrap();
}
