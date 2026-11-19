//! In-process HTTP integration tests for the `/api/*` surface.
//!
//! Each test serves the real axum router on an ephemeral localhost port with
//! a disposable temporary data root, so the suite never touches the user's
//! live instance, repository projects or saved caches.

use super::*;
use std::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct TestServer {
    addr: SocketAddr,
    task: tokio::task::JoinHandle<()>,
    root: PathBuf,
    state: Arc<AppState>,
}

impl TestServer {
    async fn spawn(writes_enabled: bool) -> Self {
        let root = std::env::temp_dir().join(format!("volley-api-http-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let state = Arc::new(AppState {
            repository_root: root.join("original"),
            data_root: root.join("sandbox"),
            writes_enabled,
            montage: montage::MontageManager::new(),
            preparation: preparation::PreparationManager::new(),
            preview_limit: tokio::sync::Semaphore::new(1),
            touch_limit: tokio::sync::Semaphore::new(1),
            continuity_limit: tokio::sync::Semaphore::new(1),
            reid_limit: tokio::sync::Semaphore::new(1),
            identity_limit: tokio::sync::Semaphore::new(1),
            statistics_limit: tokio::sync::Semaphore::new(1),
            marks_limit: tokio::sync::Semaphore::new(1),
            quick_reel: quick_reel::QuickReelManager::new(),
            inference_limit: tokio::sync::Semaphore::new(1),
            full_inference: inference_full::FullInferenceManager::new(),
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let serving = state.clone();
        let task = tokio::spawn(async move {
            axum::serve(listener, app(serving)).await.unwrap();
        });
        Self {
            addr,
            task,
            root,
            state,
        }
    }

    fn seed_project(&self, project_id: &str) {
        fs::create_dir_all(self.root.join("sandbox/projects").join(project_id)).unwrap();
        storage::write_json_atomic(
            &self.root.join("sandbox/projects/projects.json"),
            &serde_json::json!({"schema":1,"projects":[
                {"id":project_id,"name":"Match","videos":["match.mp4"]}]}),
        )
        .unwrap();
    }

    async fn request(&self, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
        let mut stream = tokio::net::TcpStream::connect(self.addr).await.unwrap();
        let body = body.unwrap_or("");
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply).into_owned();
        let code = text
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        let body = text
            .split_once("\r\n\r\n")
            .map(|(_, body)| body.to_owned())
            .unwrap_or_default();
        (code, body)
    }

    async fn close(self) {
        self.task.abort();
        let _ = self.task.await;
        fs::remove_dir_all(self.root).unwrap();
    }
}

#[tokio::test]
async fn projects_api_returns_empty_list_on_fresh_root() {
    let server = TestServer::spawn(false).await;
    let (code, body) = server.request("GET", "/api/projects", None).await;
    assert_eq!(code, 200);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["projects"], serde_json::json!([]));
    server.close().await;
}

#[tokio::test]
async fn montage_cannot_start_while_project_deletion_holds_lock() {
    use fs2::FileExt;
    let server = TestServer::spawn(true).await;
    let id = "123456abcdef";
    server.seed_project(id);
    let locks = server.root.join("sandbox/projects/.project-locks");
    fs::create_dir_all(&locks).unwrap();
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(locks.join(format!("{id}.lock")))
        .unwrap();
    lock.lock_exclusive().unwrap();
    let body = format!(r#"{{"project_id":"{id}","clips":[]}}"#);
    let (code, _) = server
        .request("POST", "/api/montage/native/start", Some(&body))
        .await;
    assert_eq!(code, 409);
    FileExt::unlock(&lock).unwrap();
    drop(lock);
    server.close().await;
}

#[tokio::test]
async fn analysis_api_renames_and_deletes_linked_results() {
    let server = TestServer::spawn(true).await;
    let project = "123456abcdef";
    let fusion = "11111111111111111111111111111111";
    let replay = "22222222222222222222222222222222";
    let rally = "33333333333333333333333333333333";
    server.seed_project(project);
    let root = server.root.join("sandbox/projects").join(project);
    for (folder, id, value) in [
        (
            "native_gpu_fusions",
            fusion,
            serde_json::json!({"schema":"volley-native-gpu-fusion/1","fusion_id":fusion}),
        ),
        (
            "native_cpu_replays",
            replay,
            serde_json::json!({"schema":"volley-native-cpu-tracking/1","replay_id":replay,"source_fusion_id":fusion}),
        ),
        (
            "native_cpu_rallies",
            rally,
            serde_json::json!({"schema":"volley-native-cpu-rally-proposals/1","rally_run_id":rally,"source_replay_id":replay}),
        ),
    ] {
        let dir = root.join(folder).join(id);
        fs::create_dir_all(&dir).unwrap();
        let mut value = value;
        value["project_id"] = serde_json::json!(project);
        value["video"] = serde_json::json!("match.mp4");
        storage::write_json_atomic(&dir.join("manifest.json"), &value).unwrap();
    }
    let rename = serde_json::json!({"project_id":project,"video":"match.mp4","kind":"fusion","id":fusion,"action":"rename","name":"Мой анализ"}).to_string();
    let (code, _) = server
        .request("POST", "/api/studio/analysis-rename", Some(&rename))
        .await;
    assert_eq!(code, 200);
    assert_eq!(
        analysis_management::display_name(&root, "fusion", fusion)
            .unwrap()
            .as_deref(),
        Some("Мой анализ")
    );
    let delete = serde_json::json!({"project_id":project,"video":"match.mp4","kind":"fusion","id":fusion,"action":"delete"}).to_string();
    let (code, _) = server
        .request("POST", "/api/studio/analysis-rename", Some(&delete))
        .await;
    assert_eq!(code, 400);
    assert!(root.join("native_gpu_fusions").join(fusion).exists());
    assert!(root.join("native_cpu_replays").join(replay).exists());
    assert!(root.join("native_cpu_rallies").join(rally).exists());
    let (code, _) = server
        .request("POST", "/api/studio/analysis-edit", Some(&delete))
        .await;
    assert_eq!(code, 200);
    assert!(!root.join("native_gpu_fusions").join(fusion).exists());
    assert!(!root.join("native_cpu_replays").join(replay).exists());
    assert!(!root.join("native_cpu_rallies").join(rally).exists());
    server.close().await;
}

#[tokio::test]
async fn original_video_is_ready_without_intermediate_mp4() {
    let server = TestServer::spawn(true).await;
    server.seed_project("123456abcdef");
    let source = server.root.join("original/project/match.mp4");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=25:duration=0.6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.6",
            "-c:a",
            "aac",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap();
    assert!(status.success());
    let original = fs::read(&source).unwrap();
    let (code, _) = server
        .request(
            "POST",
            "/api/project-preparation/retry",
            Some(r#"{"project_id":"123456abcdef"}"#),
        )
        .await;
    assert_eq!(code, 200);
    let (code, body) = server
        .request(
            "GET",
            "/api/project-preparation?project_id=123456abcdef",
            None,
        )
        .await;
    assert_eq!(code, 200);
    let status: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(status["ready"], 1);
    assert_eq!(status["videos"][0]["preparation"], "not_required");
    for code in [0, 1, 2] {
        let body = serde_json::json!({
            "project_id":"123456abcdef", "video":"match.mp4", "browser_error_code":code
        })
        .to_string();
        let (code, body) = server
            .request("POST", "/api/project-preparation/retry", Some(&body))
            .await;
        assert_eq!(code, 200);
        let result: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(result["state"], "original");
    }
    let (code, body) = server
        .request(
            "GET",
            "/api/video-media?project_id=123456abcdef&video=match.mp4",
            None,
        )
        .await;
    assert_eq!(code, 200);
    assert_eq!(body, String::from_utf8_lossy(&original));
    assert!(!server.root.join("sandbox/prepared").exists());
    assert!(!server.root.join("sandbox/browser-playback-v1").exists());
    let failure = serde_json::json!({
        "project_id":"123456abcdef", "video":"match.mp4", "browser_error_code":3
    })
    .to_string();
    let (code, _) = server
        .request(
            "GET",
            "/api/video-media?project_id=123456abcdef&video=match.mp4&playback=compatible",
            None,
        )
        .await;
    assert_eq!(code, 409);
    let paused = server
        .root
        .join("sandbox/projects/123456abcdef/.preparation-paused");
    fs::write(&paused, b"paused").unwrap();
    let (code, body) = server
        .request("POST", "/api/project-preparation/retry", Some(&failure))
        .await;
    assert_eq!(code, 200);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).unwrap()["state"],
        "cancelled"
    );
    assert!(!server.root.join("sandbox/browser-playback-v1").exists());
    fs::remove_file(paused).unwrap();
    let (code, body) = server
        .request("POST", "/api/project-preparation/retry", Some(&failure))
        .await;
    assert_eq!(code, 200);
    let first: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(matches!(
        first["state"].as_str(),
        Some("queued" | "preparing" | "ready")
    ));
    let mut ready = serde_json::Value::Null;
    for _ in 0..200 {
        let (code, body) = server
            .request(
                "GET",
                "/api/project-preparation?project_id=123456abcdef&video=match.mp4&purpose=playback",
                None,
            )
            .await;
        assert_eq!(code, 200);
        ready = serde_json::from_str(&body).unwrap();
        if ready["state"] == "ready" {
            break;
        }
        assert_ne!(ready["state"], "failed", "{ready}");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(ready["state"], "ready");
    let media = ready["media_url"].as_str().unwrap();
    assert!(media.contains("playback=compatible"));
    let (code, _) = server.request("GET", media, None).await;
    assert_eq!(code, 200);
    let cache = server.root.join("sandbox/browser-playback-v1/prepared");
    let output = fs::read_dir(cache)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().is_some_and(|ext| ext == "mp4"))
        .unwrap();
    let probe = volley_media::ffprobe(&output).unwrap();
    let streams = probe["streams"].as_array().unwrap();
    assert!(streams
        .iter()
        .any(|s| s["codec_name"] == "h264" && s["pix_fmt"] == "yuv420p"));
    assert!(streams.iter().any(|s| s["codec_name"] == "aac"));
    let before = inference_stream::source_stamp(&output).unwrap();
    let repeated = failure.replace("\"browser_error_code\":3", "\"browser_error_code\":4");
    let (code, body) = server
        .request("POST", "/api/project-preparation/retry", Some(&repeated))
        .await;
    assert_eq!(code, 200);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).unwrap()["state"],
        "ready"
    );
    assert_eq!(inference_stream::source_stamp(&output).unwrap(), before);
    let analysis = preparation::plan_for(&server.state, "123456abcdef", "match.mp4").unwrap();
    assert_eq!(
        analysis.ready_path(),
        Some(source.canonicalize().unwrap().as_path())
    );
    assert_eq!(fs::read(source).unwrap(), original);
    server.close().await;
}

#[tokio::test]
async fn preview_serves_compatible_original_instead_of_downscaled_analysis_copy() {
    check_preview_source(30, true).await;
}

#[tokio::test]
async fn preview_prefers_original_until_browser_reports_decode_failure() {
    check_preview_source(25, true).await;
}

async fn check_preview_source(rate: u32, original: bool) {
    let server = TestServer::spawn(true).await;
    server.seed_project("123456abcdef");
    let source_dir = server.root.join("original/project");
    fs::create_dir_all(&source_dir).unwrap();
    let source = source_dir.join("match.mp4");
    let input = format!("testsrc2=size=1920x1080:rate={rate}");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &input,
            "-frames:v",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap();
    assert!(status.success());
    let plan = volley_media::preparation::PreparationPlan::new(
        &source,
        &server.root.join("sandbox"),
        None,
    )
    .unwrap();
    assert_eq!(plan.width, 1280);
    fs::create_dir_all(plan.output.parent().unwrap()).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args(plan.ffmpeg_arguments_with_hardware(&plan.output, false))
        .status()
        .unwrap();
    assert!(status.success());
    let (size, sha) = volley_media::preparation::prepared_file_sha256(&plan.output).unwrap();
    storage::write_json_atomic(&plan.manifest, &serde_json::json!({"status":"ready","identity":plan.identity,"cache_size_bytes":size,"cache_sha256":sha})).unwrap();
    let run = "0123456789abcdef0123456789abcdef";
    let run_root = server
        .root
        .join("sandbox/projects/123456abcdef/native_gpu_runs")
        .join(run);
    fs::create_dir_all(&run_root).unwrap();
    let (bytes, mtime) = inference_stream::source_stamp(&plan.output).unwrap();
    storage::write_json_atomic(
        &run_root.join("manifest.json"),
        &serde_json::json!({
            "source_identity":{"bytes":bytes,"mtime_ns":mtime}
        }),
    )
    .unwrap();
    let legacy = preparation::plan_for_run(
        &server.state,
        "123456abcdef",
        "match.mp4",
        "native_gpu_runs",
        run,
    )
    .unwrap();
    assert_eq!(legacy.ready_path(), Some(plan.output.as_path()));
    let (code, body) = server
        .request(
            "GET",
            "/api/video-media?project_id=123456abcdef&video=match.mp4",
            None,
        )
        .await;
    assert_eq!(code, 200);
    let expected = if original { &source } else { &plan.output };
    assert!(body == String::from_utf8_lossy(&fs::read(expected).unwrap()));
    server.close().await;
}

#[tokio::test]
async fn pipeline_status_is_idle_for_project_without_runs() {
    let server = TestServer::spawn(false).await;
    server.seed_project("123456abcdef");
    let (code, body) = server
        .request(
            "GET",
            "/api/studio/pipeline/status?project_id=123456abcdef",
            None,
        )
        .await;
    assert_eq!(code, 200);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["status"], "idle");
    server.close().await;
}

#[tokio::test]
async fn pipeline_status_rejects_unknown_project_with_404() {
    let server = TestServer::spawn(false).await;
    let (code, body) = server
        .request(
            "GET",
            "/api/studio/pipeline/status?project_id=ffffffffffff",
            None,
        )
        .await;
    assert_eq!(code, 404);
    assert!(body.contains("error"));
    server.close().await;
}

#[tokio::test]
async fn pipeline_status_rejects_malformed_project_id_with_400() {
    let server = TestServer::spawn(false).await;
    let (code, _) = server
        .request(
            "GET",
            "/api/studio/pipeline/status?project_id=..%2Fevil",
            None,
        )
        .await;
    assert_eq!(code, 400);
    server.close().await;
}

#[tokio::test]
async fn quick_reel_status_is_idle_on_writable_root() {
    let server = TestServer::spawn(true).await;
    let (code, body) = server
        .request("GET", "/api/quick-reel/native/status", None)
        .await;
    assert_eq!(code, 200);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["status"], "idle");
    assert_eq!(json["stage_total"], 4);
    server.close().await;
}

#[tokio::test]
async fn quick_reel_status_refuses_readonly_storage_with_409() {
    let server = TestServer::spawn(false).await;
    let (code, _) = server
        .request("GET", "/api/quick-reel/native/status", None)
        .await;
    assert_eq!(code, 409);
    server.close().await;
}

#[tokio::test]
async fn players_library_read_returns_empty_catalog_without_registry() {
    let server = TestServer::spawn(false).await;
    let (code, body) = server
        .request(
            "POST",
            "/api/players/library/native",
            Some(r#"{"action":"read"}"#),
        )
        .await;
    assert_eq!(code, 200);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["players"], serde_json::json!([]));
    assert_eq!(json["revision"], "absent");
    server.close().await;
}

#[tokio::test]
async fn pipeline_start_rejects_malformed_json_with_400() {
    let server = TestServer::spawn(false).await;
    let (code, _) = server
        .request("POST", "/api/studio/pipeline/start", Some("{not json"))
        .await;
    assert_eq!(code, 400);
    server.close().await;
}

#[tokio::test]
async fn unknown_api_route_returns_404() {
    let server = TestServer::spawn(false).await;
    let (code, _) = server
        .request("GET", "/api/definitely-not-a-route", None)
        .await;
    assert_eq!(code, 404);
    server.close().await;
}

#[tokio::test]
async fn primary_preview_discovery_is_per_video_and_does_not_prepare() {
    let server = TestServer::spawn(true).await;
    server.seed_project("123456abcdef");
    let (code, body) = server
        .request(
            "GET",
            "/api/video-primary-preview?project_id=123456abcdef&video=match.mp4",
            None,
        )
        .await;
    assert_eq!(code, 200, "{body}");
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(response["state"], "missing");
    assert_eq!(response["video"], "match.mp4");
    assert!(response["media_url"].is_null());
    let (code, _) = server
        .request(
            "GET",
            "/api/video-primary-preview?project_id=123456abcdef&video=other.mp4",
            None,
        )
        .await;
    assert_eq!(code, 400);
    assert!(!server.state.data_root.join("primary-playback-v1").exists());
    server.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn primary_preview_http_keeps_two_videos_separate_and_checks_version() {
    use volley_media::{
        preparation::primary_preview::PrimaryPreviewPlan,
        raw_decoder::{
            decode_full_video_shared_with_preview, DecodePreference, PrimaryPreviewOutput,
        },
    };
    let server = TestServer::spawn(true).await;
    let project = "123456abcdef";
    server.seed_project(project);
    storage::write_json_atomic(&server.root.join("sandbox/projects/projects.json"), &serde_json::json!({"schema":1,"projects":[{"id":project,"name":"Match","videos":["match.mp4","second.mp4"]}]})).unwrap();
    let folder = server.root.join("original/project");
    fs::create_dir_all(&folder).unwrap();
    let mut urls = Vec::new();
    for (name, duration) in [("match.mp4", "0.6"), ("second.mp4", "1.1")] {
        let source = folder.join(name);
        assert!(std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size=160x90:rate=30:duration={duration}"),
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-y"
            ])
            .arg(&source)
            .status()
            .unwrap()
            .success());
        let plan =
            PrimaryPreviewPlan::new(&source, &server.state.data_root, project, name).unwrap();
        let write = plan.begin().unwrap();
        decode_full_video_shared_with_preview(
            &source,
            9,
            DecodePreference::Software,
            2,
            &std::sync::atomic::AtomicBool::new(false),
            Some(PrimaryPreviewOutput {
                path: &write.path,
                hardware: false,
            }),
            |_| Ok(()),
            |_, _| Ok(()),
        )
        .unwrap();
        write
            .publish(&std::sync::atomic::AtomicBool::new(false))
            .unwrap();
        let (code, body) = server
            .request(
                "GET",
                &format!("/api/video-primary-preview?project_id={project}&video={name}"),
                None,
            )
            .await;
        assert_eq!(code, 200, "{body}");
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["video"], name);
        assert_eq!(value["state"], "ready");
        let url = value["media_url"].as_str().unwrap().to_owned();
        let (code, body) = server.request("GET", &url, None).await;
        assert_eq!(code, 200);
        assert_eq!(
            body,
            String::from_utf8_lossy(&fs::read(plan.ready_path().unwrap()).unwrap())
        );
        let (code,_)=server.request("GET",&format!("/api/video-primary-preview?project_id={project}&video={name}&media=1&preview_id=wrong"),None).await;
        assert_eq!(code, 409);
        urls.push(url);
    }
    assert_ne!(urls[0], urls[1]);
    let forged = urls[0].replace("match%2Emp4", "second%2Emp4");
    assert_eq!(server.request("GET", &forged, None).await.0, 409);
    server.close().await;
}

#[tokio::test]
async fn montage_library_lists_all_projects_and_deletes_only_selected_output() {
    let server = TestServer::spawn(true).await;
    let project_a = "123456abcdef";
    let project_b = "abcdef123456";
    let montage_a = format!("montage_{}", "a".repeat(32));
    let montage_b = format!("montage_{}", "b".repeat(32));
    let run = "c".repeat(32);
    fs::create_dir_all(server.root.join("sandbox/projects").join(project_a)).unwrap();
    fs::create_dir_all(server.root.join("sandbox/projects").join(project_b)).unwrap();
    storage::write_json_atomic(
        &server.root.join("sandbox/projects/projects.json"),
        &serde_json::json!({"projects":[
            {"id":project_a,"name":"Первый матч","videos":["match_a.mp4"]},
            {"id":project_b,"name":"Второй матч","videos":["match_b.mp4"]}
        ]}),
    )
    .unwrap();
    for (project, montage) in [(project_a, &montage_a), (project_b, &montage_b)] {
        let folder = server
            .root
            .join("sandbox/projects")
            .join(project)
            .join("montages")
            .join(montage);
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("montage.mp4"), vec![7_u8; 2048]).unwrap();
        storage::write_json_atomic(
            &folder.join("selection.json"),
            &serde_json::json!({
                "schema":"volley-rust-montage/1",
                "id":montage,
                "project_id":project,
                "render_source":"original_recording",
                "file":"montage.mp4",
                "duration_sec":32.0,
                "preferred_rally_run_id":run,
                "montage_materials":[{"rally_run_id":run}]
            }),
        )
        .unwrap();
    }
    let (code, body) = server
        .request("GET", "/api/montage/native/library", None)
        .await;
    assert_eq!(code, 200, "{body}");
    let found: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(found["videos"].as_array().unwrap().len(), 2);
    assert!(found["videos"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["project_id"] == project_a
            && item["project_name"] == "Первый матч"
            && item["source_run_ids"][0] == run
            && item["preferred_rally_run_id"] == run));
    let (code, body) = server
        .request(
            "GET",
            &format!("/api/montage/native/library?project_id={project_b}"),
            None,
        )
        .await;
    assert_eq!(code, 200, "{body}");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).unwrap()["videos"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let request = serde_json::json!({"project_id":project_a,"job_id":montage_a}).to_string();
    let (code, body) = server
        .request("POST", "/api/montage/native/delete", Some(&request))
        .await;
    assert_eq!(code, 200, "{body}");
    assert!(!server
        .root
        .join("sandbox/projects")
        .join(project_a)
        .join("montages")
        .join(&montage_a)
        .exists());
    assert!(server
        .root
        .join("sandbox/projects")
        .join(project_b)
        .join("montages")
        .join(&montage_b)
        .join("montage.mp4")
        .exists());
    let (code, body) = server
        .request("GET", "/api/montage/native/library", None)
        .await;
    assert_eq!(code, 200, "{body}");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).unwrap()["videos"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let (code, _) = server
        .request("POST", "/api/montage/native/delete", Some(&request))
        .await;
    assert_eq!(code, 404);
    let invalid = serde_json::json!({"project_id":project_b,"job_id":"../../evil"}).to_string();
    let (code, _) = server
        .request("POST", "/api/montage/native/delete", Some(&invalid))
        .await;
    assert!(code == 400 || code == 404);
    server.close().await;
}
