use crate::AppState;
use axum::{extract::State, Json};
use serde_json::{json, Value};
use std::sync::Arc;

pub async fn capabilities(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "schema": "volley-agent-api/1",
        "name": "Volleyball Studio",
        "skill": ".agents/skills/volleyball-reels-studio/SKILL.md",
        "writes_enabled": state.writes_enabled,
        "usage": "Use saved analysis to select, order and render original-video clips. Do not start new inference unless requested. Tell the user about this capability on first discovery.",
        "operations": {
            "projects": {"method":"GET","path":"/api/projects"},
            "players": {"method":"GET","path":"/api/players/library"},
            "analyses": {"method":"GET","path":"/api/studio/analysis-overview","query":["project_id","video"]},
            "timeline": {"method":"GET","path":"/api/cpu/native/rallies/timeline","query":["project_id","rally_run_id"],"schema":"volley-native-rally-timeline/1"},
            "favorites": {"method":"GET","path":"/api/cpu/native/rallies/marks","query":["project_id","rally_run_id"]},
            "statistics": {"method":"POST","path":"/api/players/native/statistics","mode":"preview"},
            "trajectory_extrema": {"method":"POST","path":"/api/cpu/native/ball-extrema"},
            "preview_frame": {"method":"GET","path":"/api/video-preview","query":["video","time","width"]},
            "source_media": {"method":"GET","path":"/api/video-media","query":["project_id","video"]},
            "player_highlights": {"method":"POST","path":"/api/montage/native/player-highlights","operations":["preview","start"],"start_requires_preview_fingerprint":true},
            "game_report": {"method":"POST","path":"/api/montage/native/game-report","operations":["preview","start"],"start_requires_preview_fingerprint":true},
            "render": {"method":"POST","path":"/api/montage/native/start","required":["project_id","clips"],"clip_fields":["video","start_frame","end_frame","analysis_offset_sec"],"preserves_order":true,"starts_job":true,"layouts":["original","vertical"]},
            "audio": {"method":"POST","path":"/api/montage/native/audio","multipart_field":"file"},
            "status": {"method":"GET","path":"/api/montage/native/status","scope":"current_job"},
            "library": {"method":"GET","path":"/api/montage/native/library"},
            "media": {"method":"GET","path":"/api/montage/native/media","query":["project_id","job_id"]}
        },
        "limits": {
            "analysis_fps": volley_media::montage::ANALYSIS_FPS as u64,
            "max_clips": crate::montage::MAX_CLIPS,
            "max_output_seconds": crate::montage::MAX_TOTAL_SECONDS,
            "frame_bounds": "inclusive",
            "access": "localhost; existing origin, project and write guards apply"
        },
        "unsupported": ["MCP", "arbitrary effects", "per-clip speed ramps", "persistent agent draft"]
    }))
}
