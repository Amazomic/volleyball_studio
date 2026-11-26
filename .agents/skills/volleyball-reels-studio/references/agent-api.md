# Agent HTTP API v1

The agent interface is plain HTTP on the existing local Studio, not a separate service. Read [SKILL.md](../SKILL.md) and [recipes](montage-recipes.md).

## Discovery

```powershell
$base = 'http://127.0.0.1:8875'
Invoke-RestMethod "$base/api/agent/v1/capabilities"
```

Expect schema `volley-agent-api/1`. The response advertises existing routes, query names, supported operations and rendering limits. Discovery never starts work. If the route is absent, use the verified runtime contract only and report the version limitation. A false `writes_enabled` means no rendering on that instance. Keep the existing loopback/Host/Origin checks; do not expose the port to the network. A remote agent needs authorized execution on the Studio host.

## Read the saved analysis

| Operation | Method and path |
| --- | --- |
| Projects | GET /api/projects |
| Player library | GET /api/players/library |
| Analysis per video | GET /api/studio/analysis-overview?project_id=...&video=... |
| Rally and contact timeline | GET /api/cpu/native/rallies/timeline?project_id=...&rally_run_id=... |
| Favorite marks | GET /api/cpu/native/rallies/marks?project_id=...&rally_run_id=... |
| Statistics | POST /api/players/native/statistics with mode=preview, project_id, sources=[{rally_run_id}] |
| Ball extrema | POST /api/cpu/native/ball-extrema; fields in BallTrajectoryRequest |
| Source preview | GET /api/video-preview?video=...&time=...&width=320 |
| Original playback | GET /api/video-media?project_id=...&video=... |

URL-encode query values. Select one compatible saved rally run per source video, never mix revisions for the same video. The overview is metadata only; the timeline validates source evidence. Check schema `volley-native-rally-timeline/1`, project_id, rally_run_id, video, fps=30 and total_frames. Read rallies and touch_annotations; preserve candidate_id, frame_index, player_id, player_name, touch_attributed, reviewed_action, and available provenance. An automatic classification is not a manual confirmation. A Track ID is not a person. Match favorite candidate_ids within their own run. Missing marks are unknown, not empty.

## Ordered custom montage

The agent keeps the draft as a list of clips in its own session. The Rust renderer consumes the same list as the Creative UI. Preview individual source clips using video media; display the ordered list, durations and assumptions to the user before starting a requested render. There is no separate server-side dry run or persistent draft for custom clips. For semantic selection prefer the fingerprint-protected player/report preview/start APIs.

POST /api/montage/native/start with JSON (illustrative values, replace from a real timeline):

```json
{
  "project_id": "PROJECT_ID_FROM_API",
  "clips": [
    {"video":"first.mp4","start_frame":150,"end_frame":209,"analysis_offset_sec":0},
    {"video":"second.mp4","start_frame":300,"end_frame":389,"analysis_offset_sec":0}
  ],
  "layout": "vertical",
  "encoder": "auto",
  "audio_volume": 40,
  "include_recognition_data": false
}
```

Use exact source video names, inclusive analysis-frame bounds and the timeline's analysis_offset_sec for each clip. At 30 analysis FPS a clip has (end_frame - start_frame + 1)/30 seconds; original playback starts at offset + start_frame/30. Never exceed total_frames or fabricate a source. Limit 1–80 clips, 1–600 seconds of output; without a target duration the input must also fit 600 seconds. Custom clips use original or vertical layout; vertical_follow needs verified camera tracks via a supported semantic montage API. Optional audio_file must come from POST /api/montage/native/audio (multipart file). No audio_file preserves original sound. Do not set target_duration_sec unless intentionally requesting duration adjustment; choose the clip bounds to fit the user's budget.

Start only when the user asks for rendering. Confirm queued and record job_id. Poll GET /api/montage/native/status; it returns the current job, so check id and project_id before trusting it. Stop polling if it is replaced; inspect GET /api/montage/native/library for the exact completed job. Do not cancel or alter other jobs. On complete use the returned media_url; on failure report error. Verify actual output duration, geometry, sound and playback if available.

No custom VFX, per-clip retiming or persistent agent draft is provided by this contract. Do not invent endpoints. The existing Rust modules remain owners of validation, source mapping and rendering.
