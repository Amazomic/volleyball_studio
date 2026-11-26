# Montage recipes for agents

Read [Agent HTTP API](agent-api.md). All requests start with project and saved analysis discovery; no GPU re-analysis merely for editing.

| User request | Procedure |
| --- | --- |
| All attacks by a named athlete | Resolve player_id from the roster. POST /api/montage/native/player-highlights with project_id, player_id, rally_run_ids, action_kind=attack, operation=preview, and the requested clip budget. Inspect selected_clips and provenance. Start with expected_source_fingerprint from that preview only. |
| Ten longest rallies | Read each selected timeline. Rank valid rallies by inclusive frame duration; stable tie-break by source and start. Keep at most ten, select an order, report durations and use custom render if requested. |
| Favorites from multiple videos | Read marks per run. Match candidate_ids with that run's rallies; preserve source and offset. Order as requested. |
| Blocks followed by serves | Filter touch_annotations by touch_attributed, action and optional resolved player_id. Prefer reviewed_action.action_kind only if action_verified=true, otherwise use kind as an estimate. Select context around frame_index, clamp to bounds and preserve source offset. |
| First attack, then full rally from another video | Combine the verified source clips in the specified order. Keep the sequence explicit; do not reorder by timestamp. |
| Replace music, remove a clip or change order | Modify the session's prior clip list; preserve source references, show revised duration, revalidate timeline and render a new file only when requested. |

For a short contact highlight, the Creative UI uses frame_index +/- 30 frames clamped to [0,total_frames-1]. This is a starting point, not a guarantee of action visibility. Review the surrounding video and use wider context when requested. A ball hitting the ground is not an athlete attack. Do not infer a winning attack merely from the last touch; verify available evidence or report uncertainty. Do not silently substitute other players, events or arbitrary clips to fill duration.

For an exact 30-second budget, choose approximately 900 inclusive analysis frames across the sequence, respecting complete actions. If the material is insufficient, disclose the shorter result. Output quality and exact duration are checked after export.
