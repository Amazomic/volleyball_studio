# Rally card CPU Touch timeline · 2026-11-15

## Scope / file ownership
- `rust/volley-app/static/studio.js` — card progress lane, validated CPU Touch ticks, playback synchronization
- `rust/volley-app/static/studio.css` — stable reserved progress slot and color-coded ticks
- `scripts/ci/test_studio_rally_cards.mjs` — update existing rendering harness for new card dependencies
- `scripts/ci/test_studio_rally_card_progress.mjs` — new regression test

## Implementation
- Reserve a compact progress lane on **every** rally card to prevent layout shifts.
- Show progress and ticks only on the selected card (including paused inspection).
- Source the tick frames from the source-matched `/api/cpu/native/rallies/timeline` `touch_annotations`, filtered to attributed or contact-verified events; never use the truncated `motion_events` card preview.
- Use the exact `cpuTouchStyle` palette shared by the video overlay, including the beige untyped-touch fallback.
- Compute positions from played clip bounds (including a manual clip when present). Update progress on video time updates and decoded video frames.
- Clear stale ticks when the selected CPU run changes. Do not alter CPU evidence or statistics.

## Verification
- Passed: `node --check rust/volley-app/static/studio.js`
- Passed: `node scripts/ci/test_studio_rally_cards.mjs`
- Passed: `node scripts/ci/test_studio_rally_card_progress.mjs`
- Passed: `node scripts/ci/test_studio_preview_identity_labels.mjs`
- Passed: `node scripts/ci/test_studio_preview_source.mjs`
- Passed: `git diff --check` on touched tracked files
- Existing Rust Studio controller: PID 664388, idle, live development assets; verified `/studio/app.js` and `/studio/app.css` serve new code. No rebuild/restart needed for this asset-only change.
- Browser screenshot/pixel checks not executed. Other preexisting unrelated worktree modifications were left untouched. No commit made in the shared dirty checkout.

## Follow-up: interactive seeking (2026-11-15)
- `rust/volley-app/static/studio.js`: Clicking any horizontal point on the selected card's mini-timeline seeks to that proportional point inside the rally's actual selected clip, resumes playback and preserves the selected CPU source; keyboard arrows/Home/End/PageUp/PageDown are supported; focused mini-slider is excluded from global media shortcuts. Same-card seeking does not rebuild the card or lose keyboard focus.
- `rust/volley-app/static/studio.css`: 20px touch target, hover/focus feedback and explicit pointer affordance; colors and touch markers remain unchanged.
- `scripts/ci/test_studio_rally_cards.mjs`: Simulate pointer positioning, stopPropagation and keyboard End.
- `scripts/ci/test_studio_rally_card_progress.mjs`: Exercise real view() seeking, resume-from-pause and end-of-clip clamping with a mocked video element.
- Checks: `node --check`, `test_studio_rally_cards.mjs`, `test_studio_rally_card_progress.mjs`, `test_studio_preview_identity_labels.mjs`, `test_studio_preview_source.mjs`, and `git diff --check` passed.
- Rust release embeds its JS/CSS; existing live binary initially served older assets, requiring a safe release build and restart; active jobs checked via Studio controller before requesting restart.
- No modifications to gameplay evidence or user media.

## Deployment verification
- Release build succeeded; foreground restart was converted to a safe, owned detached Studio using the existing controller after idle/active-job check.
- Persistent Studio: PID 751354, http://127.0.0.1:8875/studio.
- Verified live `/studio/app.js` includes `playFromFraction` and live `/studio/app.css` includes the interactive mini-timeline styling.
- Final post-deploy tests: `test_studio_rally_card_progress.mjs`, `test_studio_rally_cards.mjs` passed.
- Existing `test_studio_overlay_continuity.mjs` unrelated fixture fails because its synthetic CPU annotations lack current `touch_attributed/contact_verified` flags; no test or gameplay data changes were made to work around it.

## Whole-video rally-band palette update (2026-11-15 02:26)
- `rust/volley-app/static/studio.js`: replace high-saturation cyan and gold band fills with muted slate-blue (#365469 default, #597a91 active) and low-glare amber (#736448 favorite, #8d7955 selected favorite), preserving the brightly colored CPU Touch ticks.
- `rust/volley-app/static/studio.css`: match the timeline legend swatch to #365469.
- Validation: `node --check`, `test_studio_rally_card_progress.mjs`, `test_studio_rally_cards.mjs`, `test_studio_preview_source.mjs`, and `git diff --check` passed.
- Built-in asset release rebuild/start requested through the existing safe `-Action Detach` launcher; check live response before declaring deployed.

- Deployment completed: safe release rebuild and detached restart via `scripts/start-volley-rust-dev.ps1`; live Studio PID 785327 on 127.0.0.1:8875.
- Live `/studio/app.js` serves muted rally colors, `/studio/app.css` serves matching legend. Post-deploy card tests passed. No user data or analysis jobs touched.
