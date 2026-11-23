import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const handlers=readFileSync(new URL("../../rust/volley-app/src/native_player_tracks/reid_handlers.rs",import.meta.url),"utf8");
const start=handlers.indexOf("pub async fn reid_review(");
const end=handlers.indexOf('if req.action == "refresh_matches"',start);
assert.ok(start>0&&end>start);
const auto=handlers.slice(start,end);
const cached=auto.indexOf("save_cached_automatic_photo_links(");
const strict=auto.indexOf('if req.action == "auto_assign"');
const expensive=auto.indexOf("photo_comparison::compare(");
assert.ok(cached>=0&&strict>cached&&expensive>strict,
  "Manual auto-assign must return a cache miss before entering full photo inference");
assert.match(auto.slice(strict,expensive),/return Err\(storage::conflict/);
assert.match(auto.slice(strict,expensive),/Обновить признаки/);
assert.match(auto,/postprocess_auto_assign/,"Postprocessing may compute new scores on a fresh run");

const js=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const button=js.slice(js.indexOf("  const autoAssignSimilar=async"),js.indexOf("  const refreshPhotoMatches=async"));
assert.match(button,/action:"auto_assign"/);
assert.doesNotMatch(button,/action:"refresh_matches"|refreshPhotoMatches\(/,
  "The auto-link button must never start a separate ReID comparison");
console.log("Manual auto-assignment uses saved scores only; explicit refresh owns inference: OK");
