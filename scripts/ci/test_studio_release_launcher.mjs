import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const launcher = readFileSync("scripts/volley-rust.sh", "utf8");
const html = readFileSync("rust/volley-app/static/studio.html", "utf8");
const tasks = readFileSync("rust/volley-app/static/studio_tasks.js", "utf8");
assert.match(launcher, /binary="\$repo\/rust\/target\/release\/volley-app"/);
assert.match(launcher, /cargo build --release --locked --offline -p volley-app/);
assert.match(launcher, /git -C "\$repo" symbolic-ref --quiet --short HEAD/);
assert.match(launcher, /git -C "\$repo" rev-parse --verify refs\/remotes\/origin\/develop/);
assert.match(launcher, /start\|detach\|restart\) require_develop_checkout/);
assert.match(launcher, /\[\[ "\$branch" == develop \]\]/);
assert.doesNotMatch(launcher, /profile="\$\{8/);
assert.doesNotMatch(html, /studio-restart-profile/);
assert.doesNotMatch(tasks, /restart\?profile=/);
assert.match(html, /id="studio-restart"/);
