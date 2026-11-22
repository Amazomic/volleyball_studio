import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const read = path => readFileSync(new URL("../../" + path, import.meta.url), "utf8");
const html = read("rust/volley-app/static/studio.html");
const css = read("rust/volley-app/static/studio.css");
const js = read("rust/volley-app/static/studio.js");
const i18n = read("rust/volley-web/src/studio_i18n.tsv");

assert.match(html, /id="studio-nav-training"[^>]*>[^<]*<span>[^<]*<\/span> Дебаг<\/a>/);
assert.match(html, /class="studio-training-caption">Дебаг<\/span>/);
assert.ok(i18n.includes("Дебаг\tDebug"));

const viewerHead = html.indexOf('<div class="viewer-head">');
const toolbar = html.indexOf('class="studio-training-video-actions"', viewerHead);
const panel = html.indexOf('id="studio-training-panel"', toolbar);
const settings = html.indexOf('id="studio-debug-settings"', panel);
const video = html.indexOf('<div class="video-wrap">', viewerHead);
const timeline = html.indexOf('id="studio-timeline"');
assert.ok(viewerHead >= 0 && toolbar > viewerHead && panel > toolbar);
assert.ok(settings > panel && settings < video, "Debug settings must be above the player");
assert.ok(panel < video && video < timeline, "Nothing should be placed below the timeline");
assert.ok(html.indexOf('id="studio-annotation-save"', panel) < video);
assert.ok(html.indexOf('id="studio-comparison-min-seconds"', panel) < video);
assert.ok(html.indexOf('id="studio-training-pose-advanced"', panel) < video);
assert.ok(html.indexOf('id="studio-temporal-labels"') === -1);
assert.ok(html.indexOf('class="studio-temporal-labels"', panel) < video);
assert.ok(!html.slice(timeline).includes('id="studio-training-panel"'));

for (const id of ["studio-comparison-recalculate", "studio-annotation-calculate", "studio-annotation-classify"]) {
  assert.ok(html.indexOf('id="' + id + '"', toolbar) < panel, "Calculation buttons must remain adjacent to player: " + id);
}
assert.ok(css.includes(".studio-debug-settings-body"));
assert.ok(css.includes(".studio-debug-settings-body .studio-temporal-labels"));
assert.ok(!css.includes('grid-area:training'));
assert.ok(!js.includes('layout.append(training.panel)'), "Do not move debug tools below the timeline");
const internalOptions = html.match(/<div hidden aria-hidden="true" id="studio-training-internal-options">([\s\S]*?)<\/div>/);
assert.ok(internalOptions, "Obsolete debug controls must be absent from the visible UI");
for (const id of ["studio-training-merge-gap-seconds", "studio-training-ball-touch-candidates",
  "studio-training-ball-touch-types", "studio-training-ball-touch-provenance",
  "studio-training-ball-touch-types-status"]) {
  assert.ok(internalOptions[1].includes('id="' + id + '"'), "Keep the internal state binding: " + id);
}
assert.ok(!html.includes('Объединять при паузе меньше'));
assert.ok(!html.includes('<span>Определение касания</span>'));
assert.ok(!html.includes('<span>Определение типа касания</span>'));
assert.match(internalOptions[1], /id="studio-training-ball-touch-candidates" checked/);
assert.match(internalOptions[1], /id="studio-training-ball-touch-types" checked/);
assert.ok(js.includes('training.panel.classList.remove("hidden")'));
assert.ok(js.includes('annotation.save.addEventListener("click"'));

console.log("Debug UI: player controls, collapsible settings, obsolete panels hidden, timeline clean: OK");
