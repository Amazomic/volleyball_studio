import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {join} from "node:path";

const repo=fileURLToPath(new URL("../..",import.meta.url));
const html=readFileSync(join(repo,"rust/volley-app/static/studio.html"),"utf8");
const css=readFileSync(join(repo,"rust/volley-app/static/studio.css"),"utf8");
const editor=readFileSync(join(repo,"rust/volley-app/static/studio_editor.js"),"utf8");
const stageStart=html.indexOf('id="studio-analysis-stage"');
const timelineStart=html.indexOf('class="analysis-editor-timeline"',stageStart);
assert.ok(stageStart>=0&&timelineStart>stageStart);
const stage=html.slice(stageStart,timelineStart);
const zoom=stage.indexOf('class="analysis-editor-zoom"');
assert.ok(zoom>stage.indexOf('id="studio-analysis-court"'),"Zoom toolbar must overlay court canvas inside video stage");
for(const id of ["studio-analysis-zoom-out","studio-analysis-zoom-in","studio-analysis-zoom-label"])
  assert.equal((html.match(new RegExp('id="'+id+'"','g'))||[]).length,1);
const actions=html.slice(html.indexOf('class="analysis-editor-actions"',timelineStart),html.indexOf('id="studio-analysis-edit-status"',timelineStart));
assert.ok(!actions.includes("analysis-editor-zoom"),"Zoom toolbar must not consume a row in the settings");
const style=css.match(/\.analysis-editor-stage>\.analysis-editor-zoom\{([^}]+)\}/)?.[1]||"";
assert.match(style,/position:absolute/);
assert.match(style,/top:10px/);
assert.match(style,/right:10px/);
assert.match(style,/z-index:4/);
assert.match(style,/pointer-events:auto/);
assert.match(css,/\.analysis-editor-stage\.marking canvas\{pointer-events:auto\}/);
assert.match(editor,/zoomOut\.addEventListener\("click"/);
assert.match(editor,/zoomIn\.addEventListener\("click"/);
console.log("PASS: zoom controls overlay top-right of video and retain click handlers above court canvas");
