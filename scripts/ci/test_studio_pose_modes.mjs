import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
assert.match(html,/RF-DETR и RTMPose-M синхронизированы: каждые 10 кадров \(3 раза\/с\)/);
assert.doesNotMatch(html,/id="studio-gpu-pose-mode"/);
assert.doesNotMatch(html,/id="studio-gpu-pose-stride"/);
const start=source.indexOf("  const gpuProfile=()=>");
const end=source.indexOf("  // Mirrors volley-core",start);
assert.ok(start>0&&end>start);
const fields={
  "studio-gpu-pose-enabled":{value:"1"},
  "studio-gpu-interpolate-players":{checked:true},
  "studio-gpu-interpolate-pose":{checked:true}
};
const values={"studio-gpu-stride":9,"studio-gpu-memory":0,"studio-gpu-threads":8};
const ctx={$:id=>fields[id],inputValue:id=>values[id]};
runInNewContext(source.slice(start,end)+"\nthis.test={gpuProfile};",ctx);
const profile=ctx.test.gpuProfile();
assert.equal(profile.pose_mode,"photo");
assert.equal(profile.pose_stride,10);
assert.equal(profile.pose_enabled,true);
fields["studio-gpu-pose-enabled"].value="0";
assert.equal(ctx.test.gpuProfile().pose_enabled,false);
console.log("Studio RTMPose: synchronized with 10-frame RF-DETR sampling; old mode/stride controls absent.");
