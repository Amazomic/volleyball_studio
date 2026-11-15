import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
assert.match(html,/<select id="studio-gpu-pose-mode" data-default="photo">/);
assert.match(html,/<option value="photo">/);
assert.match(html,/<option value="video">/);
assert.match(html,/id="studio-gpu-pose-stride"[^>]*max="30"[^>]*data-default="3"/);
const start=source.indexOf("  const updatePoseMode=()=>");
const end=source.indexOf("  // Mirrors volley-core",start);
assert.ok(start>0 && end>start);
const fields={
  "studio-gpu-pose-mode":{value:"photo"},
  "studio-gpu-pose-stride":{disabled:false},
  "studio-gpu-pose-step-field":{classList:{toggle(name,value){this.hidden=name==="hidden"&&value;}}},
  "studio-gpu-pose-enabled":{value:"1"},
  "studio-gpu-interpolate-players":{checked:true},
  "studio-gpu-interpolate-pose":{checked:true}
};
const values={
  "studio-gpu-stride":9,"studio-gpu-memory":0,
  "studio-gpu-threads":8,"studio-gpu-pose-stride":10
};
const ctx={$:id=>fields[id],inputValue:id=>values[id]};
runInNewContext(source.slice(start,end)+"\nthis.test={updatePoseMode,gpuProfile};",ctx);
let profile=ctx.test.gpuProfile();
assert.equal(profile.pose_mode,"photo");
assert.equal(profile.pose_stride,10);
ctx.test.updatePoseMode();
assert.equal(fields["studio-gpu-pose-stride"].disabled,false);
assert.equal(fields["studio-gpu-pose-step-field"].classList.hidden,false);
fields["studio-gpu-pose-mode"].value="video";
ctx.test.updatePoseMode();
profile=ctx.test.gpuProfile();
assert.equal(profile.pose_mode,"video");
assert.equal(profile.pose_stride,1);
assert.equal(fields["studio-gpu-pose-stride"].disabled,true);
assert.equal(fields["studio-gpu-pose-step-field"].classList.hidden,true);
fields["studio-gpu-pose-mode"].value="bad";
assert.throws(()=>ctx.test.gpuProfile(),/Выберите режим/);
console.log("Studio pose mode: photo stride, full-video stride and conditional UI passed.");
