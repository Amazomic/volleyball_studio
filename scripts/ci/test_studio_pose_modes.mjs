import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
assert.match(html,/id="studio-gpu-person-stride"[^>]*min="1"[^>]*max="30"[^>]*value="10"[^>]*data-default="10"/);
assert.match(html,/30 кадров в секунду \(30 FPS\)/);
assert.match(html,/Каждый 10-й кадр → 3 раза\/с/);
assert.doesNotMatch(html,/id="studio-gpu-pose-mode"/);

const start=source.indexOf("  const refreshPersonSamplingRate=()=>");
const end=source.indexOf("  // Mirrors volley-core",start);
assert.ok(start>=0&&end>start,"UI profile must use shared detector/pose sampling");
const rate={value:"10"};
const display={textContent:""};
const fields={
  "studio-gpu-person-stride":rate,
  "studio-gpu-person-rate":display,
  "studio-gpu-pose-enabled":{value:"1"},
  "studio-gpu-interpolate-players":{checked:true},
  "studio-gpu-interpolate-pose":{checked:true}
};
const values={"studio-gpu-stride":9,"studio-gpu-memory":0,"studio-gpu-threads":8,
  "studio-gpu-person-stride":10};
const ctx={$:id=>fields[id],inputValue:id=>values[id]};
runInNewContext(source.slice(start,end)+
  "\nthis.test={refreshPersonSamplingRate,gpuProfile};",ctx);
const api=ctx.test;
for(const [step,expected] of [
  [1,"30 раз/с"],[2,"15 раз/с"],[3,"10 раз/с"],
  [5,"6 раз/с"],[7,"4,29 раза/с"],[10,"3 раза/с"],[15,"2 раза/с"],[30,"1 раз/с"]
]){
  values["studio-gpu-person-stride"]=step;
  rate.value=String(step);
  api.refreshPersonSamplingRate();
  assert.equal(api.gpuProfile().pose_stride,step);
  assert.ok(display.textContent.includes("Каждый "+step+"-й кадр"),display.textContent);
  assert.ok(display.textContent.includes(expected),display.textContent);
  assert.ok(display.textContent.includes("30 кадров в секунду"),display.textContent);
}
rate.value="31";
api.refreshPersonSamplingRate();
assert.match(display.textContent,/от 1 до 30/);
values["studio-gpu-person-stride"]=10;
fields["studio-gpu-pose-enabled"].value="0";
assert.equal(api.gpuProfile().pose_enabled,false);
console.log("Studio shared RF-DETR/RTMPose sampling: steps 1–30, displayed rates and GPU payload passed.");
