import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(
  new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const start=source.indexOf("  const rawDetectorKey=");
const end=source.indexOf("  const overlayWindowStart=",start);
assert.ok(start>=0&&end>start,"raw detector selector is present");
const modelName={
  textContent:"",classList:{add(){}},removeAttribute(){}
};
const label={
  querySelector:selector=>selector==="#studio-raw-person-model"||
    selector===".raw-source-detector-label"?modelName:null
};
const checkbox={checked:true,disabled:false,closest:()=>label};
const s={project:"project",video:"match.mp4",analysis:"rf",
  chain:{choices:[
    {value:"rf",kind:"fusion",id:"fusion-rf"},
    {value:"rt",kind:"fusion",id:"fusion-rt"}
  ]}};
const overlay={rawPeopleVisible:checkbox,rawDetectors:new Map([
  ["project:fusion-rf:match.mp4","rfdetr-medium"],
  ["project:fusion-rt:match.mp4","legacy"]
])};
const context={overlay,s,$:()=>null,rawFusion:()=>s.chain.choices.find(
  choice=>choice.value===s.analysis&&choice.kind==="fusion")||null};
runInNewContext(source.slice(start,end)+
  "\nthis.updateRawDetectorLabel=updateRawDetectorLabel;",context);
context.updateRawDetectorLabel();
assert.equal(modelName.textContent,"RF-DETR Medium");
assert.equal(checkbox.disabled,false);
checkbox.checked=false;
s.analysis="rt";
context.updateRawDetectorLabel();
assert.equal(modelName.textContent,"Архивный анализ");
assert.equal(checkbox.checked,false,"switching analysis preserves visibility choice");
s.analysis="missing";
context.updateRawDetectorLabel();
assert.equal(checkbox.disabled,true,"no selected GPU fusion disables the toggle");
assert.match(source,/if\(!overlay\.stabilizerView&&overlay\.rawPeopleVisible\.checked\)/,
  "The regular raw GPU preview keeps RF-DETR controls; trajectory-only comparison hides people");
assert.match(source,/const poseBoxes=raw\.poses\?\.length\?raw\.poses:\(raw\.poses_interpolated\|\|\[\]\)/,
  "unobserved detector frames reuse measured or bridged RTMPose crop boxes");
assert.match(source,/if\(player\.interpolated===true\)\{ctx\.setLineDash/,
  "CPU preview distinguishes interpolated track boxes");
assert.match(source,/rawDetectors\.set\(rawDetectorKey\(project,run,video\),value\.person_detector\)/);
console.log("Raw preview checkbox follows RF-DETR / archived saved GPU fusion: OK");