import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const first=source.indexOf("    const current=frames.find(row=>row.frame===frame);",
  source.indexOf("    if(overlay.flags.trajectory&&!overlay.raw)"));
const last=source.indexOf("    // Only CPU-processed mode may display verified timeline touch annotations.",first);
assert.ok(first>=0&&last>first,"processed pose rendering is present");
const visible=source.slice(source.indexOf("  const poseVisibleInRally="),source.indexOf("  const drawOverlay="));
const render="(()=>{"+visible+source.slice(first,last)+"})()";
const points=Array.from({length:17},(_,i)=>[0.2+i*0.01,0.4,0.98]);
const events=[];
const canvas={
  beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(x,y,r){events.push(["arc",x,y,r,this.globalAlpha]);},
  fill(){},globalAlpha:1
};
const row={frame:1,in_rally:true,players:[],pose_points:[{keypoints:points}],pose_points_interpolated:false};
const context={frames:[row],frame:1,ctx:canvas,overlay:{flags:{pose:true}},
  point:p=>[p[0]*100,p[1]*100],Number,Array};
runInNewContext(render,context);
assert.equal(events.length,17,"all measured skeleton joints draw without any Track ID or box");
assert.equal(events[0][4],1,"measured joints draw at full opacity");
assert.equal(canvas.globalAlpha,1);
events.length=0;
row.pose_points_interpolated=true;
runInNewContext(render,context);
assert.equal(events.length,17);
assert.ok(events.every(event=>event[4]===0.78),"inferred joints must be visually distinct");
assert.equal(canvas.globalAlpha,1,"drawing restores global alpha");
row.in_rally=false;
events.length=0;
runInNewContext(render,context);
assert.equal(events.length,0,"processed poses stay within rally boundaries");
console.log("Processed RTMPose skeletons render without CPU player boxes: OK");
