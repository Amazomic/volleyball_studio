import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const begin=js.indexOf("  let overlaySeekTimer=null;");
const end=js.indexOf("  const cpuTouchStyle=",begin);
assert.ok(begin>0&&end>begin,"Seek prefetch coalescing must exist");

const requested=[],timers=new Map();
let serial=0;
const state={project:"project",video:"match.mp4",run:"cpu-1"};
const overlay={raw:false};
const timeline={data:{project_id:"project",rally_run_id:"cpu-1",total_frames:50000}};
const context={
  s:state,overlay,timeline,
  overlaySourceId:()=>state.run,
  rawFrameLimit:()=>50000,
  overlayWindowStart:frame=>Math.max(0,Math.floor(frame/90)*90-60),
  requestOverlayWindow:(...args)=>requested.push(args),
  setTimeout:callback=>{const id=++serial;timers.set(id,callback);return id;},
  clearTimeout:id=>timers.delete(id),
  Number,Math
};
runInNewContext(js.slice(begin,end)+"\nthis.prefetch=prefetchOverlayAtFrame;",context);
context.prefetch(90);
context.prefetch(1230);
context.prefetch(4200);
assert.equal(timers.size,1,"Fast rally changes should schedule one window");
for(const [id,work] of timers){timers.delete(id);work();}
assert.equal(requested.length,1,"Only the last selected rally should be fetched");
assert.equal(requested[0][3],4080,"The selected frame window should be prioritized");
context.prefetch(5300);
state.run="cpu-2";
for(const work of timers.values())work();
assert.equal(requested.length,1,"Old analysis selection must not fetch stale windows");
console.log("Rally seek prefetch coalescing: OK");
