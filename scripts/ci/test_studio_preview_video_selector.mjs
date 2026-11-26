import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
assert.match(html,/<div class="viewer-head">\s*<strong>[\s\S]*?<select id="studio-preview-video"/);
const start=js.indexOf("  const newestPrimaryAnalysis=");
const end=js.indexOf("  const renderChain=",start);
const a="a.mp4",b="b.mp4",c="c.mp4";
const choices=[{video:a,value:"a1",created_at:100},{video:b,value:"b1",created_at:200},{video:b,value:"b2",created_at:300}];
const s={projectVideos:[a,b,c],chain:{choices},video:a,analysis:"a1"};
const ui={analysis:{value:"a1"},previewVideo:{value:b,disabled:false}};
const calls=[];
const context={s,ui,refreshPicker(){},setAnalysis:async(...args)=>calls.push([ui.analysis.value,...args])};
runInNewContext(js.slice(start,end)+"\nthis.selectPreviewVideo=selectPreviewVideo;",context);
await context.selectPreviewVideo();
assert.equal(ui.analysis.value,"b2");
assert.equal(calls.length,1);
ui.previewVideo.value=c;
await context.selectPreviewVideo();
assert.deepEqual(calls[1],["","",true,c]);
ui.previewVideo.value="foreign.mp4";
await context.selectPreviewVideo();
assert.equal(calls.length,2);
console.log("PASS: preview video selector isolates source and selects matching analysis");

const chain={choices,byAnalysis:new Map([
  ["a1",[{video:a,rally_run_id:"run-a"}]],
  ["b1",[{video:b,rally_run_id:"run-b-old"}]],
  ["b2",[{video:b,rally_run_id:"run-b"}]]
])};
const state={project:"project",projectVideos:[a,b,c],chain,video:a,analysis:"a1",run:"run-a",items:["old"]};
const controls={analysis:{value:"a1"},run:{value:"run-a"},video:{currentTime:0},previewVideo:{value:b}};
const loaded=[],saved=[];
const runtime={s:state,ui:controls,overlay:{raw:false},
  reset(){state.video="";state.run="";state.items=[];},
  options(target,rows,key,label){target.options=rows.map(row=>({value:row[key],label:label(row)}));},
  refreshPicker(){},ensurePreviewVideo(){loaded.push(["media",state.video]);},
  setOverlaySource(){},updateRawDetectorLabel(){},prefetchOverlayAtFrame(){},fps:()=>30,rawFusion:()=>null,
  window:{VolleyStudioChain:{stamp:()=>"",latestRun:(rows,preferred)=>rows.find(row=>row.rally_run_id===preferred)?.rally_run_id||rows[0]?.rally_run_id||""},dispatchEvent(){}},
  CustomEvent:class{},sourceSelection:{save:(...args)=>saved.push(args)},
  loadPage:async()=>loaded.push(["cards",state.run]),loadTimeline:async()=>loaded.push(["timeline",state.run]),
  message(){},failed:e=>{throw e;}
};
const begin=js.indexOf("  const setRuns=async(");
const finish=js.indexOf("  const pipelineActive=",begin);
runInNewContext(js.slice(begin,finish)+"\nthis.selectPreviewVideo=selectPreviewVideo;this.renderChain=renderChain;",runtime);
await runtime.selectPreviewVideo();
assert.equal(state.video,b);
assert.equal(state.run,"run-b");
assert.equal(controls.previewVideo.value,b);
assert.equal(state.items.length,0);
assert.deepEqual(loaded,[["media",b],["cards","run-b"],["timeline","run-b"]]);
controls.previewVideo.value=c;
await runtime.selectPreviewVideo();
assert.equal(state.video,c);
assert.equal(state.run,"");
assert.deepEqual(loaded.at(-1),["media",c]);
assert.deepEqual(saved.at(-1),["project","","",c]);
await runtime.renderChain(chain);
assert.equal(state.video,c,"Metadata refresh must preserve an explicitly selected video without analysis");
controls.previewVideo.value=a;
await runtime.selectPreviewVideo();
assert.equal(state.video,a);
assert.equal(state.run,"run-a");
assert.deepEqual(loaded.slice(-3),[["media",a],["cards","run-a"],["timeline","run-a"]]);
console.log("PASS: actual source transition resets old cards and loads matching media, CPU list and timeline");
