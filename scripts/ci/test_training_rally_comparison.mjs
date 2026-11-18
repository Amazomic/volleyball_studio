import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const html=read("rust/volley-app/static/studio.html");
const js=read("rust/volley-app/static/studio.js");
const rust=read("rust/volley-app/src/rally_training_comparison.rs");
const annotation=read("rust/volley-app/src/touch_annotation.rs");
const main=read("rust/volley-app/src/main.rs");

for(const id of ["studio-comparison-recalculate","studio-comparison-status",
  "studio-comparison-metrics","studio-comparison-differences",
  "studio-comparison-min-impacts","studio-comparison-context",
  "studio-comparison-duration","studio-comparison-gap","studio-comparison-filter"]){
  assert.ok(html.includes('id="'+id+'"'),"Missing comparison control: "+id);
}
assert.equal((html.match(/id="studio-timeline-track"/g)||[]).length,1);
assert.ok(html.includes('id="studio-training-pose-advanced"'));
assert.ok(js.includes('training.poseAdvanced.open'));
assert.ok(js.includes('request(url("/api/training/rally-comparison",options))'));
assert.ok(js.includes('void refreshComparison();'));
assert.ok(js.includes('void refreshComparison(true);'));
assert.ok(js.includes('comparison.epoch++;comparison.data=null;comparison.sourceKey=""'));
assert.ok(js.includes('comparison.data.trajectory_rallies'));
assert.match(main,/\/api\/training\/rally-comparison"[\s\S]{0,90}get\(rally_training_comparison::get\)/);
assert.ok(rust.includes('preliminary_rally_intervals(&impacts, run.frames, settings)'));
assert.ok(rust.includes('Sha256::digest(&bytes)'));
assert.ok(rust.includes('source_replay_sha256'));
assert.ok(rust.includes('touch_annotation::comparison_truth(state, run)'));
assert.ok(rust.includes('published_cpu_unchanged'));
assert.ok(annotation.includes('pub(crate) fn comparison_truth('));

class Node {
  constructor(){this.children=[];this.textContent="";this.listeners={};this.disabled=false;this.value="";}
  replaceChildren(...items){this.children=[...items];this.textContent="";}
  append(...items){this.children.push(...items);}
  addEventListener(key,fn){this.listeners[key]=fn;}
  click(){this.listeners.click?.();}
}
const elements=new Map();
const $=id=>{
  if(!elements.has(id))elements.set(id,new Node());
  return elements.get(id);
};
const start=js.indexOf("  const comparison={");
const end=js.indexOf("  const refreshComparison=async",start);
assert.ok(start>0&&end>start);
let seeked=null,paused=false;
const s={project:"p",run:"r",video:"source.mp4"};
const context={
  $,s,document:{createElement:()=>new Node()},
  ui:{video:{pause:()=>{paused=true;}}},
  fps:()=>30,
  seekTimeline:seconds=>{seeked=seconds;},
  training:{active:true},
  drawTimeline:()=>{},
  url:(path,query)=>({path,query}),
  request:async()=>{throw Error("Expected mocked API response");}
};
const finish=js.indexOf("  comparison.recalculate.addEventListener",end);
const {comparison,renderComparison,refreshComparison}=runInNewContext(js.slice(start,finish)+
  "\n({comparison,renderComparison,refreshComparison})",context);
comparison.filter.value="disagreement";
comparison.data={
  schema:"volley-rally-training-comparison/1",
  project_id:"p",rally_run_id:"r",video:"source.mp4",trajectory_status:"ready",
  cpu_rallies:[{start:10,end:30},{start:90,end:150}],
  trajectory_rallies:[{start:12,end:31},{start:220,end:260}],
  trajectory_impacts:[12,220],
  reference_rallies:[],
  reference_reviewed:false,
  metrics:{cpu_trajectory:{matched:1,missed:1,extra:1,f1:0.5},
    cpu_reference:null,trajectory_reference:null},
  differences:[
    {kind:"agreed",cpu:{start:10,end:30},trajectory:{start:12,end:31},iou:0.85,seek_frame:10},
    {kind:"cpu_only",cpu:{start:90,end:150},trajectory:null,iou:null,seek_frame:90},
    {kind:"trajectory_only",cpu:null,trajectory:{start:220,end:260},iou:null,seek_frame:220},
    {kind:"both_missed",cpu:null,trajectory:null,reference:{start:340,end:380},iou:null,seek_frame:340}
  ]
};
renderComparison();
const rows=elements.get("studio-comparison-differences").children.filter(
  row=>row.className?.includes("studio-training-compare-row"));
assert.equal(rows.length,3,"disagreements should exclude agreed rallies");
rows[1].click();
assert.equal(seeked,220/30);
assert.equal(paused,true);
assert.equal(elements.get("studio-comparison-metrics").children.length,7);
comparison.filter.value="all";
renderComparison();
assert.equal(elements.get("studio-comparison-differences").children.filter(
  row=>row.className?.includes("studio-training-compare-row")).length,4);
s.run="another";
renderComparison();
assert.match(elements.get("studio-comparison-differences").textContent,/Выберите CPU-анализ/);
assert.equal(comparison.recalculate.disabled,true);
let release;
context.request=()=>new Promise(resolve=>{release=resolve;});
const pending=refreshComparison();
s.run="newer";
release(comparison.data);
await pending;
assert.equal(comparison.sourceKey,"","An old request must not claim the newer run");
assert.equal(comparison.epoch,1);
const saved=comparison.data;
s.run="r";
comparison.data={...saved,trajectory_status:"unavailable"};
renderComparison();
assert.equal(comparison.recalculate.disabled,false,
  "Older CPU versions must offer simulation from their saved replay");
const queries=[];
context.request=async (_request)=> {
  queries.push(_request.query);
  return {...saved,trajectory_status:"ready",
    simulated_touch_candidates:[12,220],
    parameters:{min_impacts:2,context_frames:45,min_duration_frames:45,max_touch_gap_frames:120}};
};
comparison.sourceKey="p:r:source.mp4";
await refreshComparison(true);
assert.equal(queries.at(-1).simulate,true);
assert.equal(comparison.simulated,true);
assert.equal(elements.get("studio-comparison-touches").children.length,2);
assert.match(elements.get("studio-simulation-summary").textContent,/возможных касаний/);
await refreshComparison();
assert.equal(queries.at(-1).simulate,true,
  "Manual edits must preserve the current simulation and its parameters");
console.log("Training comparison: source guard, editable params, separate timeline overlays and discrepancy navigation: OK");
