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
  "studio-comparison-duration","studio-comparison-gap","studio-comparison-filter",
  "studio-comparison-calibrate","studio-comparison-mode",
  "studio-comparison-min-seconds","studio-comparison-impact-settings",
  "studio-comparison-continuous-settings"]){
  assert.ok(html.includes('id="'+id+'"'),"Missing comparison control: "+id);
}
assert.equal((html.match(/id="studio-timeline-track"/g)||[]).length,1);
assert.ok(html.includes('id="studio-training-pose-advanced"'));
assert.ok(js.includes('training.poseAdvanced.open'));
assert.ok(js.includes('request(url("/api/training/rally-comparison",options))'));
assert.ok(!js.includes('void refreshComparison();'),"Opening Debug must not recalculate full-video rallies");
assert.ok(js.includes('void refreshComparison(true);'));
assert.ok(js.includes('void refreshComparison(true,true)'));
assert.ok(rust.includes('fn fit_trajectory_settings('));
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
const sidebar=[];

const s={project:"p",run:"r",video:"source.mp4"};
const context={
  $,s,document:{createElement:()=>new Node()},
  window:{VolleyballTasks:{
    set:(key,details)=>sidebar.push({type:"set",key,details}),
    clear:key=>sidebar.push({type:"clear",key})
  }},
  ui:{video:{pause:()=>{paused=true;}}},
  fps:()=>30,
  seekTimeline:seconds=>{seeked=seconds;},
  training:{active:true},
  drawTimeline:()=>{},cards:()=>{},
  invalidateTrainingTouchPrediction:()=>{},
  timelineRallyLayers:{candidates:{checked:false}},
  url:(path,query)=>({path,query}),
  request:async()=>{throw Error("Expected mocked API response");}
};
const finish=js.indexOf("  comparison.recalculate.addEventListener",end);
const {comparison,renderComparison,refreshComparison}=runInNewContext("const trainingGapSeconds=()=>4;\n"+js.slice(start,finish)+
  "\n({comparison,renderComparison,refreshComparison})",context);
comparison.mode.value="continuous";
comparison.filter.value="all";
comparison.minSeconds.value="1.5";
comparison.stationaryMs.value="1000";
const ready={
  schema:"volley-rally-training-comparison/1",
  project_id:"p",rally_run_id:"r",video:"source.mp4",trajectory_status:"ready",
  mode:"continuous",
  cpu_rallies:[{start:10,end:30}],
  trajectory_rallies:[{start:30,end:180}],
  trajectory_impacts:[],
  reference_rallies:null,reference_reviewed:false,
  metrics:{cpu_trajectory:{matched:0,missed:1,extra:1,f1:0},
    cpu_reference:null,trajectory_reference:null},
  differences:[],
  simulated_touch_candidates:[],unassigned_touch_candidates:[]
};
comparison.data=ready;
comparison.simulated=true;
renderComparison();
assert.equal(elements.get("studio-comparison-metrics").children.length,2,
  "Only trajectory result and manual reference status are presented");
assert.ok(elements.get("studio-comparison-metrics").children.every(
  metric=>!metric.children[0]?.textContent.includes("CPU")));
assert.match(elements.get("studio-simulation-summary").textContent,/Устойчивая траектория/);
assert.equal(comparison.calibrate.disabled,true,"Legacy impact calibration must remain disabled");
s.run="another";
renderComparison();
assert.equal(comparison.recalculate.disabled,false,"A selected run must allow manual rally calculation even without cached comparison");
assert.equal(comparison.minSeconds.disabled,false,"Rally parameters must remain editable before manual calculation");
s.run="";
renderComparison();
assert.equal(comparison.recalculate.disabled,true,"Without a selected run the simulation is disabled");
s.run="another";
assert.match(elements.get("studio-comparison-differences").textContent,/Выберите CPU-анализ/);
let release;
context.request=()=>new Promise(resolve=>{release=resolve;});
const pending=refreshComparison(true);
s.run="newer";
release(ready);
await pending;
assert.equal(comparison.sourceKey,"","Stale request must not claim a different run");
s.run="r";
const queries=[];
context.request=async query=>{
  queries.push(query.query);
  return {...ready,parameters:{min_trajectory_seconds:3.5,stationary_ball_ms:1300,stationary_intervals:[]}};
};
comparison.minSeconds.value="3.5";
comparison.stationaryMs.value="1300";
await refreshComparison(true);
assert.equal(queries.at(-1).mode,"continuous");
assert.equal(queries.at(-1).min_trajectory_seconds,3.5);
assert.equal(queries.at(-1).stationary_ball_ms,1300);
assert.equal(queries.at(-1).merge_gap_seconds,4);
for(const obsolete of ["min_impacts","context_frames","min_duration_frames","max_touch_gap_frames"])
  assert.equal(Object.hasOwn(queries.at(-1),obsolete),false,
    "Legacy impact detection setting must not reach server: "+obsolete);
assert.equal(comparison.simulated,true);
assert.match(elements.get("studio-simulation-summary").textContent,/Устойчивая траектория/);
assert.equal(comparison.calibrate.disabled,true);
assert.equal(comparison.impactSettings.hidden,true);
assert.equal(comparison.continuousSettings.hidden,false);
assert.equal(comparison.recalculate.textContent,"Рассчитать розыгрыши");
comparison.data=null;
renderComparison();
assert.equal(comparison.recalculate.disabled,false,"Manual rally button must stay available after clearing results");
assert.equal(comparison.minSeconds.disabled,false);
assert.equal(comparison.stationaryMs.disabled,false);
assert.equal(comparison.minImpacts.disabled,true);
assert.equal(comparison.minSeconds.value,3.5);
assert.equal(comparison.stationaryMs.value,1300);
assert.match(html,/id="studio-comparison-mode"><option value="continuous" selected>/);
assert.doesNotMatch(html,/<option value="impacts">/);
assert.match(html,/class="studio-training-compare-review" hidden/);
assert.match(html,/id="studio-comparison-min-seconds"[^>]*value="1\.5"/);
assert.match(rust,/Only continuous-ball rally detection is supported/);
const calculateStart=js.indexOf("  const calculateTrajectory=async()=>");
const calculateEnd=js.indexOf("  const calculateTouchTypes=async()=>",calculateStart);
assert.ok(calculateStart>0&&calculateEnd>calculateStart);
assert.ok(!js.slice(calculateStart,calculateEnd).includes("refreshComparison"),
  "Touch calculation must not trigger any separate full-video comparison");
{
  const first=js.indexOf("  const annotationTrajectoryRallySeeds=()=>");
  const last=js.indexOf("  const annotationSavePresentation=",first);
  assert.ok(first>0&&last>first);
  let nextId=0;
  const edited={data:{total_frames:300,touches:[],rallies:[]},
    prediction:{total_frames:300,proposals:[{frame:42}],rally_source:"calculated_ball_trajectory_rallies",
      rally_intervals:[{start:30,end:180}]}};
  const proposed={data:{schema:"volley-rally-training-comparison/1",
    trajectory_status:"ready",trajectory_rallies:[{start:15,end:260}]}};
  const {annotationFromPrediction}=runInNewContext(js.slice(first,last)+
    "\n({annotationFromPrediction})",{
    annotation:edited,comparison:proposed,timeline:{data:{rallies:[]}},
    comparisonMatches:data=>data===proposed.data,
    annotationNewId:()=>String(++nextId)
  });
  const prediction=annotationFromPrediction();
  assert.equal(prediction.touches.length,1);
  assert.equal(prediction.rallies.length,1,
    "A zero-rally CPU run must seed its draft from the calculated ball intervals");
  assert.equal(prediction.rallies[0].start,30);
  edited.data.rallies=[{id:"manual",start:40,end:170}];
  assert.equal(annotationFromPrediction().rallies[0].id,"manual",
    "Existing manually labeled rallies must not be overwritten");
}
console.log("Training comparison: single continuous mode, stale-source guard, trajectory-only request and editing: OK");
