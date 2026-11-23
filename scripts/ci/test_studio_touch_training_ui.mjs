import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const html=read("rust/volley-app/static/studio.html");
const js=read("rust/volley-app/static/studio.js");
const rust=read("rust/volley-app/src/touch_annotation/trajectory.rs");
const typesRust=read("rust/volley-app/src/touch_annotation/types.rs");
const server=read("rust/volley-app/src/main.rs");
const annotationSource=read("rust/volley-app/src/touch_annotation.rs");
const scanner=read("rust/volley-app/src/native_player_tracks/trajectory.rs");
const rallyScope=read("rust/volley-app/src/rally_training_comparison.rs");
assert.equal((html.match(/id="studio-timeline-track"/g)||[]).length,1);
assert.ok(html.indexOf('id="studio-impact-settings"') > html.indexOf('class="viewer-head"'));
assert.ok(html.indexOf('id="studio-impact-settings"') < html.indexOf('class="video-wrap"'),
  "Collision sensitivity must be editable alongside the debug player, not beneath the timeline");
for(const id of ["studio-annotation-calculate","studio-annotation-seed-trajectory",
  "studio-annotation-save","studio-training-method","studio-annotation-hide-extra",
  "studio-annotation-review","studio-annotation-rallies",
  "studio-impact-angle","studio-impact-join","studio-impact-fit-error",
  "studio-impact-min-separation"]){
  assert.ok(html.includes('id="'+id+'"'),id);
}
assert.match(server,/\/api\/training\/trajectory\/calculate"[\s\S]{0,90}touch_annotation::trajectory::calculate/);
assert.ok(rust.includes("scan_collision_trajectory(&state, &run, &intervals, req.impact)"));
assert.ok(!rust.includes('method":"vballnet_trajectory_breaks_and_motion_extrema"'));
assert.ok(scanner.includes("full_video_ball_impacts_with_reconstruction"));
assert.ok(scanner.includes("reconstruct_short_ball_motion"));
assert.ok(!rust.includes("touch_rally_scope("),"Touch calculation must not substitute CPU intervals");
assert.match(js,/comparison\.recalculate\.addEventListener\("click",\(\)=>\{\s*invalidateTrainingTouchPrediction\(\)/,
  "Recalculating rallies must invalidate previously computed touches");
assert.ok(scanner.includes("reconstruction.processed_frames += 1"));
assert.ok(scanner.includes("reader.finish() != expected_sha"));
assert.ok(typesRust.includes("scan_collision_trajectory(&state, &run, &intervals, req.impact)"));
assert.match(server,/\/api\/training\/trajectory\/types"[\s\S]{0,100}touch_annotation::types::calculate/);
assert.ok(typesRust.includes("observed_pose_motion")&&typesRust.includes("refine_with_motion"));
assert.ok(typesRust.includes("expected != req.frames"),"Types must never invent a second touch detector");
assert.ok(rust.includes("saved_manual_timing_calibrates_recalculation"));
assert.ok(scanner.includes("reader.finish() != expected_sha")&&
  scanner.includes("verified_ball_trajectory"),
  "Touch training must verify the selected canonical trajectory checksum");
assert.ok(annotationSource.includes('"save" => {'));

class Element {
  constructor(){
    this.textContent="";this.title="";this.value="";this.style={};this.dataset={};
    this.children=[];this.disabled=false;this.listeners={};this.open=false;
    this.className="";this.checked=false;this.classList={
      add(){},remove(){},toggle(){}
    };
  }
  addEventListener(type,cb){this.listeners[type]=cb;}
  click(){if(!this.disabled)return this.listeners.click?.();}
  setAttribute(){}
  replaceChildren(...children){this.children=[...children];this.textContent="";}
  append(...children){this.children.push(...children);}
  querySelectorAll(){return [];}
}
const elements=new Map();
const $=id=>{
  if(!elements.has(id))elements.set(id,new Element());
  return elements.get(id);
};
const logs=[],writeActions=[],calculations=[],classifications=[];
let comparisonRefreshes=0;
for(const [id,value] of [
  ["studio-impact-angle","45"],["studio-impact-join","28"],
  ["studio-impact-fit-error","4"],["studio-impact-min-separation","6"]
])elements.set(id,Object.assign(new Element(),{value}));
const original={
  schema:"volley-touch-annotations/1",project_id:"p",rally_run_id:"r",
  video:"match.mp4",total_frames:200,revision:0,reviewed:false,
  touches:null,rallies:null,metrics:null,overall:{reviewed_videos:0}
};
let uuid=0;
const s={project:"p",run:"r",video:"match.mp4"};
const timeline={data:{
  total_frames:200,touch_frames:[10,20],
  rallies:[{start_frame:8,end_frame:35}]
}};
const ui={video:{currentTime:3.333,addEventListener(){}}};
let drawn=0;
const context={
  $,s,timeline,ui,manualTouchKinds:{attack:"Атака",dig:"Приём снизу",overhead_pass:"Передача сверху"},
  trainingNames:{attack:"Атака",dig:"Приём снизу",overhead_pass:"Передача сверху"},comparison:{epoch:0,simulated:true,
    data:{trajectory_status:"ready",trajectory_rallies:[{start:80,end:180}],
      source_replay_sha256:"a".repeat(64),parameters:{
        min_trajectory_seconds:1.5,stationary_ball_ms:1000,merge_gap_seconds:4}},
    mode:{value:"continuous"},minSeconds:{value:"1.5"},stationaryMs:{value:"1000"},
    recalculate:{disabled:false}},
  comparisonMatches:data=>data?.trajectory_status==="ready",
  trainingGapSeconds:()=>4,
  document:{createElement:()=>new Element()},
  window:{crypto:{randomUUID:()=>`00000000-0000-4000-8000-${String(++uuid).padStart(12,"0")}`},
    addEventListener(){},confirm:()=>true,VolleyballTasks:{
      set:(key,task)=>logs.push({key,task}),clear:key=>logs.push({clear:key})
    }},
  fps:()=>30,
  timelineDuration:()=>200/30,
  drawTimeline:()=>{drawn++;},drawOverlay:()=>{},seekTimeline:()=>{},showAnnotationContext:()=>{},
  refreshComparison:async()=>{comparisonRefreshes++;},
  timelineRallyLayers:{candidates:{checked:false},types:{checked:false}},
  updateTrainingRallyLayers:()=>{},
  training:{active:true},
  request:async()=>({...original}),
  url:(path,query)=>({path,query}),
  post:async(path,payload)=>{
    if(path==="/api/training/trajectory/calculate"){
      calculations.push(payload);
      return {schema:"volley-trajectory-touch-proposals/1",project_id:"p",
        rally_run_id:"r",video:"match.mp4",total_frames:200,
        source_replay_sha256:"a".repeat(64),
        rally_count:1,rally_source:"calculated_ball_trajectory_rallies",rally_intervals:[{end:180,start:80}],
        proposals:[{frame:100,score:9},{frame:140,score:3}],
        serve_proposals:[{kind:"serve",identity_verified:false,
          source:"rally_trajectory_start_v1",rally_start_frame:80,
          trajectory_start_frame:80,provisional_track_id:null}],
        ground_contacts:[{frame:120,position:[0.4,0.7],landing:"near_half"}],
        court_filter_active:true,
        method:"touch_detector_v4",
        ball_motion_interpolation:{stage:"ball_motion_interpolation_v1",
          result:{raw_observations:40,processed_frames:101,rejected_outliers:2,interpolated_frames:9}},
        impact_settings:Object.fromEntries(Object.entries(payload.impact).reverse()),
        calibration:{min_score:0,shift_frames:0,training_videos:0},
        metrics_on_labeled_video:null};
    }
    if(path==="/api/training/trajectory/types"){
      classifications.push(payload);
      return {schema:"volley-trajectory-touch-types/1",project_id:"p",
        video:"match.mp4",rally_run_id:"r",total_frames:200,
        source_replay_sha256:"a".repeat(64),total:payload.frames.length,
        classified:1,touches:payload.frames.map((frame,i)=>({frame,
          kind:i===0?"attack":null,track_id:i===0?7:null,
          score:i===0?0.85:null}))};
    }
    if(path==="/api/training/annotations"){
      writeActions.push(payload);
      assert.equal(payload.action,"save");
      return {...original,revision:1,touches:payload.touches.map(t=>({...t})),
        rallies:payload.rallies.map(t=>({...t})),trajectory_cached:true};
    }
    throw Error("Unexpected route "+path);
  }
};
const start=js.indexOf("  const annotation={");
const end=js.indexOf("  const comparison={",start);
assert.ok(start>0&&end>start);
const api=runInNewContext(js.slice(start,end)+
  "\n({annotation,drawAnnotation,refreshAnnotation,editAnnotation,invalidateTrainingTouchPrediction})",context);
const {annotation,refreshAnnotation,editAnnotation,invalidateTrainingTouchPrediction}=api;

await refreshAnnotation();
assert.equal(comparisonRefreshes,0,"Opening Debug must not start a full-video rally simulation");
assert.equal(annotation.data.revision,0);
assert.equal(annotation.calculate.disabled,false);
assert.equal(annotation.save.disabled,true);
context.comparison.simulated=true;
context.comparison.data={trajectory_status:"ready",trajectory_rallies:[{start:80,end:180}],
  source_replay_sha256:"a".repeat(64),parameters:{
    min_trajectory_seconds:1.5,stationary_ball_ms:1000,merge_gap_seconds:4}};
await elements.get("studio-annotation-calculate").click();
assert.equal(calculations.length,1);
assert.equal(comparisonRefreshes,0,"Touch calculation must not launch full-video rally comparison");
assert.deepEqual(JSON.parse(JSON.stringify(calculations[0].rally_intervals)),
  [{start:80,end:180}],"Touch calculation must use calculated ball rally intervals");
assert.equal(calculations[0].source_replay_sha256,"a".repeat(64));
assert.equal(Object.hasOwn(calculations[0],"rally"),false,
  "Touch calculation must not re-run CPU rally scope");
assert.equal(annotation.prediction.proposals.length,2);
assert.equal(annotation.prediction.ground_contacts.length,1);
assert.equal(annotation.prediction.ground_contacts[0].landing,"near_half");
assert.equal(annotation.prediction.ball_motion_interpolation.result.rejected_outliers,2);
assert.equal(annotation.prediction.ball_motion_interpolation.result.interpolated_frames,9);
assert.equal(annotation.prediction.ball_motion_interpolation.result.processed_frames,101);
assert.match(elements.get("studio-annotation-status").textContent,/обработано 101 кадров в 1 рассчитанных по мячу розыгрышах/);
assert.match(elements.get("studio-annotation-status").textContent,/вне розыгрышей касания не искались/);
assert.deepEqual(JSON.parse(JSON.stringify(calculations[0].impact)),{
  min_turn_angle_deg:45,max_join_error_px:28,
  max_fit_error_px:4,min_separation_frames:6
});
assert.equal(annotation.classify.disabled,false,"Type stage requires ball candidates");
await elements.get("studio-annotation-classify").click();
assert.deepEqual(JSON.parse(JSON.stringify(classifications[0].frames)),[100,140]);
assert.deepEqual(JSON.parse(JSON.stringify(classifications[0].rally_intervals)),
  [{start:80,end:180}],"Types must reuse the same ball rally scope");
assert.deepEqual(JSON.parse(JSON.stringify(classifications[0].impact)),
  JSON.parse(JSON.stringify(calculations[0].impact)));
assert.equal(classifications[0].source_replay_sha256,"a".repeat(64));
assert.equal(annotation.typePrediction.classified,1);
assert.equal(context.timelineRallyLayers.types.checked,true);
assert.equal(writeActions.length,0,"Type prediction does not modify labels");
assert.equal(elements.get("studio-annotation-seed-trajectory").disabled,false);
assert.equal(elements.get("studio-trajectory-proposal"),undefined);
assert.ok(elements.get("studio-timeline-track").children.length>=2);
assert.equal(writeActions.length,0,"Calculation must not persist corrections");

elements.get("studio-annotation-seed-trajectory").click();
assert.equal(annotation.dirty,true);
assert.equal(annotation.data.touches.length,2);
assert.equal(annotation.data.rallies.length,1);
assert.equal(annotation.data.rallies[0].start,80,"Adoption must use the calculated ball rally start");
assert.equal(annotation.data.rallies[0].end,180);
assert.equal(elements.get("studio-annotation-calculate").disabled,true);
assert.equal(writeActions.length,0,"Adopting candidates is a local edit");
const touched=annotation.data.touches[0];
await editAnnotation("move_touch",{id:touched.id,frame:102});
await editAnnotation("update_touch",{id:touched.id,kind:"dig",position:[0.3,0.6]});
assert.equal(annotation.data.touches[0].frame,102);
assert.equal(annotation.data.touches[0].kind,"dig");
assert.deepEqual(JSON.parse(JSON.stringify(annotation.data.touches[0].position)),[0.3,0.6]);
assert.equal(writeActions.length,0,"Local touch edits must not contact server");
assert.equal(elements.get("studio-annotation-save").disabled,false);

await elements.get("studio-annotation-save").click();
assert.equal(writeActions.length,1);
assert.equal(writeActions[0].revision,0);
assert.equal(writeActions[0].touches[0].frame,102);
assert.equal(writeActions[0].touches[0].kind,"dig");
assert.deepEqual(JSON.parse(JSON.stringify(writeActions[0].touches[0].position)),[0.3,0.6]);
assert.equal(annotation.dirty,false);
assert.equal(annotation.data.revision,1);
assert.equal(annotation.prediction,null,"Saved labels require a fresh calculation");
assert.equal(annotation.typePrediction,null,"Saved labels must invalidate classified proposals");
assert.equal(elements.get("studio-annotation-calculate").disabled,false);
context.comparison.simulated=true;
context.comparison.data={trajectory_status:"ready",trajectory_rallies:[{start:80,end:180}],
  source_replay_sha256:"a".repeat(64),parameters:{
    min_trajectory_seconds:1.5,stationary_ball_ms:1000,merge_gap_seconds:4}};
await elements.get("studio-annotation-calculate").click();
assert.equal(calculations.length,2);
assert.equal(comparisonRefreshes,0,"Repeated touch calculation must stay within saved rallies");
assert.equal(annotation.prediction.calibration.shift_frames,0);
assert.equal(annotation.prediction.calibration.training_videos,0);
assert.equal(annotation.prediction.proposals[0].frame,100);
elements.get("studio-impact-angle").value="75";
elements.get("studio-impact-angle").listeners.change();
assert.equal(annotation.prediction,null,"Changing direction sensitivity invalidates stale candidates");
await elements.get("studio-annotation-calculate").click();
assert.equal(calculations[2].impact.min_turn_angle_deg,75);
invalidateTrainingTouchPrediction();
assert.equal(annotation.prediction,null,"Changing rally settings must invalidate old touch candidates");
assert.ok(drawn>=4);
context.comparison.data=null;
const beforeMissingRallies=calculations.length;
await elements.get("studio-annotation-calculate").click();
assert.equal(calculations.length,beforeMissingRallies,
  "Without a calculated rally result the touch API must not be called");
assert.match(elements.get("studio-annotation-status").textContent,/Сначала нажмите «Рассчитать розыгрыши»/);
context.comparison.data={trajectory_status:"ready",trajectory_rallies:[{start:80,end:180}],
  source_replay_sha256:"a".repeat(64),parameters:{
    min_trajectory_seconds:1.5,stationary_ball_ms:1000,merge_gap_seconds:4}};
context.comparison.recalculate.disabled=true;
await elements.get("studio-annotation-calculate").click();
assert.equal(calculations.length,beforeMissingRallies,
  "Touches must not run while a newer rally calculation is in flight");
assert.match(elements.get("studio-annotation-status").textContent,/Расчёт розыгрышей ещё выполняется/);
assert.ok(logs.some(entry=>entry.key==="training-trajectory"&&
  entry.task.stage==="Touch Detector · розыгрыши по мячу"));
assert.ok(logs.some(entry=>entry.key==="training-annotation"));
console.log("PASS: ballistic collision parameters -> calculate -> classify -> edit -> save -> invalidate and recalculate");
