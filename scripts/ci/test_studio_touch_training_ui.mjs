import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const html=read("rust/volley-app/static/studio.html");
const js=read("rust/volley-app/static/studio.js");
const rust=read("rust/volley-app/src/touch_annotation/trajectory.rs");
const server=read("rust/volley-app/src/main.rs");
const annotationSource=read("rust/volley-app/src/touch_annotation.rs");
const scanner=read("rust/volley-app/src/native_player_tracks/trajectory.rs");
assert.equal((html.match(/id="studio-timeline-track"/g)||[]).length,1);
for(const id of ["studio-annotation-calculate","studio-annotation-seed-trajectory",
  "studio-annotation-save","studio-training-method","studio-annotation-hide-extra",
  "studio-annotation-review","studio-annotation-rallies"]){
  assert.ok(html.includes('id="'+id+'"'),id);
}
assert.match(server,/\/api\/training\/trajectory\/calculate"[\s\S]{0,90}touch_annotation::trajectory::calculate/);
assert.ok(rust.includes("trajectory_profile(&lib)"));
assert.ok(rust.includes("saved_manual_timing_calibrates_recalculation"));
assert.ok(scanner.includes("reader.finish() != run.replay_sha"));
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
const logs=[],writeActions=[],calculations=[];
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
  $,s,timeline,ui,document:{createElement:()=>new Element()},
  window:{crypto:{randomUUID:()=>`00000000-0000-4000-8000-${String(++uuid).padStart(12,"0")}`},
    addEventListener(){},confirm:()=>true,VolleyballTasks:{
      set:(key,task)=>logs.push({key,task}),clear:key=>logs.push({clear:key})
    }},
  fps:()=>30,
  timelineDuration:()=>200/30,
  drawTimeline:()=>{drawn++;},seekTimeline:()=>{},showAnnotationContext:()=>{},
  refreshComparison:async()=>{},
  training:{active:true},
  request:async()=>({...original}),
  url:(path,query)=>({path,query}),
  post:async(path,payload)=>{
    if(path==="/api/training/trajectory/calculate"){
      calculations.push(payload);
      return {schema:"volley-trajectory-touch-proposals/1",project_id:"p",
        rally_run_id:"r",video:"match.mp4",total_frames:200,
        source_replay_sha256:"a".repeat(64),
        proposals:calculations.length===1?
          [{frame:100,score:9},{frame:140,score:3}]:
          [{frame:102,score:9}],
        calibration:{min_score:8,shift_frames:calculations.length===1?0:2,
          training_videos:calculations.length===1?0:1},
        metrics_on_labeled_video:null};
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
  "\n({annotation,drawAnnotation,refreshAnnotation,editAnnotation})",context);
const {annotation,refreshAnnotation,editAnnotation}=api;

await refreshAnnotation();
assert.equal(annotation.data.revision,0);
assert.equal(annotation.calculate.disabled,false);
assert.equal(annotation.save.disabled,true);
await elements.get("studio-annotation-calculate").click();
assert.equal(calculations.length,1);
assert.equal(annotation.prediction.proposals.length,2);
assert.equal(elements.get("studio-annotation-seed-trajectory").disabled,false);
assert.equal(elements.get("studio-trajectory-proposal"),undefined);
assert.ok(elements.get("studio-timeline-track").children.length>=2);
assert.equal(writeActions.length,0,"Calculation must not persist corrections");

elements.get("studio-annotation-seed-trajectory").click();
assert.equal(annotation.dirty,true);
assert.equal(annotation.data.touches.length,2);
assert.equal(annotation.data.rallies.length,1);
assert.equal(elements.get("studio-annotation-calculate").disabled,true);
assert.equal(writeActions.length,0,"Adopting candidates is a local edit");
const touched=annotation.data.touches[0];
await editAnnotation("move_touch",{id:touched.id,frame:102});
assert.equal(annotation.data.touches[0].frame,102);
assert.equal(writeActions.length,0,"Local touch edits must not contact server");
assert.equal(elements.get("studio-annotation-save").disabled,false);

await elements.get("studio-annotation-save").click();
assert.equal(writeActions.length,1);
assert.equal(writeActions[0].revision,0);
assert.equal(writeActions[0].touches[0].frame,102);
assert.equal(annotation.dirty,false);
assert.equal(annotation.data.revision,1);
assert.equal(annotation.prediction,null,"Saved labels require a fresh calculation");
assert.equal(elements.get("studio-annotation-calculate").disabled,false);
await elements.get("studio-annotation-calculate").click();
assert.equal(calculations.length,2);
assert.equal(annotation.prediction.calibration.shift_frames,2);
assert.equal(annotation.prediction.calibration.training_videos,1);
assert.equal(annotation.prediction.proposals[0].frame,102);
assert.ok(drawn>=4);
assert.ok(logs.some(entry=>entry.key==="training-trajectory"));
assert.ok(logs.some(entry=>entry.key==="training-annotation"));
console.log("PASS: trajectory calculate -> accept -> edit -> explicit save -> recalibrate using saved labels");
