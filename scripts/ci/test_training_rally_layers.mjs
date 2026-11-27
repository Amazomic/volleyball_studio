import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {previewWasmWindow} from "./helpers/preview_wasm.mjs";
import {runInNewContext} from "node:vm";

const read=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const html=read("rust/volley-app/static/studio.html");
const css=read("rust/volley-app/static/studio.css");
const js=read("rust/volley-app/static/studio.js");
for(const id of ["studio-training-rallies-by-touches","studio-training-rallies-by-trajectory",
  "studio-training-merge-gap-seconds","studio-training-ball-touch-candidates",
  "studio-training-ball-touch-types"]){
  assert.equal((html.match(new RegExp('id="'+id+'"',"g"))||[]).length,1);
}
assert.ok(html.indexOf('class="studio-training-rally-visibility"')>html.indexOf('class="rallies-panel-head"'),
  "Both filter checkboxes must live inside the rally list header");
assert.ok(html.indexOf('class="studio-training-rally-visibility"')<html.indexOf('id="studio-rallies"'));
assert.ok(css.includes("#studio-preview-view.training-workspace .rallies-panel-head .studio-training-rally-visibility{display:grid"));
assert.ok(css.includes("#studio-preview-view.training-workspace .studio-rallies-sort-control{display:none}"));
assert.ok(css.includes('.studio-trajectory-proposal)[hidden]{display:none!important}'),
  "Hidden timeline marker elements must not remain visible");
assert.ok(js.includes('overlay.canvas.addEventListener("contextmenu"'));
assert.match(js,/overlay\.canvas\.addEventListener\("contextmenu",event=>\{[\s\S]{0,240}showAnnotationContext\(annotationFrame\(\),event\);/,
  "Training context actions are restricted to the right-click handler");
assert.match(js,/overlay\.canvas\.addEventListener\("click",event=>\{[\s\S]{0,190}if\(reference\.active\|\|!annotation\.markMode/,
  "Video clicks may open marking only when manual touch mode is enabled");

class Node {
  constructor(){this.style={};this.checked=true;this.hidden=false;this.listeners={};this.textContent="";}
  addEventListener(type,cb){this.listeners[type]=cb;}
  setAttribute(){}
}
const elements=new Map();
const $=id=>{
  if(!elements.has(id))elements.set(id,new Node());
  return elements.get(id);
};
const rects=[];
const ctx={
  fillStyle:"",
  setTransform(){},clearRect(){},
  fillRect(x,y,w,h){rects.push({color:this.fillStyle,x,y,w,h});}
};
const canvas=Object.assign(new Node(),{
  clientWidth:500,clientHeight:38,width:0,height:0,getContext:()=>ctx
});
elements.set("studio-timeline-canvas",canvas);
const training={active:true,panel:{before(){}}};
const reference={active:false};
const overlay={raw:false,flags:{rallies:false,touches:true},actionFlags:{unknown_touch:true}};
const comparison={simulated:false,data:{trajectory_status:"ready",
  trajectory_rallies:[{start:30,end:105},{start:115,end:140}],
  trajectory_impacts:[17],mode:"continuous"}};
const marks={
  ".studio-annotation-tick":[{hidden:false}],
  ".studio-trajectory-proposal":[{hidden:false}]
};
const annotation={
  prediction:{proposals:[{frame:37}]},
  data:{rallies:[{start:16,end:91}],touches:[{frame:48}]},
  lane:{querySelectorAll:selector=>marks[selector]||[]}
};
const s={project:"p",run:"r",video:"match.mp4",videoKey:"p:match.mp4",
  analysisOffset:0,seekAt:0,current:null,gold:new Set(),items:[]};
let renderedLists=0;
const ui={video:{duration:10,currentTime:0},list:{scrollTop:99}};
const start=js.indexOf("  const timeline={");
const end=js.indexOf("  const clearTimeline=()=>",start);
const mergeStart=js.indexOf("  const mergeTrainingIntervals=");
const mergeEnd=js.indexOf("  const comparisonTime=",mergeStart);
assert.ok(start>0&&end>start);
const {timeline,drawTimeline,timelineRallyLayers}=runInNewContext(
  js.slice(mergeStart,mergeEnd)+"\n"+js.slice(start,end)+
    "\n({timeline,drawTimeline,timelineRallyLayers})",{
    $,s,ui,training,overlay,comparison,annotation,reference,
    window:{...previewWasmWindow(),devicePixelRatio:1},requestAnimationFrame:()=>{},
    document:{querySelector:()=>({append(){}})},
    comparisonMatches:()=>true,annotationReady:()=>true,
    annotationSourceValid:()=>true,fps:()=>30,
    cards:()=>{renderedLists++;},renderComparison:()=>{},drawOverlay:()=>{},
    invalidateTrainingTouchPrediction:()=>{},
    cpuTouchKind:kind=>kind,
    cpuTouchStyle:{unknown_touch:["#f14f4f"]}
  }
);
timeline.data={
  project_id:"p",rally_run_id:"r",
  rallies:[{start_frame:0,end_frame:90,candidate_id:"r1"},
    {start_frame:100,end_frame:110,candidate_id:"r2"}],
  touch_annotations:[{frame_index:32,kind:"unknown_touch",touch_attributed:true}]
};
const palette=()=>rects.map(r=>r.color);
timelineRallyLayers.touches.checked=false;
timelineRallyLayers.ball.checked=true;
timelineRallyLayers.candidates.checked=false;
timelineRallyLayers.gap.value="4";
drawTimeline();
assert.ok(!palette().includes("#37b9ce80"),
  "Old CPU touch-based rally intervals must never render in training");
assert.equal(rects.filter(row=>row.color==="#efb15773").length,1,
  "Ball trajectory intervals separated by less than four seconds merge");
comparison.data.trajectory_rallies=[{start:30,end:105},{start:210,end:280}];
comparison.data.parameters={stationary_intervals:[],unobserved_intervals:[[106,209]]};
rects.length=0;
drawTimeline();
assert.equal(rects.filter(row=>row.color==="#efb15773").length,2,
  "Long unobserved pauses must remain separate on the Debug timeline");
comparison.data.trajectory_rallies=[{start:30,end:105},{start:115,end:140}];
comparison.data.parameters=null;
rects.length=0;
drawTimeline();
assert.equal(rects.filter(row=>row.color==="#efb15773").length,1);

rects.length=0;
timelineRallyLayers.gap.value="0";
timelineRallyLayers.gap.listeners.input();
assert.equal(rects.filter(row=>row.color==="#efb15773").length,2,
  "Zero gap restores independent ball trajectory intervals");
rects.length=0;
timelineRallyLayers.gap.value="4";
timelineRallyLayers.gap.listeners.change();
assert.equal(rects.filter(row=>row.color==="#efb15773").length,1);
assert.ok(!palette().includes("#ffbd55"),"Ball touch candidates default hidden");
rects.length=0;
timelineRallyLayers.candidates.checked=true;
timelineRallyLayers.candidates.listeners.change();
assert.equal(timelineRallyLayers.types.disabled,false,
  "Type display requires candidate detection to be enabled");
timelineRallyLayers.types.checked=true;
timelineRallyLayers.types.listeners.change();
assert.match(timelineRallyLayers.typeStatus.textContent,/Типы ещё не рассчитаны/);
assert.ok(palette().includes("#ffbd55"),"Optional touch candidates follow explicit toggle");
annotation.typePrediction={touches:[{frame:37,kind:"attack",track_id:7}],classified:1,total:1};
rects.length=0;
timelineRallyLayers.types.listeners.change();
assert.match(timelineRallyLayers.typeStatus.textContent,/RTMPose/);
assert.ok(palette().includes("#ff6b6b"),"Independent RTMPose classification colors ball candidates");
annotation.typePrediction=null;
rects.length=0;
timelineRallyLayers.types.listeners.change();
assert.ok(!palette().includes("#37b9ce80"),
  "Optional touch candidates must not restore the old rally detector");
assert.ok(!palette().includes("#c6bf96")&&!palette().includes("#f14f4f"),
  "Training timeline must not inherit classified or unclassified CPU touches");
assert.ok(!palette().includes("#10deb2"),
  "Manual marks must remain distinct from trajectory candidates");
rects.length=0;
timelineRallyLayers.touches.checked=true;
timelineRallyLayers.touches.listeners.change();
assert.ok(!palette().includes("#37b9ce80"),
  "Even an outdated saved checkbox cannot reactivate touch-based rallies");
assert.ok(!palette().includes("#f14f4f"),
  "Even a previously enabled CPU layer cannot restore CPU touch ticks");
rects.length=0;
timelineRallyLayers.ball.checked=false;
timelineRallyLayers.ball.listeners.change();
assert.ok(!palette().includes("#efb15773"),
  "Turning off the only rally layer hides its intervals");
assert.ok(!palette().includes("#ffbd55"),
  "Ball candidate markers must obey the trajectory layer");
assert.equal(timelineRallyLayers.types.disabled,true);
assert.equal(marks[".studio-annotation-tick"][0].hidden,false,
  "Manually annotated touches remain independent from trajectory candidates");
assert.equal(marks[".studio-trajectory-proposal"][0].hidden,true);
assert.equal(ui.list.scrollTop,0);
assert.ok(renderedLists>=5,"Changing trajectory settings refreshes the rally list");
reference.active=true;
annotation.data.reviewed=true;
timelineRallyLayers.ball.checked=true;
timelineRallyLayers.touches.checked=false;
rects.length=0;
drawTimeline();
assert.ok(palette().includes("#10deb2"),
  "Verified touch marks remain visible in reference mode even when algorithm layers are disabled");
assert.ok(!palette().includes("#efb15773"),
  "Reference mode does not mix simulated trajectory rallies into the timeline");
reference.active=false;
assert.match(css,/#studio-preview-view\.training-workspace :is\(\.studio-training-source-cpu/);
assert.match(css,/\[hidden\]\{display:none!important\}/);
assert.match(html,/class="studio-training-source studio-training-source-cpu" hidden/);
assert.match(html,/id="studio-training-rallies-by-touches">/);
assert.match(html,/id="studio-training-rallies-by-trajectory" checked hidden/);

const overlayStart=js.indexOf("  const trainingTouchCandidateAtFrame=");
const overlayEnd=js.indexOf("  const drawOverlay=()=>",overlayStart);
assert.ok(overlayStart>0&&overlayEnd>overlayStart);
let detectionVisible=true,typeVisible=false;
const rendered=[];
const paint={
  save(){},restore(){},beginPath(){},arc(){},stroke(){rendered.push("ring");},
  fill(){},fillRect(){},fillText(label){rendered.push(label);},
  measureText:label=>({width:label.length*8})
};
const digest="b".repeat(64);
const overlayAnnotation={prediction:{
  source_replay_sha256:digest,proposals:[{frame:100,score:12}]
}};
const {drawTrainingTouchCandidate}=runInNewContext(
  js.slice(overlayStart,overlayEnd)+"\n({drawTrainingTouchCandidate})",{
    annotation:overlayAnnotation,trainingNames:{attack:"Атака"},annotationSourceValid:()=>true,
    showBallCandidates:()=>detectionVisible,showBallTouchTypes:()=>typeVisible
  });
const detectedBall={replay_sha256:digest,frames:[{frame:100,ball:[0.5,0.4]}]};
drawTrainingTouchCandidate(paint,detectedBall,100,0,0,640,360);
assert.ok(rendered.includes("ring"),"Video must highlight trajectory candidates on the ball");
assert.ok(rendered.some(value=>value.includes("Касание?")));
rendered.length=0;typeVisible=true;
drawTrainingTouchCandidate(paint,detectedBall,100,0,0,640,360);
assert.ok(rendered.some(value=>value.includes("тип не определён")),
  "Unknown type must not be supplied by the CPU postprocessing");
rendered.length=0;
overlayAnnotation.typePrediction={touches:[{frame:100,kind:"attack",track_id:7}]};
drawTrainingTouchCandidate(paint,detectedBall,100,0,0,640,360);
assert.ok(rendered.includes("Атака"),"RTMPose type appears on source-matched video frames");
rendered.length=0;
drawTrainingTouchCandidate(paint,{...detectedBall,replay_sha256:"a".repeat(64)},100,0,0,640,360);
assert.equal(rendered.length,0,"A different replay must never display candidates");
drawTrainingTouchCandidate(paint,detectedBall,109,0,0,640,360);
assert.equal(rendered.length,0,"Candidate marker must not float on unrelated frames");
detectionVisible=false;
drawTrainingTouchCandidate(paint,detectedBall,100,0,0,640,360);
assert.equal(rendered.length,0,"Disabling ball detection must hide its video marker");
console.log("Training rally visibility: single continuous-ball source, merge gap and optional touch candidates: OK");
