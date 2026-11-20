import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const html=read("rust/volley-app/static/studio.html");
const css=read("rust/volley-app/static/studio.css");
const js=read("rust/volley-app/static/studio.js");
for(const id of ["studio-training-rallies-by-touches","studio-training-rallies-by-trajectory",
  "studio-training-merge-gap-seconds","studio-training-ball-touch-candidates"]){
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
assert.ok(!js.includes('showAnnotationContext(annotationFrame(),event);\n  });\n  const pickTrainingPlayer'));

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
    $,s,ui,training,overlay,comparison,annotation,
    window:{devicePixelRatio:1},requestAnimationFrame:()=>{},
    document:{querySelector:()=>({append(){}})},
    comparisonMatches:()=>true,annotationReady:()=>true,
    annotationSourceValid:()=>true,fps:()=>30,
    cards:()=>{renderedLists++;},renderComparison:()=>{},
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
timelineRallyLayers.candidates.checked=false;
timelineRallyLayers.gap.value="4";
drawTimeline();
assert.ok(palette().includes("#37b9ce80"),"CPU rallies must be translucent cyan");
assert.ok(palette().includes("#efb15773"),"Ball rallies must be translucent amber");
assert.equal(rects.filter(row=>row.color==="#37b9ce80").length,1,
  "Two CPU rallies separated by less than 4 seconds must share one band");
assert.equal(rects.filter(row=>row.color==="#efb15773").length,1,
  "Two trajectory rallies separated by less than 4 seconds must share one band");
assert.ok(!palette().includes("#f8ce87"),
  "Raw direction changes are not automatically attributed to trajectory touch candidates");
rects.length=0;timelineRallyLayers.gap.value="0";
timelineRallyLayers.gap.listeners.input();
assert.equal(rects.filter(row=>row.color==="#37b9ce80").length,2,
  "Zero gap restores distinct CPU intervals");
assert.equal(rects.filter(row=>row.color==="#efb15773").length,2,
  "Zero gap restores distinct ball intervals");
rects.length=0;timelineRallyLayers.gap.value="4";
timelineRallyLayers.gap.listeners.change();
for(const color of ["#00ad9470","#10deb2","#f14f4f"])
  assert.ok(palette().includes(color),"Initial manual/CPU marker missing: "+color);
assert.ok(!palette().includes("#ffbd55"),
  "Unverified ball touch candidates must be hidden by default");
rects.length=0;timelineRallyLayers.candidates.checked=true;
timelineRallyLayers.candidates.listeners.change();
assert.ok(palette().includes("#ffbd55"),"Explicitly enabled candidate markers are shown");
const cpu=rects.find(r=>r.color==="#37b9ce80");
const ball=rects.find(r=>r.color==="#efb15773");
assert.ok(cpu.x<ball.x&&ball.x<cpu.x+cpu.w,"Overlapping rally layers share the same lane");
rects.length=0;timelineRallyLayers.touches.checked=false;
timelineRallyLayers.touches.listeners.change();
assert.equal(renderedLists,4,"Changing the gap and checkboxes immediately refreshes the rally list");
assert.equal(ui.list.scrollTop,0);
assert.ok(!palette().includes("#37b9ce80"));
for(const color of ["#00ad9470","#10deb2","#f14f4f"])
  assert.ok(!palette().includes(color),"Touch-based mark remains after disabling touch rallies: "+color);
assert.equal(marks[".studio-annotation-tick"][0].hidden,true,
  "Manual touch marker must be hidden with touch-rally layer");
assert.equal($("studio-timeline-touch-key").hidden,true,
  "The CPU touch legend must also disappear with its marks");
assert.equal(marks[".studio-trajectory-proposal"][0].hidden,false);
assert.ok(palette().includes("#efb15773"));
rects.length=0;timelineRallyLayers.ball.checked=false;
timelineRallyLayers.ball.listeners.change();
assert.ok(!palette().includes("#37b9ce80")&&!palette().includes("#efb15773"));
assert.ok(!palette().includes("#ffbd55"),"Trajectory proposal markers must obey ball checkbox");
assert.equal(marks[".studio-trajectory-proposal"][0].hidden,true);
rects.length=0;timelineRallyLayers.touches.checked=true;
timelineRallyLayers.touches.listeners.change();
assert.ok(palette().includes("#37b9ce80")&&!palette().includes("#efb15773"));
assert.ok(palette().includes("#00ad9470")&&palette().includes("#f14f4f"));
assert.equal(marks[".studio-annotation-tick"][0].hidden,false);
assert.equal($("studio-timeline-touch-key").hidden,false);
rects.length=0;training.active=false;drawTimeline();
assert.ok(!palette().includes("#37b9ce80"),"Outside training, the existing rally toggle still controls the lane");
console.log("Training rally visibility: both checkboxes hide associated canvas AND DOM marks, preserving non-training behavior: OK");
