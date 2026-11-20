import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const html=read("rust/volley-app/static/studio.html");
const css=read("rust/volley-app/static/studio.css");
const js=read("rust/volley-app/static/studio.js");
for(const id of ["studio-training-rallies-by-touches","studio-training-rallies-by-trajectory"]){
  assert.equal((html.match(new RegExp('id="'+id+'"',"g"))||[]).length,1);
}
assert.ok(css.includes("#studio-preview-view.training-workspace .studio-training-rally-visibility{display:grid}"));
assert.ok(css.includes("left:69%"),"Checkboxes must be inside the upper right of the video area");
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
const overlay={raw:false,flags:{rallies:false,touches:false}};
const comparison={simulated:false,data:{trajectory_status:"ready",
  trajectory_rallies:[{start:30,end:105}],trajectory_impacts:[]}};
const annotation={prediction:null};
const s={project:"p",video:"match.mp4",videoKey:"p:match.mp4",
  analysisOffset:0,seekAt:0,current:null,gold:new Set()};
const ui={video:{duration:10,currentTime:0}};
const start=js.indexOf("  const timeline={");
const end=js.indexOf("  const clearTimeline=()=>",start);
assert.ok(start>0&&end>start);
const {timeline,drawTimeline,timelineRallyLayers}=runInNewContext(
  js.slice(start,end)+"\n({timeline,drawTimeline,timelineRallyLayers})",{
    $,s,ui,training,overlay,comparison,annotation,
    window:{devicePixelRatio:1},requestAnimationFrame:()=>{},
    document:{querySelector:()=>({append(){}})},
    comparisonMatches:()=>true,annotationReady:()=>false,
    annotationSourceValid:()=>false,fps:()=>30
  }
);
timeline.data={rallies:[{start_frame:0,end_frame:90,candidate_id:"r1"}],touch_annotations:[]};
const palette=()=>rects.map(r=>r.color);
drawTimeline();
assert.ok(palette().includes("#37b9ce80"),"CPU rallies must be translucent cyan");
assert.ok(palette().includes("#efb15773"),"Ball rallies must be translucent amber");
const cpu=rects.find(r=>r.color==="#37b9ce80");
const ball=rects.find(r=>r.color==="#efb15773");
assert.ok(cpu.x<ball.x&&ball.x<cpu.x+cpu.w,"Overlapping rally layers share the same lane");
rects.length=0;timelineRallyLayers.touches.checked=false;
timelineRallyLayers.touches.listeners.change();
assert.ok(!palette().includes("#37b9ce80"));
assert.ok(palette().includes("#efb15773"));
rects.length=0;timelineRallyLayers.ball.checked=false;
timelineRallyLayers.ball.listeners.change();
assert.ok(!palette().includes("#37b9ce80")&&!palette().includes("#efb15773"));
rects.length=0;timelineRallyLayers.touches.checked=true;
timelineRallyLayers.touches.listeners.change();
assert.ok(palette().includes("#37b9ce80")&&!palette().includes("#efb15773"));
rects.length=0;training.active=false;drawTimeline();
assert.ok(!palette().includes("#37b9ce80"),"Outside training, the existing rally toggle still controls the lane");
console.log("Training rally visibility: two independent translucent layers, overlap and non-training fallback: OK");
