import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const js=read("rust/volley-app/static/studio.js");
const css=read("rust/volley-app/static/studio.css");
const draws=[];
const ctx={
  fillStyle:"",setTransform(){},clearRect(){},
  fillRect(x,y,width,height){draws.push({color:this.fillStyle,x,y,width,height});}
};
const canvas={clientWidth:1200,clientHeight:36,width:0,height:0,getContext:()=>ctx};
const nodes=new Map();
const $=id=>{
  if(!nodes.has(id))nodes.set(id,{hidden:false});
  return nodes.get(id);
};
const proposals=Array.from({length:939},(_,i)=>({
  frame:Math.floor(i*164799/939),score:3
}));
const prediction={
  proposals,ground_contacts:[{frame:5555,landing:"near_half"}],rally_count:161
};
const annotation={prediction};
const timeline={canvas,data:{rallies:[]},playhead:{style:{}},track:{setAttribute(){}},
  middle:{textContent:""},end:{textContent:""},status:{textContent:""}};
const context={
  $,timeline,annotation,s:{analysisOffset:0},ui:{video:{currentTime:0}},
  window:{devicePixelRatio:1},training:{active:true},reference:{active:false},
  overlay:{raw:false,flags:{rallies:false,touches:false},actionFlags:{}},
  timelineRallyLayers:{touches:{checked:false},ball:{checked:true},
    candidates:{checked:true},types:{checked:false}},
  comparison:{data:{trajectory_status:"ready"}},
  comparisonMatches:()=>true,annotationSourceValid:()=>true,
  trainingRallyGroups:()=>({touches:[],ball:[{start:100,end:500}]}),
  showBallCandidates:()=>true,showBallTouchTypes:()=>false,
  annotationReady:()=>false,updateTimelineProgress(){},
  timelineDuration:()=>164799/30,fps:()=>30,
  groundContactAppearance:{near_half:["#34d399","near"]},
  cpuTouchStyle:{},cpuTouchKind:kind=>kind
};
const start=js.indexOf("  const drawTimeline=()=>{");
const end=js.indexOf("  const clearTimeline=()=>{",start);
assert.ok(start>=0&&end>start,"Timeline renderer not found");
const drawTimeline=runInNewContext(js.slice(start,end)+"\ndrawTimeline",context);
drawTimeline();
const touchRects=draws.filter(d=>d.color==="#ffbd55");
assert.equal(touchRects.length,939,"Every proposed touch must remain represented on the ruler");
assert.ok(touchRects.every(d=>d.height<=8&&d.height>0&&d.width<=2),
  "Dense touch candidates must be compact markers, not full-height stripes");
const groundRects=draws.filter(d=>d.color==="#34d399");
assert.equal(groundRects.length,1);
assert.ok(groundRects[0].height<=8,"Ground contacts must not fill the timeline height");
assert.match(css,/#studio-preview-view\s+\.video-timeline-track\s+\.studio-trajectory-proposal\{[^}]*height:\s*\d+px/);
assert.doesNotMatch(css,/#studio-preview-view\s+\.video-timeline-track\s+\.studio-trajectory-proposal\{[^}]*top:3px;bottom:3px/);
assert.match(css,/#studio-preview-view\s+\.video-timeline-track\s+\.studio-trajectory-ground\{[^}]*bottom:13px/);
assert.ok(js.includes('marker.className="studio-trajectory-proposal studio-trajectory-ground"'));

const summaryStart=js.indexOf("  const trainingTouchCountSummary=()=>");
const summaryEnd=js.indexOf("  const annotationNewId=()=>",summaryStart);
assert.ok(summaryStart>=0&&summaryEnd>summaryStart,"Visible touch count helper missing");
const summary=runInNewContext(js.slice(summaryStart,summaryEnd)+
  "\ntrainingTouchCountSummary",{annotation,annotationSourceValid:()=>true});
assert.match(summary(),/939/);
assert.match(summary(),/песок: 1/);
assert.ok((js.match(/trainingTouchCountSummary\(\)/g)||[]).length>=2,
  "Count must appear above the video and in the full-video timeline header");
const summaries={cpuSummary:{textContent:""},ballSummary:{textContent:""},
  provenance:{textContent:""}};
const updates={
  timelineRallyLayers:summaries,
  comparison:{data:{trajectory_status:"ready",parameters:{}}},
  comparisonMatches:()=>true,training:{active:true},timeline,annotation,
  annotationSourceValid:()=>true,trainingGapSeconds:()=>4,
  trainingRallyNoun:()=> "розыгрыш",durationTrainingGroups:()=> "26:15",
  trainingTouchCountSummary:summary
};
const summaryBegin=js.indexOf("  const updateTrainingSummaries=groups=>{");
const summaryAfter=js.indexOf("  const showBallCandidates=()=>",summaryBegin);
assert.ok(summaryBegin>=0&&summaryAfter>summaryBegin);
const updateTrainingSummaries=runInNewContext(
  js.slice(summaryBegin,summaryAfter)+"\nupdateTrainingSummaries",updates);
updateTrainingSummaries({ball:Array.from({length:161},()=>({start:200,end:400}))});
assert.match(timeline.status.textContent,/Касания: 939/);
assert.match(timeline.status.textContent,/песок: 1/);
assert.match(summaries.provenance.textContent,/939 кандидатов/);
assert.match(js,/const touchCounts=trainingTouchCountSummary\(\);\s*if\(touchCounts\)comparison\.summary\.textContent\+=" · "\+touchCounts/);
annotation.prediction=null;
assert.equal(summary(),"","Stale results must not display a touch count");
updateTrainingSummaries({ball:[]});
assert.doesNotMatch(timeline.status.textContent,/Касания:/);
console.log("Debug timeline: 939 thin touch markers, separate sand marks and visible source-bound counts");