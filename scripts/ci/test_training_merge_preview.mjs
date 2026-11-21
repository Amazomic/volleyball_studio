import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const js=read("rust/volley-app/static/studio.js");
const html=read("rust/volley-app/static/studio.html");
assert.match(html,/id="studio-training-merge-gap-seconds"[^>]*value="4"/);
assert.equal((html.match(/id="studio-training-merge-gap-seconds"/g)||[]).length,1);
const start=js.indexOf("  const mergeTrainingIntervals=");
const end=js.indexOf("  const comparisonTime=",start);
assert.ok(start>0&&end>start);
const merge=runInNewContext(js.slice(start,end)+"\nmergeTrainingIntervals");
const input=[
  {start:330,end:389,members:[{id:"cpu3"}]},
  {start:0,end:89,members:[{id:"cpu1"}]},
  {start:150,end:209,members:[{id:"cpu2"}]}
];
const original=JSON.stringify(input);
const result=JSON.parse(JSON.stringify(merge(input,120)));
assert.deepEqual(result,[
  {start:0,end:209,members:[{id:"cpu1"},{id:"cpu2"}]},
  {start:330,end:389,members:[{id:"cpu3"}]}
],"2-second gap merges, exactly 4 seconds does not");
assert.equal(JSON.stringify(input),original,"Source evidence must be immutable");
assert.equal(merge(input,150).length,1,"Increasing threshold merges the next group");
assert.equal(merge(input,0).length,3,"Zero gap disables short-gap merging");
assert.equal(merge([
  {start:0,end:100,members:[{id:1}]},
  {start:100,end:190,members:[{id:2}]}
],0).length,1,"Overlapping evidence is always one interval");
assert.equal(merge([
  {start:0,end:10,members:[{id:1}]},
  {start:-1,end:10,members:[{id:2}]},
  {start:12.5,end:20,members:[{id:3}]}
],0).length,1,"Invalid boundaries are ignored");

const p=js.indexOf("  const trainingGapSeconds=");
const q=js.indexOf("  const updateTrainingRallyLayers=",p);
assert.ok(p>0&&q>p);
const cpuRecords=[
  {start_frame:330,end_frame:389,candidate_id:"3"},
  {start_frame:0,end_frame:89,candidate_id:"1"},
  {start_frame:150,end_frame:209,candidate_id:"2"}
];
const ballRecords=[
  {start:30,end:120},
  {start:180,end:220}
];
const timeline={data:{project_id:"p",rally_run_id:"r",rallies:cpuRecords},status:{textContent:""}};
const s={project:"p",run:"r",items:[cpuRecords[1]],video:"v"};
const comparison={simulated:true,data:{trajectory_status:"ready",mode:"continuous",
  trajectory_rallies:ballRecords}};
const textNode=()=>({textContent:""});
const layers={gap:{value:"4"},cpuSummary:textNode(),ballSummary:textNode(),
  provenance:textNode(),touches:{checked:true},ball:{checked:true},
  candidates:{checked:false}};
const annotation={prediction:null};
const context={timelineRallyLayers:layers,timeline,s,comparison,annotation,
  training:{active:true},mergeTrainingIntervals:merge,fps:()=>30,
  comparisonMatches:data=>Boolean(data),annotationSourceValid:()=>true};
const helpers=runInNewContext(js.slice(p,q)+
  "\n({trainingRallyGroups,trainingGapSeconds,updateTrainingSummaries,showBallCandidates})",
  context);
let groups=JSON.parse(JSON.stringify(helpers.trainingRallyGroups()));
assert.equal(groups.touches.length,0,"Legacy CPU rally groups are unavailable in training");
assert.equal(groups.ball.length,1,"Adjacent ball intervals merge with a four-second gap");
assert.equal(groups.ball[0].start,30);
assert.equal(groups.ball[0].end,220);
helpers.updateTrainingSummaries(groups);
assert.equal(layers.cpuSummary.textContent,"Устаревший способ отключён");
assert.equal(layers.ballSummary.textContent,"1 розыгрыш · 0:06");
assert.match(layers.provenance.textContent,/непрерывная траектория определяет только розыгрыши/);
assert.equal(helpers.showBallCandidates(),false);
layers.candidates.checked=true;
assert.equal(helpers.showBallCandidates(),true);
layers.ball.checked=false;
assert.equal(helpers.showBallCandidates(),false,"Optional touches require the ball-rally layer");
layers.ball.checked=true;
layers.gap.value="0";
groups=JSON.parse(JSON.stringify(helpers.trainingRallyGroups()));
assert.equal(groups.touches.length,0);
assert.equal(groups.ball.length,2,"No merge gap shows both ball trajectory intervals");
layers.gap.value="4";
comparison.data=null;
groups=JSON.parse(JSON.stringify(helpers.trainingRallyGroups()));
helpers.updateTrainingSummaries(groups);
assert.match(layers.ballSummary.textContent,/не рассчитаны/);
const selectStart=js.indexOf("  const rallyAtTimelineTime=");
const selectEnd=js.indexOf("  const schedulePreviewRecovery=",selectStart);
assert.ok(selectStart>0&&selectEnd>selectStart);
comparison.data={trajectory_status:"ready",mode:"continuous",trajectory_rallies:ballRecords};
const {rallyAtTimelineTime}=runInNewContext(js.slice(selectStart,selectEnd)+
  "\n({rallyAtTimelineTime})",{s,training:{active:true},fps:()=>30,
    timeline,trainingRallyGroups:helpers.trainingRallyGroups,timelineRallyLayers:layers});
const insideBallGap=rallyAtTimelineTime(130/30);
assert.equal(insideBallGap.start_frame,30);
assert.equal(insideBallGap.end_frame,220,
  "Full-width timeline must select the merged ball interval");
layers.touches.checked=true;
const stillBall=rallyAtTimelineTime(130/30);
assert.equal(stillBall.start_frame,30,
  "Legacy touch toggle never replaces the trajectory-based source");
layers.ball.checked=false;
assert.equal(rallyAtTimelineTime(130/30),null,
  "Disabling the only training rally layer prevents timeline selection");
console.log("Training-only merging: exact four-second trajectory gap, single source, timeline selection and optional candidates: OK");
