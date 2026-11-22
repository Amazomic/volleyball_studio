import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {join} from "node:path";
import {fileURLToPath} from "node:url";
import vm from "node:vm";

const root=fileURLToPath(new URL("../..",import.meta.url));
const js=readFileSync(join(root,"rust/volley-app/static/studio_tabs.js"),"utf8");
const rust=readFileSync(join(root,"rust/volley-app/src/studio_pipeline.rs"),"utf8");
const html=readFileSync(join(root,"rust/volley-app/static/studio.html"),"utf8");
const css=readFileSync(join(root,"rust/volley-app/static/studio.css"),"utf8");
assert.match(css,/\.analysis-flow-diagram\{[^}]*max-width:none;width:100%/);
assert.match(css,/\.analysis-flow-grid\{[^}]*grid-template-columns:repeat\(4,minmax\(0,1fr\)\)/);
assert.match(css,/\.analysis-flow-node\{[^}]*width:100%;height:84px/);
assert.match(css,/\.analysis-flow-node-stats\{[^}]*font-size:/);
assert.match(js,/cell\.append\(node,stats\)/);
assert.match(js,/!!entry&&entry\.finished_ms==null&&entry===stages\.at\(-1\)/);
const metricsSource=js.slice(js.indexOf("const stageTime="),js.indexOf("const pipelineEdges="));
assert.ok(metricsSource.startsWith("const stageTime="));
const metricsContext=vm.createContext({});
vm.runInContext(metricsSource+"this.format=stageResultText;",metricsContext);
const format=metricsContext.format;
const run={status:"running",gpu_rally_mask:{merged:[[0,30],[90,130]]}};
const entry={started_ms:1000,finished_ms:null,frames_processed:1450,frames_total:1450};
assert.equal(format("players",entry,run,3662000,true,true),"1 450 / 1 450 кадров · 01:01:01");
assert.equal(format("rallies",{started_ms:1000,finished_ms:3662000,duration_ms:3661000},run,4000000,true,false),"2 розыгрыша · 01:01:01");
assert.equal(format("rallies",null,{status:"complete",gpu_rally_mask:{merged:[]}},0,false,false),"—");
assert.equal(format("video",{duration_ms:423000,finished_ms:500000},{status:"complete"},0,true,false),"Готово · 00:07:03");
assert.equal(format("touches",null,null,0,false,false),"—");
const sharedPose={gpu_jobs:{pose:"pose-run"},stages:[{id:"players",duration_ms:36000}],
  shared_pose:{source_stage_id:"players",sampled_frames:175,player_poses:612,duration_ms:36000}};
assert.equal(format("pose",null,sharedPose,0,true,false),
  "175 кадр. · 612 поз · 00:00:36 (общ.)");
assert.equal(format("pose",null,{gpu_jobs:{pose:"pose-run"},stages:[{id:"players",name:"GPU · RF-DETR + RTMPose",duration_ms:36000}]},0,true,false),
  "С RF-DETR · 00:00:36");
assert.equal(format("pose",null,{gpu_jobs:{pose:"pose-run"},stages:[]},0,true,false),
  "Позы сохранены · время не измерено");
for(const kind of ["input","gpu","cpu","output"]){
  assert.match(css,new RegExp('\\.analysis-flow-node\\[data-kind="'+kind+'"\\]\\{background:linear-gradient\\('));
}
assert.match(css,/\.analysis-flow-progress-dot\.is-visible\{opacity:1\}/);
assert.match(js,/dot\.append\(element\("animateMotion"/);
assert.match(js,/edge\.classList\.toggle\("is-active",!!running\)/);

const extract=(begin,end)=>{
  const start=js.indexOf(begin),stop=js.indexOf(end,start+begin.length);
  assert.ok(start>=0&&stop>start,begin);
  return js.slice(start,stop);
};
const definitions=extract("const pipelineStages=","const pipelineEdges=");
const context=vm.createContext({});
context.$=()=>({checked:false});
vm.runInContext(definitions+"this.stages=pipelineStages;this.id=pipelineId;this.model=pipelineModelMeta;",context);
assert.equal(context.model("ball")[0],"VballNetV4c");
assert.equal(context.model("players")[0],"RF-DETR Medium");
assert.equal(context.model("pose")[0],"RTMPose-M Body7");
assert.equal(context.model("tracking")[1],"algorithm");
assert.equal(context.model("touch_types")[0],"PoseMotion");
assert.equal(context.model("touch_types",{profile:{post:{touches:{algorithm:"trajectory_tcn"}}}})[0],"Volley3D TCN");
assert.equal(context.model("rallies"),null);
const stages=context.stages;
const ids=stages.map(item=>item[0]);
assert.equal(new Set(ids).size,ids.length);
assert.equal(stages.length,12);
assert.deepEqual([...new Set(stages.filter(row=>row[2]==="gpu"||row[2]==="input").map(row=>row[3]))].sort(),[1,2]);
assert.deepEqual([...new Set(stages.filter(row=>row[2]==="cpu"||row[2]==="output").map(row=>row[3]))].sort(),[3,4]);
assert.match(js,/analysis-flow-phase/);
assert.match(css,/\.analysis-flow-phase\[data-phase="gpu"\]/);
assert.match(css,/\.analysis-flow-phase\[data-phase="cpu"\]/);
for(const item of stages)assert.ok(item[1].split(/\s+/u).length<=2,item[1]);
const examples=[
  ["Проверка исходного видео","video"],
  ["GPU · VBallNet: траектория мяча и розыгрыши","ball"],
  ["Поиск розыгрышей","rallies"],
  ["GPU · RF-DETR + RTMPose: игровые интервалы","players"],
  ["GPU · RTMPose-M: позы игроков","pose"],
  ["Объединение · VBallNet и рамки игроков","fusion"],
  ["Постобработка · Track ID","tracking"],
  ["Постобработка · касания в первичных розыгрышах","touches"],
  ["Публикация розыгрышей","publish"],
  ["Постобработка · автоматическая привязка игроков","identity"],
  ["Монтажная временная разметка","material"]
];
for(const [name,id] of examples)assert.equal(context.id(name),id,name);
const edges=vm.runInNewContext(extract("const pipelineEdges=","const graphSettings=")
  .replace("const pipelineEdges=","").replace(/;\s*$/u,""));
assert.deepEqual(Array.from(edges.find(row=>row[2]==="no-pose")).slice(0,2),["fusion","tracking"]);
for(const [from,to] of edges){
  assert.ok(ids.includes(from)&&ids.includes(to),from+" -> "+to);
  assert.ok(ids.indexOf(from)<ids.indexOf(to),from+" -> "+to);
}
const timeline=extract("const renderStageTimeline=","const renderHistory=");
assert.doesNotMatch(timeline,/updatePipelineGraph|analysis-flow-grid|analysis-flow-diagram/);
assert.match(timeline,/analysis-history-stage-duration/);
assert.doesNotMatch(timeline,/analysis-history-stage-rail|analysis-history-stage-frames/);
assert.match(html,/id="analysis-flow-result"/);
assert.match(html,/id="studio-primary-heading">1 фаза/);
assert.match(html,/id="studio-post-heading">2 фаза/);
assert.match(html,/data-history-filter="gpu">1 фаза/);
assert.match(html,/data-history-filter="post">2 фаза/);
assert.match(js,/recordHistoryRun\(project,response\.id\)/);
assert.match(js,/recordHistoryRun\(project,status\.id\)/);
assert.match(js,/sessionStorage\?\.setItem\("volley:analysis-session-runs:v1:"/);
assert.match(js,/visibleIds\.has\(row\.id\)/);
assert.match(timeline,/make\("span","analysis-history-stage-name",pipelineLabel\(entry\)\)/);
assert.match(js,/updatePipelineGraph\(mainGraph,status\)/);
assert.match(rust,/rows\.push\(json!\(\{"name":stage_name,"id":stage_id/);
assert.match(rust,/status\["stage_id"\] = json!\(stage_id\)/);
assert.match(rust,/status\["shared_pose"\] = shared_pose_metrics\(status, &pose\)/);
assert.match(rust,/status\["shared_pose"\] = source\["shared_pose"\]\.clone\(\)/);
assert.match(html,/id="analysis-flow-diagram"/);
assert.match(html,/id="analysis-flow-dialog"/);
const debugSource=readFileSync(join(root,"rust/volley-app/static/studio.js"),"utf8");
const gpuRallyFields=["studio-gpu-rally-min-seconds","studio-gpu-rally-merge-seconds","studio-gpu-rally-stationary-ms"];
const debugRallyFields=["studio-comparison-min-seconds","studio-comparison-stationary-ms"];
for(const id of gpuRallyFields){
  assert.match(html,new RegExp('id="'+id+'"[^>]*data-default='));
  assert.match(js,new RegExp('inputValue\\("'+id+'"\\)'));
  assert.doesNotMatch(debugSource,new RegExp(id));
}
for(const id of debugRallyFields){
  assert.match(html,new RegExp('id="'+id+'"'));
  assert.doesNotMatch(js,new RegExp('inputValue\\("'+id+'"\\)'));
}
assert.match(js,/rallies:\["#studio-gpu-rally-min-seconds","#studio-gpu-rally-merge-seconds","#studio-gpu-rally-stationary-ms"\]/);
assert.match(js,/primary_rallies:\{\s*min_trajectory_seconds:/);
assert.match(js,/merge_gap_seconds:inputValue\("studio-gpu-rally-merge-seconds"\)/);
assert.match(js,/stationary_ball_ms:inputValue\("studio-gpu-rally-stationary-ms"\)/);

console.log("PASS: 12 full-width color-coded stages, verified neural models, animated edges, history parity and conditional branches");
