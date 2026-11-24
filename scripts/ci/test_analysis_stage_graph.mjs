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
assert.match(css,/\.analysis-flow-diagram\{[^}]*container-type:inline-size/);
assert.match(css,/\.analysis-flow-node\{[^}]*min-height:76px/);
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
const scopedPose={gpu_jobs:{pose:"pose-run"},stages:[{id:"players",duration_ms:36000}],
  shared_pose:{source_stage_id:"players",sampled_frames:175,eligible_pose_frames:50,player_poses:612,duration_ms:36000}};
assert.equal(format("pose",null,scopedPose,0,true,false),
  "50 игровых кадр. · 612 поз · 00:00:36 (общ.)");
assert.equal(format("pose",null,sharedPose,0,true,false),
  "175 выборок · 612 поз · 00:00:36 (общ.)");
assert.equal(format("pose",null,{gpu_jobs:{pose:"pose-run"},stages:[{id:"players",name:"GPU · RF-DETR + RTMPose",duration_ms:36000}]},0,true,false),
  "С RF-DETR · 00:00:36");
assert.equal(format("pose",null,{gpu_jobs:{pose:"pose-run"},stages:[]},0,true,false),
  "Позы сохранены · время не измерено");
for(const kind of ["input","gpu","cpu","mixed","output"]){
  assert.match(css,new RegExp('\\.analysis-flow-node\\[data-kind="'+kind+'"\\]\\{background:linear-gradient\\('));
}
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
assert.equal(context.model("ball_stabilizer")[0],"Ball Stabilizer");
assert.equal(context.model("players")[0],"RF-DETR Medium");
assert.equal(context.model("pose")[0],"RTMPose-M Body7");
assert.equal(context.model("tracking")[1],"algorithm");
assert.equal(context.model("touches")[0],"Touch Detector");
assert.equal(context.model("touch_types")[1],"algorithm");
assert.equal(context.model("rallies")[1],"algorithm");
const stages=context.stages;
const ids=stages.map(item=>item[0]);
assert.equal(new Set(ids).size,ids.length);
assert.equal(stages.length,12);
assert.ok(ids.includes("publish")&&!ids.includes("material"));
for(const [row,count] of [[1,6],[2,6]]){
  const nodes=stages.filter(item=>item[3]===row);
  assert.equal(nodes.length,count);
}
assert.match(css,/\.analysis-flow-node::after\{[^}]*linear-gradient\(to right,/);
assert.doesNotMatch(css,/\.analysis-flow-node::after\{[^}]*conic-gradient/);
assert.match(js,/node\.style\.setProperty\("--stage-progress"/);
assert.match(css,/\.analysis-flow-node\.is-complete\{[^}]*border-color:#efba4b/);
assert.match(css,/\.analysis-flow-node::before,\.analysis-flow-node::after\{[^}]*width:var\(--stage-progress,0%\)/);
assert.match(css,/\.analysis-flow-node::before\{top:0/);
assert.match(css,/\.analysis-flow-node::after\{bottom:0/);
assert.doesNotMatch(css,/\.analysis-flow-node\.is-active\{[^}]*22px/);
assert.match(css,/@media\(prefers-reduced-motion:reduce\)[^\n]*\.analysis-flow-node\.is-active::before/);
assert.equal((css.match(/@property --stage-progress/g)||[]).length,1);
assert.match(css,/@property --stage-progress\{syntax:"<percentage>";inherits:true/);
assert.match(css,/\.analysis-flow-node::before,\.analysis-flow-node::after\{[^}]*height:1\.5px/);
assert.match(css,/\.analysis-flow-dialog-body\{[^}]*grid-template-columns:repeat\(2,minmax\(0,1fr\)\)/);
assert.match(css,/\.analysis-flow-dialog-body>div:not\(\.analysis-profile-toggle\):not\(\.analysis-editor\)\{[^}]*grid-template-columns:minmax\(0,1fr\) minmax\(95px,138px\)/);
assert.match(css,/\.analysis-flow-dialog-panel\{[^}]*width:min\(1040px,100%\)/);
assert.match(css,/\.analysis-flow-dialog-panel\.is-editor\{width:min\(1200px,100%\)/);
assert.match(css,/\.analysis-flow-dialog-panel\.is-compact\{width:min\(780px,100%\)/);
assert.match(css,/\.analysis-flow-dialog-panel\.is-compact \.analysis-flow-dialog-body\{overflow:visible/);
assert.match(css,/\.analysis-flow-dialog-body\{[^}]*scrollbar-gutter:auto/);
assert.match(js,/ball:\["#studio-gpu-stride","#studio-gpu-memory","#studio-gpu-threads"\]/);
assert.match(js,/panel\.classList\.toggle\("is-compact",id!=="video"&&body\.children\.length<=4\)/);
assert.match(js,/openGraphSettings\("ball_stabilizer"\)/);
assert.match(js,/analysis-flow-dialog-related/);
assert.match(js,/id==="ball"/);
assert.match(js,/graphDescriptions\[id\]/);
assert.doesNotMatch(css,/\.analysis-flow-dialog-panel\.is-(single|medium)/);
assert.match(css,/\.analysis-flow-dialog-panel>footer\{[^}]*justify-content:space-between/);
assert.match(html,/<footer><p id="analysis-flow-dialog-description" class="panel-note"><\/p><button id="analysis-flow-dialog-ok"[^>]*>Готово<\/button><\/footer>/);
assert.match(css,/#studio-analysis-view \.analysis-flow-dialog \.analysis-setting-help:is\(:hover,:focus-visible\)::after\{display:none\}/);
assert.match(css,/\.analysis-flow-stage-hint\{position:fixed;z-index:3/);
assert.match(css,/#studio-analysis-view \.analysis-flow-dialog \.analysis-help>\.analysis-tooltip\{display:none\}/);
assert.match(css,/@media\(max-width:700px\)[^\n]*\.analysis-flow-dialog-body\{grid-template-columns:minmax\(0,1fr\)/);
assert.match(js,/graphDialog\.append\(graphHint\)/);
assert.match(js,/graphDialog\.addEventListener\("pointerover"/);
assert.match(js,/graphDialog\.addEventListener\("focusin"/);
assert.match(js,/graphDialog\.addEventListener\("scroll",hideGraphHint,true\)/);
assert.match(js,/for\(const \[node,placeholder,wasOpen\] of graphMoved\)/);
assert.match(js,/placeholder\.replaceWith\(node\)/);
const stagePercent=extract("const stageProgressPercent=","const pipelineEdges=");
const pctContext=vm.createContext({});
vm.runInContext(stagePercent+"this.percent=stageProgressPercent;",pctContext);
assert.equal(pctContext.percent({frames_processed:50,frames_total:100},true,false),50);
assert.equal(pctContext.percent({progress_percent:42.5},true,false),42.5);
assert.equal(pctContext.percent(null,false,true),100);
assert.equal(pctContext.percent(null,true,false),null);
assert.equal(pctContext.percent({frames_processed:50,frames_total:100},false,false),0);
assert.match(js,/analysis-flow-phase/);
const examples=[
  ["Проверка исходного видео","video"],
  ["GPU · VBallNet: траектория мяча и розыгрыши","ball"],
  ["Интерполяция траектории мяча · Ball Stabilizer","ball_stabilizer"],
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
const edges=vm.runInNewContext(extract("const pipelineEdges=","const graphSettings=")+"pipelineEdges;");
for(const [from,to] of edges){
  assert.ok(ids.includes(from)&&ids.includes(to),from+" -> "+to);
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
assert.match(js,/updatePipelineGraph\(mainGraph,source\)/);
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

const dataLinks=vm.runInNewContext(extract("const pipelineEdges=","const graphSettings=")
  +"({links:pipelineEdges,state:graphLinkState});");
const actualInputs=target=>Array.from(dataLinks.links.filter(link=>link[1]===target),link=>link[0]).sort();
assert.deepEqual(actualInputs("fusion"),["ball","players"]);
assert.deepEqual(actualInputs("tracking"),["fusion","pose","rallies"]);
assert.deepEqual(actualInputs("touches"),["ball_stabilizer","rallies","tracking"]);
assert.deepEqual(actualInputs("publish"),["rallies","touches","tracking"]);
assert.deepEqual(actualInputs("identity"),["publish","tracking","video"]);
assert.equal(dataLinks.state("pose",null),"optional");
assert.equal(dataLinks.state("pose",{profile:{gpu:{pose_enabled:false}}}),"disabled");
assert.equal(dataLinks.state("pose",{gpu_reused:true,gpu_jobs:{pose:"saved"}}),"available");
assert.deepEqual(actualInputs("pose"),["fusion","players","video"]);
assert.deepEqual(actualInputs("touch_types"),["pose","touches"]);
assert.equal(dataLinks.state("pose",{profile:{post:{pose_classification:false}},gpu_jobs:{pose:"saved"}}),"available");
const withClass=(stageId,classes)=>({dataset:{stageId},classList:{contains:name=>classes.includes(name)}});
const link={
  dataset:{edgeSource:"fusion",edgeTarget:"tracking",edgeCondition:"",edgeData:"Объединённые наблюдения"},
  classList:{toggle(name,enabled){this[name]=enabled;}},
  setAttribute(name,value){this[name]=value;}
};
const diagram={querySelectorAll(selector){
  return selector===".analysis-flow-node"?
    [withClass("fusion",["is-reused"]),withClass("tracking",["is-active"])]:[link];
}};
const edgeContext=vm.createContext({diagram,window:{VolleyballUi:{
  locale:"en",
  translate(value){return this.locale==="en"?{"Данные GPU":"GPU data","Трекинг ID":"Track IDs"}[value]||value:value;}
}}});
vm.runInContext(definitions+extract("const pipelineEdges=","const graphSettings=")+
  extract("const paintGraphEdges=",'$("analysis-flow-feedback")')+"paintGraphEdges(diagram,{gpu_reused:true});",edgeContext);
assert.equal(link.textContent,"GPU data · Объединённые наблюдения");
assert.equal(link["aria-label"],"GPU data → Track IDs: Объединённые наблюдения");
assert.equal(link.classList["is-active"],true);
const card=id=>({
  dataset:{stageId:id},
  classList:{
    toggle(name,enabled){this[name]=enabled;},
    contains(name){return Boolean(this[name]);}
  },
  style:{setProperty(){}},
  querySelector(){return {classList:{toggle(){}}};}
});
const ballCard=card("ball"),poseCard=card("pose");
const graph={
  querySelector:()=>true,
  querySelectorAll:selector=>selector===".analysis-flow-node"?[ballCard,poseCard]:[]
};
const stageContext=vm.createContext({graph,window:edgeContext.window,document:{},$:context.$});
vm.runInContext(definitions+extract("const pipelineEdges=","const graphSettings=")+
  extract("const graphLabel=",'$("analysis-flow-feedback")')+
  'updatePipelineGraph(graph,{status:"complete",gpu_jobs:{pose:"saved"},stages:[{id:"ball",finished_ms:2000}]});',stageContext);
assert.equal(ballCard.classList["is-complete"],true);
assert.equal(poseCard.classList["is-complete"],true);
const historyProjection=rust.slice(rust.indexOf('rows.push(json!({"id":id'),rust.indexOf("    rows.sort_by"));
for(const key of ["gpu_jobs","cpu_runs","analysis_id","profile","gpu_rally_count","automatic_identity_available"])
  assert.ok(historyProjection.includes('"'+key+'":'),"Saved graph needs "+key);
assert.doesNotMatch(js,/graphDraw\(/);
console.log("PASS: analysis stages, verified data inputs, optional pose links and history parity");
