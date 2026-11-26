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
const metricsSource=js.slice(js.indexOf("const stageTime="),js.indexOf("const pipelineEdges="));
assert.ok(metricsSource.startsWith("const stageTime="));
const metricsContext=vm.createContext({});
vm.runInContext(metricsSource+"this.format=stageResultText;",metricsContext);
const format=metricsContext.format;
const blockRun={status:"running",stage_id:"players",rally_stage_progress:{
  boxes:{state:"complete",started_ms:1000,finished_ms:2000,frames_processed:100,frames_total:100,rallies_completed:2,rallies_total:2},
  pose:{state:"running",started_ms:2000,finished_ms:null,frames_processed:40,frames_total:100,rallies_completed:1,rallies_total:2,sampled_frames:20},
  detector_finished_ms:3000
}};
const blockContext=vm.createContext({});
vm.runInContext(metricsSource+"this.block=rallyStageEntry;this.state=rallyStageState;this.history=rallyStageHistory;",blockContext);
const detector=blockContext.block("players",{started_ms:500,frames_processed:100,frames_total:100},blockRun);
assert.equal(detector.finished_ms,3000);
assert.equal(detector.duration_ms,2500);
assert.equal(blockContext.state("players",detector,blockRun),"complete");
assert.equal(blockContext.state("boxes",blockContext.block("boxes",null,blockRun),blockRun),"complete");
assert.equal(blockContext.state("pose",blockContext.block("pose",null,blockRun),blockRun),"running");
const blockHistory=blockContext.history([{id:"players",started_ms:500},{id:"boxes",started_ms:5000}],blockRun);
assert.equal(blockHistory.length,3);
assert.equal(blockHistory[0].finished_ms,3000);
assert.equal(blockHistory[1].started_ms,1000);
assert.equal(blockHistory[2].started_ms,2000);
assert.match(format("pose",blockRun.rally_stage_progress.pose,blockRun,4000,false,true),/20 кадров.*1\/2 розыгр\./);
assert.match(format("boxes",{rallies_total:2,state:"waiting"},blockRun,4000,false,false),/— Track ID.*Ожидание/);
const touchRun={status:"running",stage_id:"players",rally_stage_progress:{
  ...blockRun.rally_stage_progress,
  touch_detector:{state:"running",started_ms:1500,finished_ms:null,frames_processed:45,frames_total:100,rallies_completed:1,rallies_total:2,active_duration_ms:30,block_started_ms:3500}
}};
const touchEntry=blockContext.block("touch_detector",null,touchRun);
assert.equal(blockContext.state("touch_detector",touchEntry,touchRun),"running");
assert.equal(touchEntry.frames_processed,45);
assert.match(format("touch_detector",touchEntry,touchRun,4000,false,true),/45 \/ 100 кадров.*1\/2 розыгр\./);
assert.equal(blockContext.history([{id:"touch_detector",started_ms:9000}],touchRun).filter(e=>e.id==="touch_detector").length,1);
assert.equal(blockContext.history([{id:"touch_detector",started_ms:9000}],touchRun).find(e=>e.id==="touch_detector").started_ms,1500);
assert.equal(blockContext.history([{id:"touch_detector",started_ms:9000}],blockRun).find(e=>e.id==="touch_detector").started_ms,9000);
const run={status:"running",gpu_rally_mask:{merged:[[0,30],[90,130]]}};
const entry={started_ms:1000,finished_ms:null,frames_processed:1450,frames_total:1450};
assert.equal(format("players",entry,run,3662000,true,true),"1 450 / 1 450 кадров · 01:01:01");
assert.equal(format("rallies",{started_ms:1000,finished_ms:3662000,duration_ms:3661000},run,4000000,true,false),"2 розыгрыша · 01:01:01");
assert.equal(format("rallies",{duration_ms:2000,finished_ms:3000},{gpu_rally_count:2,gpu_rally_mask:{ground_detector:{calibrated:true,candidate_count:3}}},4000,true,false),"2 розыгрыша · 00:00:02");
assert.equal(format("rallies",{duration_ms:2000,finished_ms:3000},{gpu_rally_count:2,gpu_rally_mask:{ground_detector:{calibrated:false,candidate_count:3}}},4000,true,false),"2 розыгрыша · 00:00:02");
assert.equal(format("rallies",null,{status:"complete",gpu_rally_mask:{merged:[]}},0,false,false),"—");
assert.equal(format("video",{duration_ms:423000,finished_ms:500000},{status:"complete"},0,true,false),"00:07:03");
assert.equal(format("touches",null,null,0,false,false),"—");
assert.equal(format("pose",{started_ms:1000,frames_processed:30,frames_total:120},run,3000,false,true),"30 / 120 кадров · 00:00:02");
assert.equal(format("pose",{duration_ms:4000,finished_ms:5000,frames_processed:120,frames_total:120},run,5000,true,false),"120 / 120 кадров · 00:00:04");
assert.equal(format("pose",null,{primary_tracking:{pose_ready:true},pose_frame_progress:{processed:120,total:120}},0,true,false),"120 / 120 кадров · Из кэша");
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
assert.match(css,/\.analysis-flow-node\{[^}]*background:var\(--flow-card\)/);
for(const kind of ["input","gpu","cpu","mixed","output"]){
  assert.match(css,new RegExp('\\.analysis-flow-node\\[data-kind="'+kind+'"\\]\\s+\\.analysis-flow-node-icon\\{color:'));
  assert.doesNotMatch(css,new RegExp('\\.analysis-flow-node\\[data-kind="'+kind+'"\\]\\{[^}]*background:'));
}
assert.match(css,/\.analysis-flow-node\.is-complete\{[^}]*border-color:var\(--flow-complete\)/);
assert.match(css,/\.analysis-flow-node\.is-active\{[^}]*border-color:var\(--flow-active\)/);
assert.doesNotMatch(css,/\.analysis-flow-node[^{}]*(?:hover|focus)[^{}]*\{[^}]*outline:[^;}]*var\(--flow-active\)/);
assert.match(css,/\.analysis-flow-diagram:not\(:has\(\.analysis-flow-node\.is-active\)\) \.analysis-flow-node\.is-linked-focus\{[^}]*outline:1px dashed var\(--flow-muted\)/);
assert.match(css,/body\[data-ui-theme="light"\] \.analysis-flow\{/);
assert.doesNotMatch(html,/analysis-flow-legend|analysis-flow-focus/);
assert.ok(html.indexOf('id="analysis-flow-result"')<html.indexOf('id="analysis-flow-diagram"'));
assert.doesNotMatch(js,/analysis-flow-phase-summary|graphPhaseSummary|focusSelect|_graphFocusSelect/);
assert.match(js,/const id=hoveredStage\|\|keyboardStage;/);
assert.match(css,/\.analysis-flow-phase\[data-phase="cpu"\] \.analysis-flow-cards\{grid-template-columns:repeat\(4,minmax\(0,1fr\)\);direction:rtl/);
assert.match(js,/edge\.classList\.toggle\("is-active",!!running\)/);

const extract=(begin,end)=>{
  const start=js.indexOf(begin),stop=js.indexOf(end,start+begin.length);
  assert.ok(start>=0&&stop>start,begin);
  return js.slice(start,stop);
};
const definitions=extract("const pipelineStages=","const pipelineEdges=");
const context=vm.createContext({});
context.$=()=>({checked:false});
vm.runInContext(definitions+"this.stages=pipelineStages;this.id=pipelineId;this.model=pipelineModelMeta;this.combine=combineRallyStages;this.icon=pipelineIcon;",context);
assert.equal(context.model("ball")[0],"VballNetV4c");
assert.equal(context.model("ball_stabilizer")[0],"Ball Stabilizer");
assert.equal(context.model("players")[0],"RF-DETR Medium");
assert.equal(context.model("pose")[0],"RTMPose-M Body7");
assert.equal(context.model("tracking")[1],"algorithm");
assert.equal(context.model("touch_detector")[0],"Touch Detector");
context.document={createElementNS:(_ns,tag)=>({tag,children:[],setAttribute(){},append(child){this.children.push(child);}})};
for(const id of context.stages.map(item=>item[0])){
  const icon=context.icon(id);
  assert.ok(icon.children.length>0,"Every stage must render an icon: "+id);
}
assert.equal(context.model("ground_detector"),null);
assert.equal(context.id({id:"pose",name:"Ожидание очереди RTMPose"}),"");
const legacyGround={id:"ground_detector",name:"Ground Detector",started_ms:1000,finished_ms:4000,duration_ms:3000};
const legacyRally={id:"rallies",name:"Поиск розыгрышей",started_ms:4000,finished_ms:5000,duration_ms:1000};
const historical=context.combine([legacyGround,legacyRally]);
assert.equal(historical.length,1);
assert.equal(historical[0].id,"rallies");
assert.equal(historical[0].started_ms,1000);
assert.equal(historical[0].duration_ms,4000);
assert.equal(context.combine([legacyRally]).length,1);
assert.equal(context.id({id:"touch_types"}),"touches");
const oldTouch={id:"touches",started_ms:5000,finished_ms:6000,duration_ms:1000};
const oldPose={id:"touch_types",started_ms:6000,finished_ms:8000,duration_ms:2000};
const historicalTouch=context.combine([oldTouch,oldPose]);
assert.equal(historicalTouch.length,1);
assert.equal(historicalTouch[0].id,"touches");
assert.equal(historicalTouch[0].duration_ms,3000);
assert.equal(context.combine([oldTouch,{...oldPose,started_ms:5000,finished_ms:6000,duration_ms:1000}])[0].duration_ms,1000);
assert.equal(legacyRally.duration_ms,1000);
assert.equal(context.model("touch_types"),null);
assert.equal(context.model("rallies")[1],"algorithm");
const labelContext=vm.createContext({window:{VolleyballUi:{translate:x=>x}},$:context.$});
vm.runInContext(definitions+extract("const graphLabel=","const graphText=")+"this.label=graphLabel;",labelContext);
for(const [id,name] of [["ball","VballNetV4c"],["ball_stabilizer","Ball Stabilizer"],["players","RF-DETR Medium"],["touch_detector","Touch Detector"],["boxes","ByteTrack · BBox"],["tracking","ByteTrack · Track ID"],["touches","Touch Attribution"],["rallies","Rally Detector"],["identity","ReID"]])assert.equal(labelContext.label(id),name,id);
assert.equal(labelContext.label("players",{profile:{gpu:{person_detector:"rfdetr-large"}}}),"RF-DETR Large");
assert.doesNotMatch(extract("const createPipelineGraph=","const updatePipelineGraph="),/analysis-flow-node-model/);
assert.match(extract("const updatePipelineGraph=","const paintGraphEdges="),/node\.title=graphHoverDescriptions\[id\]/);
const stages=context.stages;
const ids=stages.map(item=>item[0]);
assert.equal(new Set(ids).size,ids.length);
assert.equal(stages.length,12);
assert.ok(!ids.includes("fusion"));
assert.match(css,/\.analysis-flow-phase\[data-phase="gpu"\] \.analysis-flow-cards\{grid-template-columns:repeat\(8,minmax\(0,1fr\)\)/);
assert.ok(!ids.includes("photos"));
assert.ok(!ids.includes("touch_types"));
assert.ok(!ids.includes("ground_detector"));
assert.ok(ids.includes("publish")&&!ids.includes("material"));
for(const [row,count] of [[1,8],[2,4]]){
  const nodes=stages.filter(item=>item[3]===row);
  assert.equal(nodes.length,count);
}
assert.match(css,/\.analysis-flow-node::after\{[^}]*linear-gradient\(to right,/);
assert.doesNotMatch(css,/\.analysis-flow-node::after\{[^}]*conic-gradient/);
assert.match(js,/node\.style\.setProperty\("--stage-progress"/);
assert.match(css,/\.analysis-flow-node\.is-complete:not\(\.is-active\)::after\{opacity:0\}/);
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
  ["Ground Detector","rallies"],
  ["Rally Detector","rallies"],
  ["Поиск розыгрышей","rallies"],
  ["GPU · RF-DETR + RTMPose: игровые интервалы","players"],
  ["GPU · RTMPose-M: позы игроков","pose"],
  ["Объединение · VBallNet и рамки игроков","fusion"],
  ["CPU · Связывание рамок игроков","boxes"],
  ["Постобработка · Track ID","tracking"],
  ["Постобработка · касания в первичных розыгрышах","touches"],
  ["Touch Attribution","touches"],
  ["Pose Geometry","touches"],
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
assert.match(js,/const rows=projectHistoryRows\(\)\.filter\(row=>/);
assert.match(timeline,/make\("span","analysis-history-stage-name",pipelineLabel\(entry\)\)/);
assert.match(js,/updatePipelineGraph\(mainGraph,source\)/);
assert.match(rust,/rows\.push\(json!\(\{"name":stage_name,"id":stage_id/);
assert.match(rust,/status\["stage_id"\] = json!\(stage_id\)/);
assert.match(rust,/status\["shared_pose"\] = shared_pose_metrics\(status, &pose\)/);
assert.match(rust,/status\["shared_pose"\] = source\["shared_pose"\]\.clone\(\)/);
assert.match(html,/id="analysis-flow-diagram"/);
assert.match(html,/id="analysis-flow-dialog"/);
const debugSource=readFileSync(join(root,"rust/volley-app/static/studio.js"),"utf8");
const gpuRallyFields=["studio-gpu-rally-min-seconds","studio-gpu-rally-merge-seconds","studio-gpu-rally-stationary-ms","studio-gpu-rally-padding-before-ms","studio-gpu-rally-padding-after-ms"];
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
assert.match(js,/rallies:\["#studio-gpu-rally-min-seconds","#studio-gpu-rally-merge-seconds","#studio-gpu-rally-stationary-ms","#studio-gpu-rally-padding-before-ms","#studio-gpu-rally-padding-after-ms"\]/);
assert.match(js,/primary_rallies:\{\s*min_trajectory_seconds:/);
assert.match(js,/merge_gap_seconds:inputValue\("studio-gpu-rally-merge-seconds"\)/);
assert.match(js,/stationary_ball_ms:inputValue\("studio-gpu-rally-stationary-ms"\)/);

assert.match(js,/identity:"[^"]*исходных кадров[^"]*"/);
const dataLinks=vm.runInNewContext(extract("const pipelineEdges=","const graphSettings=")
  +"({links:pipelineEdges,state:graphLinkState});");
const actualInputs=target=>Array.from(dataLinks.links.filter(link=>link[1]===target),link=>link[0]).sort();
assert.deepEqual(actualInputs("ball_stabilizer"),["ball"]);
assert.deepEqual(actualInputs("rallies"),["ball_stabilizer"]);
assert.deepEqual(actualInputs("fusion"),[]);
assert.deepEqual(actualInputs("boxes"),["ball","players","rallies"]);
assert.deepEqual(actualInputs("tracking"),["boxes","pose"]);
assert.equal(dataLinks.state("no_pose",{profile:{gpu:{pose_enabled:false}}}),"available");
assert.equal(dataLinks.state("no_pose",{gpu_jobs:{pose:"saved"}}),"disabled");
assert.deepEqual(actualInputs("ground_detector"),[]);
assert.equal(context.id({id:"ground_detector"}),"rallies");
assert.equal(stages.find(item=>item[0]==="boxes")[3],1);
assert.equal(stages.find(item=>item[0]==="pose")[3],1);
const pipelineBody=rust.slice(rust.indexOf("async fn pipeline("),rust.indexOf("enum CompletionQuality"));
const replayStart=pipelineBody.indexOf("cpu_retrack::process_with_cancel");
const primaryEnd=pipelineBody.indexOf("if request.mode == PipelineMode::Gpu {");
assert.ok(replayStart>=0&&primaryEnd>replayStart,"standalone primary analysis must save boxes and pose before returning");
assert.ok(primaryEnd<pipelineBody.indexOf("cpu_touches::stabilize_primary_replay"));
const detectorStart=pipelineBody.indexOf("ground_stage::prepare_touch_detection");
assert.ok(detectorStart>=0&&detectorStart<pipelineBody.indexOf("cpu_touches::attribute_primary_with_progress_cancel"));
assert.match(pipelineBody,/progress\(path, status, "Touch Attribution", 91\.5\)/);
assert.match(pipelineBody,/request\.mode\.allows_gpu_execution\(\) && request\.gpu\.pose_enabled/);
assert.match(js,/payload\.post=postProfile\(\);/);
assert.doesNotMatch(js,/if\(mode!=="gpu"\)payload\.post/);
assert.equal(stages.find(item=>item[0]==="touch_detector")[3],1);
assert.deepEqual(actualInputs("touch_detector"),["rallies"]);
assert.ok(stages.findIndex(([id])=>id==="rallies")+1===stages.findIndex(([id])=>id==="touch_detector"));
assert.ok(actualInputs("players").includes("touch_detector"));
assert.deepEqual(actualInputs("touches"),["ball_stabilizer","pose","rallies","tracking"]);
assert.deepEqual(actualInputs("publish"),["rallies","touches","tracking"]);
assert.deepEqual(actualInputs("identity"),["publish","tracking","video"]);
assert.equal(dataLinks.state("pose",null),"optional");
assert.equal(dataLinks.state("pose",{profile:{gpu:{pose_enabled:false}}}),"disabled");
assert.equal(dataLinks.state("pose",{gpu_reused:true,gpu_jobs:{pose:"saved"}}),"available");
assert.deepEqual(actualInputs("pose"),["boxes","video"]);
assert.deepEqual(actualInputs("touch_types"),[]);
assert.equal(dataLinks.state("pose",{profile:{post:{pose_classification:false}},gpu_jobs:{pose:"saved"}}),"available");
const withClass=(stageId,classes)=>({
  dataset:{stageId},classList:{contains:name=>classes.includes(name)},
  setAttribute(name,value){this[name]=value;},removeAttribute(name){delete this[name];}
});
const playerNode=withClass("players",["is-reused"]);
const trackingNode=withClass("boxes",["is-active"]);
const link={
  dataset:{edgeSource:"players",edgeTarget:"boxes",edgeCondition:""},
  classList:{toggle(name,enabled){this[name]=enabled;}},
  setAttribute(name,value){this[name]=value;}
};
const diagram={querySelectorAll(selector){
  return selector===".analysis-flow-node"?
    [playerNode,trackingNode]:[link];
}};
const edgeContext=vm.createContext({diagram,window:{VolleyballUi:{
  locale:"en",
  translate(value){return this.locale==="en"?{"Поиск игроков":"Player detection","Трекинг ID":"Track IDs"}[value]||value:value;}
}}});
vm.runInContext(definitions+extract("const pipelineEdges=","const graphSettings=")+
  extract("const paintGraphEdges=",'$("analysis-flow-feedback")')+"paintGraphEdges(diagram,{gpu_reused:true});",edgeContext);
assert.equal(link.dataset.state,"available");
assert.equal(link.textContent,undefined);
assert.equal(trackingNode["aria-description"],"Player detection");
assert.equal(link.classList["is-active"],true);
const card=id=>({
  dataset:{stageId:id},
  classList:{
    toggle(name,enabled){this[name]=enabled;},
    contains(name){return Boolean(this[name]);}
  },
  style:{setProperty(){}},
  removeAttribute(){},
  querySelector(){return {classList:{toggle(){}}};}
});
const ballCard=card("ball"),poseCard=card("pose"),detectorCard=card("touch_detector"),touchCard=card("touches");
const graph={
  querySelector:()=>true,
  querySelectorAll:selector=>selector===".analysis-flow-node"?[ballCard,poseCard,detectorCard,touchCard]:[]
};
const stageContext=vm.createContext({graph,window:edgeContext.window,document:{},$:context.$});
vm.runInContext(definitions+extract("const pipelineEdges=","const graphSettings=")+
  extract("const graphDescriptions=","let graphMoved=")+
  extract("const graphLabel=",'$("analysis-flow-feedback")')+
  'updatePipelineGraph(graph,{status:"complete",gpu_jobs:{pose:"saved"},stages:[{id:"ball",finished_ms:2000}]});',stageContext);
assert.equal(ballCard.classList["is-complete"],true);
assert.equal(ballCard.title,"Находит мяч на кадрах видео.");
assert.equal(ballCard.dataset.status,"complete");
assert.equal(poseCard.dataset.status,"complete");
assert.equal(poseCard.classList["is-complete"],true);
ballCard.classList.toggle("is-linked-focus",true);
vm.runInContext('updatePipelineGraph(graph,{status:"running",stage_id:"pose",stages:[{id:"ball",finished_ms:2000},{id:"pose",started_ms:2000}]});',stageContext);
assert.equal(ballCard.classList["is-active"],false);
assert.equal(ballCard.classList["is-complete"],true);
assert.equal(poseCard.classList["is-active"],true);
assert.equal(poseCard.dataset.status,"running");
assert.equal(ballCard.title,"Находит мяч на кадрах видео.");
vm.runInContext('updatePipelineGraph(graph,{status:"running",stage_id:"touches",profile:{gpu:{pose_enabled:false}},stages:[{id:"touch_detector",finished_ms:1000},{id:"touches",started_ms:1000}]});',stageContext);
assert.equal(touchCard.classList["is-active"],true);
assert.equal(touchCard.classList["is-skipped"],false);
vm.runInContext('updatePipelineGraph(graph,{status:"complete",profile:{gpu:{pose_enabled:false}},stages:[{id:"touches",finished_ms:3000}]});',stageContext);
assert.equal(detectorCard.classList["is-skipped"],false);
assert.equal(detectorCard.dataset.status,"idle");
assert.equal(touchCard.classList["is-complete"],true);
assert.equal(touchCard.classList["is-skipped"],false);
vm.runInContext('updatePipelineGraph(graph,{status:"complete",gpu_reused:true,primary_tracking:{pose_ready:true,pose_sampling:{version:"touch_guided_pose_sampling_v1"}}});',stageContext);
assert.equal(detectorCard.classList["is-reused"],true);
assert.equal(detectorCard.dataset.status,"complete");
vm.runInContext('updatePipelineGraph(graph,{status:"failed",stage_id:"ball",stages:[{id:"ball",finished_ms:3000}]});',stageContext);
assert.equal(ballCard.dataset.status,"failed");
assert.equal(poseCard.dataset.status,"idle");
const render=run=>{stageContext.testRun=run;vm.runInContext("updatePipelineGraph(graph,testRun)",stageContext);};
render({status:"running",stage_id:"boxes",stage:"CPU · Связывание рамок игроков",
  stages:[{id:"ball",finished_ms:1000},{id:"boxes",started_ms:2000}],
  project_queue:{active:true,videos:[{id:"next",status:"running",stage_id:"ball",video:"next.mp4"}]}});
assert.equal(ballCard.dataset.status,"running","VBallNet must stay active while another video processes CPU stages");
assert.equal(ballCard.classList["is-complete"],false);
render({status:"running",stage_id:"fusion",stage:"Объединение · VBallNet и рамки игроков",
  project_queue:{active:true,videos:[{id:"next",status:"running",stage_id:"players",video:"next.mp4",
    rally_stage_progress:{detector_finished_ms:2500,pose:{state:"running"}}}]}});
assert.equal(ballCard.dataset.status,"idle");
assert.equal(poseCard.dataset.status,"running","A live pose block from another video must reach the graph");
render({status:"running",stage_id:"pose",stage:"RTMPose",
  stages:[{id:"pose",started_ms:2000}],
  rally_stage_progress:{pose:{state:"complete",rallies_total:1}}});
assert.equal(poseCard.dataset.status,"running","A completed early pose pass must not mask a new pose pass");
render({status:"running",stage_id:"other",stage:"Ожидание очереди VBallNet",
  stages:[{id:"ball",started_ms:1000,finished_ms:2000},{id:"other",started_ms:2000}]});
assert.equal(ballCard.dataset.status,"complete","Waiting must never animate a completed model");
render({status:"running",stage_id:"ball",live_stale:true});
assert.equal(ballCard.dataset.status,"idle","A stale snapshot must stop the activity animation");
assert.match(js,/analysis-flow-node-device/);
assert.match(js,/identity:"GPU\/CPU"/);
assert.match(js,/analysis-flow-node-status/);
assert.match(css,/data-status="complete"/);
assert.match(css,/data-status="failed"/);
const runningIndicator=css.match(/\.analysis-flow-node\[data-status="running"\] \.analysis-flow-node-status\{([^}]+)\}/)?.[1];
assert.ok(runningIndicator);
assert.match(runningIndicator,/background:transparent/);
assert.match(runningIndicator,/border:2px solid var\(--flow-active\)/);
assert.match(runningIndicator,/border-right-color:transparent/);
assert.match(runningIndicator,/animation:analysis-stage-status-spin 1s linear infinite/);
assert.match(css,/@keyframes analysis-stage-status-spin\{to\{transform:rotate\(360deg\)\}\}/);
assert.doesNotMatch(css,/\.analysis-flow-node\[data-status="running"\] \.analysis-flow-node-status::after/);
assert.match(css,/@media\(prefers-reduced-motion:reduce\)\{[^}]*\.analysis-flow-node\[data-status="running"\] \.analysis-flow-node-status\{animation:none\}/);
const historyProjection=rust.slice(rust.indexOf('rows.push(json!({"id":id'),rust.indexOf("    rows.sort_by"));
for(const key of ["gpu_jobs","cpu_runs","analysis_id","profile","gpu_rally_count","automatic_identity_available"])
  assert.ok(historyProjection.includes('"'+key+'":'),"Saved graph needs "+key);
assert.doesNotMatch(js,/analysis-flow-inputs|analysis-flow-reference/);
assert.match(js,/"marker-end":"url\(#"\+arrowId/);
assert.match(css,/analysis-flow-edge\[data-state="disabled"\]\{visibility:hidden\}/);
console.log("PASS: analysis stages, verified data inputs, optional pose links and history parity");
