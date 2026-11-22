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
assert.match(css,/\.analysis-flow-node\{[^}]*width:max-content;max-width:100%;justify-self:center/);

const extract=(begin,end)=>{
  const start=js.indexOf(begin),stop=js.indexOf(end,start+begin.length);
  assert.ok(start>=0&&stop>start,begin);
  return js.slice(start,stop);
};
const definitions=extract("const pipelineStages=","const pipelineEdges=");
const context=vm.createContext({});
vm.runInContext(definitions+"this.stages=pipelineStages;this.id=pipelineId;",context);
const stages=context.stages;
const ids=stages.map(item=>item[0]);
assert.equal(new Set(ids).size,ids.length);
assert.equal(stages.length,12);
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
for(const [from,to] of edges){
  assert.ok(ids.includes(from)&&ids.includes(to),from+" -> "+to);
  assert.ok(ids.indexOf(from)<ids.indexOf(to),from+" -> "+to);
}
assert.match(js,/renderStageTimeline=[\s\S]*?updatePipelineGraph\(graph,current\|\|row\)/);
assert.match(js,/const name=make\("span","analysis-history-stage-name",pipelineLabel\(entry\)\)/);
assert.match(js,/updatePipelineGraph\(mainGraph,status\)/);
assert.match(rust,/rows\.push\(json!\(\{"name":stage_name,"id":stage_id/);
assert.match(rust,/status\["stage_id"\] = json!\(stage_id\)/);
assert.match(html,/id="analysis-flow-diagram"/);
assert.match(html,/id="analysis-flow-dialog"/);
console.log("PASS: 12 compact canonical stages, real Rust ids, history parity and conditional graph branches");
