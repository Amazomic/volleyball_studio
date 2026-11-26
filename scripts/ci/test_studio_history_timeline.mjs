// Timeline regression: a running stage has a live duration and ETA,
// finished stages retain exact durations, and live log nodes are never cloned.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const registry=source.slice(source.indexOf("  const pipelineStages="),
  source.indexOf("  const pipelineEdges="));
const from=source.indexOf("  const liveHistory=()=>{");
const to=source.indexOf("  const loadHistory=async(append=false)=>{",from);
assert.ok(from>=0&&to>from);
class Node{
  constructor(tag,klass="",text=""){
    this.tag=tag;this.className=klass;this._text=String(text);
    this.children=[];this.parent=null;this.dataset={};this.open=false;this.events={};
    this.style={};this.attributes={};
  }
  get textContent(){return this._text+this.children.map(c=>c.textContent).join("");}
  set textContent(value){
    for(const child of this.children)child.parent=null;
    this.children=[];this._text=String(value);
  }
  append(...nodes){
    for(const child of nodes){
      if(child.parent)child.remove();
      child.parent=this;this.children.push(child);
    }
  }
  remove(){if(!this.parent)return;
    const i=this.parent.children.indexOf(this);if(i>=0)this.parent.children.splice(i,1);
    this.parent=null;
  }
  querySelector(selector){
    const klass=selector.slice(1);
    for(const child of this.children){
      if(child.className.split(" ").includes(klass))return child;
      const found=child.querySelector(selector);if(found)return found;
    }
    return null;
  }
  setAttribute(name,value){this.attributes[name]=String(value);}
  addEventListener(name,callback){this.events[name]=callback;}
}
const host=new Node("div"),progress=new Node("progress"),logPanel=new Node("details");
host.append(progress,logPanel);
const historyMore={classList:{add(){},remove(){}}};
const ui={historyList:new Node("div"),progress,logPanel,historyMore,video:{value:"game.mp4"}};
const id="a".repeat(32),now=Date.now();
const stages=[
  {name:"GPU · VballNet",started_ms:now-90000,finished_ms:now-40000,duration_ms:50000},
  {name:"GPU · RF-DETR Medium + RTMPose",started_ms:now-40000,finished_ms:null,duration_ms:null,
    frames_processed:120,frames_total:400,progress_percent:30}
];
const current={
  schema:"volley-studio-pipeline/1",project_id:"aaaabbbbcccc",id,
  mode:"gpu",video:"game.mp4",stage:"GPU · RF-DETR Medium + RTMPose: running",
  status:"running",progress_percent:67.5,elapsed_ms:91000,remaining_ms:110000,stages
};
let project=current.project_id,graphSource;
const state={latestPipeline:current,historyRows:[],historyFilter:"all",historyStatusAt:now,
  historyOffset:0,historyEpoch:0,historyProject:project,formProject:project,trackCovers:new Map()};
const storage=new Map();
const sessionStorage={
  getItem:key=>storage.get(key)||null,
  setItem:(key,value)=>storage.set(key,value),
  removeItem:key=>storage.delete(key)
};
const {renderHistory,updateLiveHistory,recordHistoryRun}=runInNewContext(registry+source.slice(from,to)+
  "\n({renderHistory,updateLiveHistory,recordHistoryRun})",{
    state,ui,projectId:()=>project, $:key=>key==="studio-analysis-live-host"?host:null,
    make:(tag,klass,text)=>new Node(tag,klass,text),
    clear:node=>node.textContent="",historyDuration:ms=>ms===null?"—":Math.floor(ms/1000)+" с",
    historyLog:async()=>true,
    window:{VolleyballUi:{translate:x=>x},sessionStorage},mainGraph:host,
    showGraphResult:source=>{graphSource=source;},updatePipelineGraph:()=>{},
    requestAnimationFrame:()=>{}
  });
renderHistory();
assert.equal(ui.historyList.children.length,0,"Live stages belong to graph only");
recordHistoryRun("aaaabbbbcccc",id);
const finished={...current,status:"complete",progress_percent:100,remaining_ms:null,duration_ms:91000,
  stages:[stages[0],{...stages[1],finished_ms:now,duration_ms:40000}]};
updateLiveHistory(finished);
state.historyRows=[finished];
renderHistory();
const run=ui.historyList.children[0];
assert.equal(run.querySelector(".analysis-history-stages").children.length,2);
assert.match(run.querySelector(".analysis-history-stages").textContent,/\u0414\u0435\u0442\u0435\u043a\u0446\u0438\u044f \u043c\u044f\u0447\u0430.*50 \u0441/);
assert.match(run.querySelector(".analysis-history-stages").textContent,/40 \u0441/);
assert.equal(run.querySelector(".analysis-history-eta"),null);
assert.strictEqual(progress.parent,host);
assert.strictEqual(logPanel.parent,host);
const other={...finished,id:"b".repeat(32),video:"other.mp4"};
state.historyRows=[finished,other];
ui.video.value=other.video;
renderHistory();
assert.equal(graphSource?.id,other.id);
updateLiveHistory(finished);
assert.equal(graphSource?.id,other.id);
project="ddddeeeeffff";
renderHistory();
assert.equal(graphSource,null);
assert.equal(ui.historyList.children.length,0);
const projectReadyStart=source.indexOf('  window.addEventListener("volley:project-ready",()=>{');
const projectReadyEnd=source.indexOf("  for(const nav",projectReadyStart);
let projectReady,resolveHistory;
const loadHistory=runInNewContext(source.slice(to,source.indexOf("  ui.historyRefresh.addEventListener",to))+
  source.slice(projectReadyStart,projectReadyEnd)+"\nloadHistory",{
    state,ui,projectId:()=>project,renderHistory,liveHistory:()=>null,resetLog:()=>{},
    get:()=>new Promise(resolve=>{resolveHistory=resolve;}),
    window:{addEventListener:(event,callback)=>{projectReady=callback;}}
  });
project=current.project_id;
const pending=loadHistory();
project="ddddeeeeffff";
projectReady();
assert.equal(graphSource,null);
project=current.project_id;
projectReady();
resolveHistory({schema:"volley-studio-pipeline-history/1",project_id:project,
  runs:[other],next_offset:1});
await pending;
assert.equal(state.historyRows.length,0);
assert.equal(graphSource,null);
const resultPanel=new Node("div");
const activity=source.slice(source.indexOf("  const pipelineStageActivity="),
  source.indexOf("  const updatePipelineGraph="));
const graphLabel=source.slice(source.indexOf("  const graphLabel="),source.indexOf("  const graphText="));
const showGraphResult=runInNewContext(registry+graphLabel+activity+source.slice(
  source.indexOf("  const renderProjectQueue=queue=>{"),
  source.indexOf("  const liveHistory=()=>{")
)+"\nshowGraphResult",{
  $:id=>id==="analysis-flow-result"?resultPanel:null,
  clear:node=>node.textContent="",
  make:(tag,klass,text)=>new Node(tag,klass,text),
  ui:{video:{options:[{value:"game.mp4",textContent:"game.mp4"}]}},
  window:{VolleyballUi:{translate:text=>text}},historyDuration:ms=>Math.floor(ms/1000)+" с"
});
showGraphResult({...finished,elapsed_ms:91000});
assert.match(resultPanel.textContent,/Последний запуск · 1 фаза.*Завершён · 91 с/);
assert.equal(resultPanel.querySelector(".analysis-flow-result-title").textContent,
  "Последний запуск · 1 фаза · game.mp4");
showGraphResult({...finished,status:"running",stage_id:"fusion",stage:"Объединение · VBallNet и рамки игроков"});
assert.equal(resultPanel.dataset.activity,"running");
assert.match(resultPanel.textContent,/Объединение · VBallNet и рамки игроков/);
showGraphResult({...finished,status:"running",stage_id:"other",stage:"Ожидание очереди публикации"});
assert.equal(resultPanel.dataset.activity,"idle");
assert.match(resultPanel.textContent,/Ожидание очереди публикации/);
showGraphResult({...finished,status:"running",stage_id:"ball",stage:"VBallNet",live_stale:true});
assert.equal(resultPanel.dataset.activity,"idle");
assert.match(resultPanel.textContent,/Ожидание данных/);
const secondFeed={id:"next",video:"game-b.mp4",status:"running",stage_id:"ball",stage:"GPU · VBallNet",
  frames_processed:45,ball_frames_total:120};
const firstPass={...finished,status:"running",stage_id:"boxes",stage:"CPU · Связывание рамок игроков"};
const queue={active:true,videos:[firstPass,secondFeed]};
showGraphResult({...firstPass,project_queue:queue},queue);
assert.match(resultPanel.querySelector(".analysis-flow-result-title").textContent,
  /Видео 2\/2 → VballNetV4c · game-b\.mp4/);
const waitingFeed={...secondFeed,stage_id:"other",stage:"Ожидание очереди VBallNet"};
const waitingQueue={...queue,videos:[firstPass,waitingFeed]};
showGraphResult({...firstPass,project_queue:waitingQueue},waitingQueue);
assert.doesNotMatch(resultPanel.querySelector(".analysis-flow-result-title").textContent,/→ VballNet/);
showGraphResult(null);
assert.match(resultPanel.textContent,/После анализа здесь будет виден последний результат/);
console.log("History timeline: compact stage durations, latest result outside history and stable live log");
