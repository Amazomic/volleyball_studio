// Regression: live pipeline status and log belong to the newest history item,
// and refreshes never destroy or duplicate the streaming log nodes.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const registry=source.slice(source.indexOf("  const pipelineStages="),
  source.indexOf("  const pipelineEdges="));
const from=source.indexOf("  const liveHistory=()=>{");
const to=source.indexOf("  const loadHistory=async(append=false)=>{",from);
assert.ok(from>=0&&to>from,"Live-history logic must be present");

class Node {
  constructor(tag,klass="",text=""){
    this.tag=tag;this.className=klass;this._text=String(text);
    this.children=[];this.parent=null;this.dataset={};this.open=false;this.events={};
  }
  get textContent(){return this._text+this.children.map(node=>node.textContent).join("");}
  set textContent(value){
    for(const child of this.children)child.parent=null;
    this.children=[];this._text=String(value);
  }
  append(...nodes){
    for(const node of nodes){
      assert.ok(node instanceof Node);
      if(node.parent)node.remove();
      node.parent=this;this.children.push(node);
    }
  }
  remove(){
    if(this.parent){
      const index=this.parent.children.indexOf(this);
      if(index>=0)this.parent.children.splice(index,1);
      this.parent=null;
    }
  }
  querySelector(selector){
    const klass=selector.slice(1);
    for(const child of this.children){
      if(child.className.split(" ").includes(klass))return child;
      const nested=child.querySelector(selector);
      if(nested)return nested;
    }
    return null;
  }
  addEventListener(name,handler){this.events[name]=handler;}
}
const host=new Node("div");
const progress=new Node("progress");
const logPanel=new Node("details");
host.append(progress,logPanel);
const ui={historyList:new Node("div"),progress,logPanel};
const state={
  latestPipeline:null,historyRows:[],historyFilter:"all",historyOffset:0
};
let project="aaaabbbbcccc";
const getId=id=>id==="studio-analysis-live-host"?host:null;
const helpers={
  state,ui,projectId:()=>project,
  $:getId,make:(tag,klass,text)=>new Node(tag,klass,text),
  clear:node=>node.textContent="",
  historyDuration:()=> "—",
  historyLog:async()=>true,
  window:{VolleyballUi:{translate:x=>x}},updatePipelineGraph:()=>{},
  requestAnimationFrame:()=>{}
};
const result=runInNewContext(registry+source.slice(from,to)+
  "\n({renderHistory,updateLiveHistory,liveHistory})",helpers);
const id="a".repeat(32);
const current={
  schema:"volley-studio-pipeline/1",project_id:project,id,
  video:"match.mp4",mode:"post",status:"queued",stage:"Очередь",progress_percent:0
};
state.latestPipeline=current;
result.renderHistory();
assert.equal(ui.historyList.children.length,1);
let row=ui.historyList.children[0];
assert.equal(row.dataset.runId,id);
assert.equal(row.open,true,"Active history row should be expanded");
assert.equal(logPanel.parent.tag,"div","Streaming log is inside the history detail");
assert.equal(progress.parent,logPanel.parent);
assert.equal(row.querySelector(".analysis-history-status").textContent,"В очереди");
assert.strictEqual(row.querySelector(".analysis-history-live-stage").parent,row.children[0],
  "Live stage and percent must be visible in history summary without scrolling");

const running={...current,status:"running",stage:"CPU · касания",progress_percent:45.7};
result.updateLiveHistory(running);
assert.strictEqual(ui.historyList.children[0],row,"Polling must not replace the history row");
assert.equal(row.querySelector(".analysis-history-live-stage").textContent,"CPU · касания · 45.7%");
assert.equal(row.querySelector(".analysis-history-status").textContent,"Выполняется");

state.historyRows=[
  {id,mode:"post",video:"match.mp4",status:"queued",stages:[],started_ms:1000},
  {id:"b".repeat(32),mode:"gpu",video:"older.mp4",status:"complete",stages:[],started_ms:500}
];
result.renderHistory();
assert.equal(ui.historyList.children.length,2);
assert.strictEqual(progress.parent,logPanel.parent,"Progress and log remain colocated after reload");
assert.equal(host.children.length,0,"Live nodes must not remain duplicated in the hidden host");
row=ui.historyList.children[0];
assert.equal(row.querySelector(".analysis-history-live-stage").textContent,"CPU · касания · 45.7%");
assert.ok(ui.historyList.children[1].querySelector(".analysis-history-status"),
  "Historical run keeps its own status");

state.historyFilter="gpu";
result.renderHistory();
assert.equal(ui.historyList.children.length,2,"Active run remains visible regardless of history filter");
state.historyFilter="all";
result.updateLiveHistory({...running,status:"complete",stage:"Готово",progress_percent:100});
assert.equal(ui.historyList.children[0].querySelector(".analysis-history-status").textContent,"Завершён");

project="ddddaaaabbbb";
state.historyRows=[];state.historyFilter="all";
result.renderHistory();
assert.strictEqual(progress.parent,host,"Switching projects unmounts live progress");
assert.strictEqual(logPanel.parent,host,"Switching projects unmounts live log");
assert.equal(ui.historyList.children.length,0,"Different project must not display stale run");
console.log("Studio analysis history: live progress/log preserved across refresh, filtering and project switch");
