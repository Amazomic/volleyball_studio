// Regression: live pipeline status and log belong to the newest history item,
// and refreshes never destroy or duplicate the streaming log nodes.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import {previewWasmWindow} from "./helpers/preview_wasm.mjs";

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
  latestPipeline:null,historyRows:[],historyFilter:"all",historyOffset:0,historyProject:"aaaabbbbcccc"
};
let project="aaaabbbbcccc";
const getId=id=>id==="studio-analysis-live-host"?host:null;
const storedRuns=new Map();
const sessionStorage={
  getItem:key=>storedRuns.get(key)||null,
  setItem:(key,value)=>storedRuns.set(key,value),
  removeItem:key=>storedRuns.delete(key)
};
const helpers={
  state,ui,projectId:()=>project,mainGraph:null,
  $:getId,make:(tag,klass,text)=>new Node(tag,klass,text),
  clear:node=>node.textContent="",
  historyDuration:()=> "—",
  historyLog:async()=>true,
  window:{...previewWasmWindow(),VolleyballUi:{translate:x=>x},sessionStorage},showGraphResult:()=>{},updatePipelineGraph:()=>{},
  requestAnimationFrame:()=>{}
};
const result=runInNewContext(registry+source.slice(from,to)+
  "\n({renderHistory,updateLiveHistory,liveHistory,recordHistoryRun})",helpers);
const id="a".repeat(32);
const current={
  schema:"volley-studio-pipeline/1",project_id:project,id,
  video:"match.mp4",mode:"post",status:"queued",stage:"Очередь",progress_percent:0
};
state.latestPipeline=current;
result.renderHistory();
assert.equal(ui.historyList.children.length,0,"Queued analysis must not appear in history");
result.recordHistoryRun(project,id);
const secondId="c".repeat(32);
const running={...current,status:"running",queue_id:id,project_queue:{active:true}};
result.updateLiveHistory(running);
assert.equal(ui.historyList.children.length,0);
state.historyRows=[
  {id,project_id:project,mode:"full",video:"match.mp4",status:"complete",stages:[],started_ms:1000},
  {id:"b".repeat(32),project_id:"aaaabbbbcccc",mode:"gpu",video:"older.mp4",status:"complete",stages:[],started_ms:500}
];
result.renderHistory();
assert.equal(ui.historyList.children.length,1,"Active queue must remain hidden even after its first video completes");
assert.equal(ui.historyList.children[0].dataset.runId,"b".repeat(32));
assert.strictEqual(progress.parent,host);
assert.strictEqual(logPanel.parent,host);
const completed={...running,status:"complete",id:secondId,project_queue:{active:false}};
result.updateLiveHistory(completed);
state.historyRows=[{id,project_id:project,mode:"full",status:"complete",started_ms:1000,duration_ms:5000,
  videos:[{id,project_id:project,video:"match.mp4",status:"complete",stages:[]},
    {id:secondId,project_id:project,video:"second.mp4",status:"complete",stages:[]}]}];
result.renderHistory();
assert.equal(ui.historyList.children.length,1,"One archive entry for both videos");
let row=ui.historyList.children[0];
assert.equal(row.dataset.runId,id);
assert.match(row.textContent,/match\.mp4/);
assert.match(row.textContent,/second\.mp4/);
assert.equal(row.open,false);
assert.equal(row.querySelector(".analysis-history-live-stage"),null);
state.historyRows[0].status="failed";
state.historyRows[0].videos[1].status="failed";
state.historyRows[0].videos[1].error="test failure";
result.renderHistory();
assert.match(ui.historyList.children[0].textContent,/test failure/);
assert.equal(ui.historyList.children.length,1);
project="ddddaaaabbbb";
state.historyRows=[];state.historyFilter="all";
result.renderHistory();
assert.strictEqual(progress.parent,host,"Switching projects unmounts live progress");
assert.strictEqual(logPanel.parent,host,"Switching projects unmounts live log");
assert.equal(ui.historyList.children.length,0,"Different project must not display stale run");
const restoredUi={historyList:new Node("div"),progress:new Node("progress"),logPanel:new Node("details")};
const restoredHost=new Node("div");restoredHost.append(restoredUi.progress,restoredUi.logPanel);
const restoredState={
  latestPipeline:null,
  historyRows:[
    {id,project_id:"aaaabbbbcccc",mode:"post",video:"match.mp4",status:"complete",duration_ms:90000,stages:[]},
    {id:"b".repeat(32),project_id:"aaaabbbbcccc",mode:"gpu",video:"older.mp4",status:"complete",stages:[]}
  ],
  historyFilter:"all",historyOffset:0,historyProject:"aaaabbbbcccc"
};
const restored=runInNewContext(registry+source.slice(from,to)+
  "\n({renderHistory,recordHistoryRun})",{
    ...helpers,state:restoredState,ui:restoredUi,
    projectId:()=>"aaaabbbbcccc",
    $:key=>key==="studio-analysis-live-host"?restoredHost:null
  });
restored.renderHistory();
assert.equal(restoredUi.historyList.children.length,2,
  "All archived server runs survive browser-session changes");
assert.equal(restoredUi.historyList.children[0].dataset.runId,id,"Newest server row remains first");
assert.equal(restoredUi.historyList.children[1].dataset.runId,"b".repeat(32),"Older server row remains visible");
const graphSources=[];
const graphState={...restoredState,formProject:"aaaabbbbcccc",latestPipeline:running};
const graphUi={...restoredUi,video:{value:"older.mp4"}};
const graph=runInNewContext(registry+source.slice(from,to)+
  "\n({renderGraphSummary})",{
    ...helpers,state:graphState,ui:graphUi,
    projectId:()=>"aaaabbbbcccc",mainGraph:{querySelector(){}},
    updatePipelineGraph:(_target,value)=>graphSources.push(value)
  });
graph.renderGraphSummary();
assert.equal(graphSources.at(-1).id,running.id,
  "A running analysis must animate even when another video is selected after reload");
graphState.latestPipeline={...running,status:"complete",project_queue:{active:false}};
graph.renderGraphSummary();
assert.equal(graphSources.at(-1).video,"older.mp4",
  "An inactive graph must follow the selected video");
console.log("Studio analysis history: session-only runs, compact timing, reload persistence and project isolation");
