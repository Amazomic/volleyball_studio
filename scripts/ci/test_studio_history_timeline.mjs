// Timeline regression: a running stage has a live duration and ETA,
// finished stages retain exact durations, and live log nodes are never cloned.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
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
const ui={historyList:new Node("div"),progress,logPanel};
const id="a".repeat(32),now=Date.now();
const stages=[
  {name:"GPU · VballNet",started_ms:now-90000,finished_ms:now-40000,duration_ms:50000},
  {name:"GPU · RF-DETR Medium + RTMPose",started_ms:now-40000,finished_ms:null,duration_ms:null,
    frames_processed:120,frames_total:400,progress_percent:30}
];
const current={
  schema:"volley-studio-pipeline/1",project_id:"project",id,
  mode:"gpu",video:"game.mp4",stage:"GPU · RF-DETR Medium + RTMPose: running",
  status:"running",progress_percent:67.5,elapsed_ms:91000,remaining_ms:110000,stages
};
const state={latestPipeline:current,historyRows:[],historyFilter:"all",historyStatusAt:now,historyOffset:0};
const {renderHistory,updateLiveHistory}=runInNewContext(source.slice(from,to)+
  "\n({renderHistory,updateLiveHistory})",{
    state,ui,projectId:()=>"project", $:key=>key==="studio-analysis-live-host"?host:null,
    make:(tag,klass,text)=>new Node(tag,klass,text),
    clear:node=>node.textContent="",historyDuration:ms=>ms===null?"—":Math.floor(ms/1000)+" с",
    historyLog:async()=>true
  });
renderHistory();
const run=ui.historyList.children[0];
assert.equal(run.querySelector(".analysis-history-stages").children.length,2);
assert.match(run.querySelector(".analysis-history-stages").textContent,/GPU · VballNet.*50 с/);
assert.match(run.querySelector(".analysis-history-stages").textContent,/В процессе/);
assert.match(run.querySelector(".analysis-history-stages").textContent,/120 \/ 400 кадров · 30.0%/);
const rail=run.querySelector(".analysis-history-stage-rail");
assert.equal(rail?.attributes["aria-valuenow"],"30");
assert.equal(rail?.children[0].style.width,"30%");
assert.match(run.querySelector(".analysis-history-eta").textContent,/Осталось ≈ 10[5-9] с|Осталось ≈ 110 с/);
assert.match(run.querySelector(".analysis-history-runtime").textContent,/Выполняется · 91 с/);
assert.equal(progress.parent,logPanel.parent);

const finished={...current,status:"complete",progress_percent:100,remaining_ms:null,
  stages:[stages[0],{...stages[1],finished_ms:now,duration_ms:40000}]};
updateLiveHistory(finished);
assert.equal(ui.historyList.children[0],run);
assert.match(run.querySelector(".analysis-history-runtime").textContent,/Завершён ·/);
assert.doesNotMatch(run.querySelector(".analysis-history-stages").textContent,/В процессе/);
assert.match(run.querySelector(".analysis-history-stages").textContent,/40 с/);
assert.strictEqual(progress.parent,logPanel.parent);
console.log("History timeline: active and completed stages, elapsed and ETA, stable live log");
