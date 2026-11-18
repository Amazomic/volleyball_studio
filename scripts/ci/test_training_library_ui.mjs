import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const start=js.indexOf("  const trainingNames=");
const end=js.indexOf("  // Source-scoped human truth.",start);
assert.ok(start>=0&&end>start);

class Node {
  constructor(tag="div"){
    this.tagName=tag.toUpperCase();this.children=[];this.dataset={};
    this.textContent="";this.className="";this.listeners=new Map();
    this.value="";this.disabled=false;this.attributes=new Map();
    this.classList={
      add:k=>{if(!this.className.split(" ").includes(k))this.className=(this.className+" "+k).trim();},
      remove:k=>{this.className=this.className.split(" ").filter(v=>v!==k).join(" ");},
      toggle:(k,forced)=>{const active=forced??!this.className.split(" ").includes(k);
        if(active)this.classList.add(k);else this.classList.remove(k);}
    };
  }
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,callback){this.listeners.set(name,callback);}
  setAttribute(name,value){this.attributes.set(name,value);}
  querySelectorAll(selector){
    const klass=selector.startsWith(".")?selector.slice(1):selector;
    const list=[];
    for(const child of this.children){
      if(child.className.split(" ").includes(klass))list.push(child);
      list.push(...child.querySelectorAll(selector));
    }
    return list;
  }
}
let revision=8;
const item={id:"f3edb22d-516c-4d2c-9911-de0a6b5a13ef",type:"touch",
  project_id:"project-01",video:"game.mp4",rally_run_id:"run-01",
  frame_index:90,track_id:70,label:"attack"};
const training={
  list:new Node(),count:new Node(),status:new Node(),newExample:new Node(),
  save:new Node(),label:new Node(),revision:0,epoch:0,selectedId:null,editId:null,
  track:null,frame:null,viewBusy:false
};
const s={project:"project-01",video:"game.mp4",run:"run-01",seekAt:0,
  chain:{runToAnalysis:new Map([["run-01","gpu-01"]])}};
const ui={video:{readyState:4,currentTime:0,pause(){}},analysis:{value:"gpu-01"}};
const timeline={data:{}};
const requests=[],writes=[],seeks=[],prefetched=[];
const context={
  training,s,ui,timeline,document:{createElement:tag=>new Node(tag)},
  window:{confirm:()=>true},fps:()=>30,url:(path)=>path,
  request:async path=>{requests.push(path);return {
    schema:"volley-human-training-library/1",revision,examples:[{...item}]};},
  post:async(path,payload)=>{writes.push(payload);if(payload.action==="relabel")item.label=payload.label;
    revision++;return {ok:true,revision};},
  projects:async()=>{throw Error("Same-project sample must not reload projects");},
  setAnalysis:async()=>{throw Error("Already-selected CPU source must not reload");},
  loadTimeline:async()=>{},
  ensurePreviewVideo:()=>{},
  seekTimeline:second=>{seeks.push(second);ui.video.currentTime=second;},
  prefetchOverlayAtFrame:frame=>prefetched.push(frame),
  trainingMessage:message=>{training.status.textContent=message}
};
const {refreshTraining,openTrainingExample}=
  runInNewContext(js.slice(start,end)+"\n({refreshTraining,openTrainingExample})",context);
await refreshTraining();
assert.equal(training.count.textContent,"1");
const row=training.list.children[0];
assert.ok(row.children[0].textContent.includes("game.mp4"));
const actions=row.children[1],editor=row.children[2];
assert.deepEqual(actions.children.map(e=>e.textContent),
  ["Просмотреть","Изменить","Удалить"]);
actions.children[1].listeners.get("click")();
assert.equal(editor.className.includes("hidden"),false);
editor.children[0].value="block";
await editor.children[1].listeners.get("click")();
assert.equal(writes.length,1);
assert.deepEqual(JSON.parse(JSON.stringify(writes[0])),{
  action:"relabel",project_id:"project-01",id:item.id,label:"block",revision:8
});
assert.equal(training.revision,9);
await openTrainingExample(item);
assert.equal(training.selectedId,item.id);
assert.equal(training.editId,item.id);
assert.equal(training.label.value,"block");
assert.equal(training.save.textContent,"Перезаписать кадр и игрока");
assert.equal(s.seekAt,3);
assert.deepEqual(seeks,[3]);
assert.deepEqual(prefetched,[90]);
console.log("Training library: list, relabel with CAS, view exact source/frame, recapture mode: OK");
