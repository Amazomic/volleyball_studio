import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {join} from "node:path";
import {fileURLToPath} from "node:url";
import vm from "node:vm";
const root=fileURLToPath(new URL("../..",import.meta.url));
const src=readFileSync(join(root,"rust/volley-app/static/studio_tasks.js"),"utf8");
const html=readFileSync(join(root,"rust/volley-app/static/studio.html"),"utf8");
const css=readFileSync(join(root,"rust/volley-app/static/studio.css"),"utf8");
const web=readFileSync(join(root,"rust/volley-app/src/studio_web.rs"),"utf8");
const main=readFileSync(join(root,"rust/volley-app/src/main.rs"),"utf8");
const tabs=readFileSync(join(root,"rust/volley-app/static/studio_tabs.js"),"utf8");
const quick=readFileSync(join(root,"rust/volley-app/static/studio_quick_reel.js"),"utf8");
const montageBuilder=readFileSync(join(root,"rust/volley-app/static/studio_montage_builder.js"),"utf8");
const studio=readFileSync(join(root,"rust/volley-app/static/studio.js"),"utf8");
const report=readFileSync(join(root,"rust/volley-app/static/studio_stats.js"),"utf8");
const players=readFileSync(join(root,"rust/volley-app/static/studio_tracks.js"),"utf8");
const montageServer=readFileSync(join(root,"rust/volley-app/src/montage.rs"),"utf8");

assert.match(html,/<section id="studio-sidebar-tasks"[^>]*class="[^"]*hidden"/);
assert.match(html,/<div id="studio-sidebar-task-list"/);
assert.match(html,/\/studio\/tasks\.js/);
assert.match(css,/\.studio-sidebar-tasks\{margin-top:auto;flex:0 0 auto/);
assert.match(css,/@media\(max-width:650px\)[\s\S]*?\.studio-sidebar-tasks/);
assert.match(main,/\.route\("\/studio\/tasks\.js", get\(studio_web::tasks_js\)\)/);
assert.match(web,/static\/studio_tasks\.js/);
assert.match(tabs,/VolleyballTasks\?\.set\("analysis"/);
assert.match(tabs,/VolleyballTasks\?\.clear\("analysis"\)/);
assert.match(quick,/VolleyballTasks\?\.set\("quick"/);
assert.match(players,/VolleyballTasks\?\.set\("track-grouping"/);
assert.match(studio,/VolleyballTasks\?\.set\("training-annotation"/);
assert.match(studio,/VolleyballTasks\?\.clear\("training-annotation"/);
assert.match(studio,/VolleyballTasks\?\.set\("training-save"/);
assert.match(studio,/VolleyballTasks\?\.clear\("training-save"/);
assert.match(report,/VolleyballTasks\?\.set\("report-recalculate"/);
assert.match(report,/VolleyballTasks\?\.clear\("report-recalculate"/);
assert.match(montageBuilder,/VolleyballTasks\?\.set\("montage"/);
assert.match(studio,/VolleyballTasks\?\.set\("montage"/);
assert.match(montageServer,/"overall_percent":self\.overall_percent/);

class Node {
  constructor(){
    this.children=[];this.attributes={};this.style={};this.classNames=new Set(["hidden"]);
    this.classList={
      toggle:(name,on)=>on?this.classNames.add(name):this.classNames.delete(name),
      contains:name=>this.classNames.has(name)
    };
  }
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=[...children];}
  setAttribute(k,v){this.attributes[k]=String(v);}
  removeAttribute(k){delete this.attributes[k];}
}
const container=new Node(),list=new Node(),project=new Node();
project.value="project-a";
const nodes=new Map([["studio-sidebar-tasks",container],
  ["studio-sidebar-task-list",list],["studio-project",project]]);
const listeners=new Map();
for(const el of nodes.values())el.addEventListener=(name,fn)=>listeners.set(name,fn);
let gpu={schema:"volley-native-active-gpu/1",project_id:"project-a",active:false};
let pipeline={schema:"volley-studio-pipeline/1",project_id:"project-a",status:"idle"};
let reel={status:"idle"};
let montage={state:"idle"};
let failMontage=false;
let failQuick=false;
let preparation={project_id:"project-a",videos:[]};
let failPreparation=false;
let quickGate=null;
const hits=[];
const doc={hidden:false,getElementById:id=>nodes.get(id),createElement:()=>new Node(),
  addEventListener(){}};
const win={addEventListener:(name,fn)=>listeners.set(name,fn),VolleyballStudioActivity:{
  async fetch(url){
    hits.push(url);
    if(url.startsWith("/api/studio/pipeline/status")){
      return {ok:true,json:async()=>pipeline};
    }
    if(url==="/api/montage/native/status"){
      if(failMontage)throw Error("network");
      return {ok:true,json:async()=>montage};
    }
    if(url.startsWith("/api/quick-reel/")){
      if(quickGate)await quickGate;
      if(failQuick)throw Error("network");
      return {ok:true,json:async()=>reel};
    }
    if(url.startsWith("/api/project-preparation")){
      if(failPreparation)throw Error("network");
      return {ok:true,json:async()=>preparation};
    }
    return {ok:true,json:async()=>gpu};
  }
}};
let interval=null,timeout=null;
vm.runInNewContext(src,{document:doc,window:win,URLSearchParams,
  setInterval:fn=>{interval=fn;return 1;},
  setTimeout:fn=>{timeout=fn;return 2;},clearTimeout:()=>{}},{filename:"studio_tasks.js"});
assert.ok(timeout,"active-job card must refresh periodically");
// The module starts its first probe asynchronously at initialization.
await new Promise(resolve=>setImmediate(resolve));
await win.VolleyballTasks.refresh();
assert.ok(container.classList.contains("hidden"),"idle card must be hidden");
assert.ok(hits.some(s=>s.includes("project_id=project-a")));
assert.ok(hits.includes("/api/montage/native/status"),"montage status must be restored after reload");

gpu={schema:"volley-native-active-gpu/1",project_id:"project-a",
  active:true,model:"players",phase:"running",progress_percent:35.6};
reel={status:"running",stage:"Монтаж",progress:88,stage_index:4};
await win.VolleyballTasks.refresh();
assert.equal(list.children.length,2);
assert.ok(!container.classList.contains("hidden"));
assert.equal(list.children[0].children[0].children[0].textContent,"Анализ GPU · "+(win.VolleyballPersonDetector?.label||"RF-DETR Medium + RTMPose"));
assert.equal(list.children[0].children[0].children[1].textContent,"36%");
assert.equal(list.children[1].children[0].children[1].textContent,"88%");

win.VolleyballTasks.set("cpu",{label:"Постобработка CPU",stage:"Касания",progress:null});
assert.equal(list.children.length,3);
const cpu=list.children[1];
assert.ok(cpu.children[2].classList.contains("is-indeterminate"));
assert.equal(cpu.children[2].attributes["aria-valuenow"],undefined);
assert.equal(cpu.children[1].textContent,"Касания");
win.VolleyballTasks.clear("cpu");
assert.equal(list.children.length,2);

montage={id:"montage_123",project_id:"project-a",state:"running",stage:"Склейка",overall_percent:58.7};
await win.VolleyballTasks.refresh();
assert.equal(list.children.length,3,"montage must appear alongside GPU and Quick Reel");
assert.equal(list.children[2].children[0].children[0].textContent,"Монтаж");
assert.equal(list.children[2].children[0].children[1].textContent,"59%");
assert.equal(list.children[2].children[1].textContent,"Склейка");
failMontage=true;
await win.VolleyballTasks.refresh();
assert.equal(list.children.length,3,"temporary montage status failure must keep active progress");
failMontage=false;
montage={id:"montage_123",state:"complete",overall_percent:100};
await win.VolleyballTasks.refresh();
assert.equal(list.children.length,2,"completed montage must disappear");
failQuick=true;
await win.VolleyballTasks.refresh();
assert.equal(list.children.length,2,"network error must not falsely clear running job");
failQuick=false;
gpu={schema:"volley-native-active-gpu/1",project_id:"project-a",active:false};
reel={status:"complete",progress:100};
await win.VolleyballTasks.refresh();
assert.ok(container.classList.contains("hidden"),"after completion card should disappear");
assert.equal(list.children.length,0);
preparation={project_id:"project-a",videos:[
  {video:"match.mp4",state:"preparing",progress:37.4},
  {video:"next.mp4",state:"queued",progress:0},
  {video:"ready.mp4",state:"ready",progress:100}
]};
await win.VolleyballTasks.refresh();
assert.equal(list.children.length,2,"preparation is restored from server status on refresh");
assert.equal(list.children[0].children[0].children[0].textContent,"Подготовка видео");
assert.equal(list.children[0].children[0].children[1].textContent,"37%");
assert.equal(list.children[0].children[1].textContent,"match.mp4 · Подготовка видео");
assert.equal(list.children[1].children[1].textContent,"next.mp4 · В очереди");
project.value="";
preparation.videos[0].progress=62;
await win.VolleyballTasks.refresh();
assert.equal(list.children[0].children[0].children[1].textContent,"62%","switching projects retains active preparation");
failPreparation=true;
await win.VolleyballTasks.refresh();
assert.equal(list.children.length,2,"network errors retain preparation");
failPreparation=false;
preparation.videos[0].state="ready";
preparation.videos[1].state="cancelled";
await win.VolleyballTasks.refresh();
assert.ok(container.classList.contains("hidden"),"completed and cancelled preparation disappears");
let releaseQuick;
quickGate=new Promise(resolve=>{releaseQuick=resolve;});
const polling=win.VolleyballTasks.refresh();
await new Promise(resolve=>setImmediate(resolve));
project.value="new-project";
listeners.get("volley:project-ready")();
project.value="other-project";
listeners.get("change")();
releaseQuick();
quickGate=null;
await polling;
await new Promise(resolve=>setImmediate(resolve));
assert.ok(hits.some(url=>url.startsWith("/api/project-preparation?")&&url.includes("project_id=new-project")),
  "project created during a poll must be observed even after immediate project switch");
project.value="project-a";
pipeline={schema:"volley-studio-pipeline/1",project_id:"project-a",status:"running",
  stage:"GPU · VBallNet",progress_percent:21.3,
  stages:[{name:"GPU · VBallNet",frames_processed:120,frames_total:900}]};
await win.VolleyballTasks.refresh();
assert.equal(list.children.length,1);
assert.match(list.children[0].children[1].textContent,/120\/900 кадров \(13\.3%\)/);
assert.equal(list.children[0].children[0].children[1].textContent,"21%");
pipeline.stages[0].frames_processed=450;
pipeline.progress_percent=32;
await win.VolleyballTasks.refresh();
assert.match(list.children[0].children[1].textContent,/450\/900 кадров \(50\.0%\)/);
pipeline.status="complete";
await win.VolleyballTasks.refresh();
assert.equal(list.children.length,0);
console.log("PASS: sidebar frame progress for analysis stages, GPU, Quick Reel and montage");
