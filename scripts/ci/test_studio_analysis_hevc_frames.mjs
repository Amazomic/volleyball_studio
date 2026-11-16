// Regression: HEVC-linked source remains visible in analysis via exact frame seeking.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {join} from "node:path";
import vm from "node:vm";

const repo=fileURLToPath(new URL("../..",import.meta.url));
const html=readFileSync(join(repo,"rust/volley-app/static/studio.html"),"utf8");
const editor=readFileSync(join(repo,"rust/volley-app/static/studio_editor.js"),"utf8");
const tabs=readFileSync(join(repo,"rust/volley-app/static/studio_tabs.js"),"utf8");
const css=readFileSync(join(repo,"rust/volley-app/static/studio.css"),"utf8");
assert.match(html,/id="studio-analysis-still"/);
assert.match(css,/\.analysis-editor-frame \.analysis-editor-still/);
assert.match(tabs,/recorded\.replace\(\/\\\\\/g,"\/"\)\.split\("\/"\)\.pop\(\)/);

const nodes=new Map();
function element(id=""){
  if(id&&nodes.has(id))return nodes.get(id);
  const listeners=new Map(),classes=new Set();
  const e={
    id,tagName:"DIV",value:"0",textContent:"",disabled:false,
    controls:true,paused:true,hidden:false,children:[],style:{
      setProperty(k,v){this[k]=v;}
    },
    classList:{
      toggle(k,on){if(on)classes.add(k);else classes.delete(k);},
      contains:k=>classes.has(k)
    },
    addEventListener(k,cb){listeners.set(k,[...(listeners.get(k)||[]),cb]);},
    async fire(k,evt={}){for(const cb of listeners.get(k)||[])await cb(evt);},
    setAttribute(){},removeAttribute(){},setPointerCapture(){},
    appendChild(child){this.children.push(child);return child;},
    append(...children){this.children.push(...children);},
    insertBefore(child){this.children.push(child);},
    getBoundingClientRect(){
      if(id==="studio-analysis-frame"){
        const z=Number(element("studio-analysis-stage").style["--editor-zoom"]||1);
        return {left:(960-960*z)/2,top:(540-540*z)/2,width:960*z,height:540*z};
      }
      return {left:0,top:0,width:960,height:540};
    },
    querySelectorAll(){return [];}
  };
  if(id)nodes.set(id,e);
  return e;
}
for(const match of html.matchAll(/id="([^"]+)"/g))element(match[1]);
const video=element("studio-analysis-player");
video.tagName="VIDEO";video.currentTime=0;video.videoWidth=0;video.videoHeight=0;
video.canPlayType=()=>"";video.pause=()=>{video.paused=true;};
video.load=()=>{};
const canvas=element("studio-analysis-court");
canvas.getContext=()=>new Proxy({},{
  get:(_,name)=>name==="measureText"?()=>({width:60}):()=>{}
});
element("studio-analysis-stage").parentElement=element("analysis-media-fixture");
const requests=[],writes=[];
const win={
  devicePixelRatio:1,addEventListener(){},
  ResizeObserver:class {observe(){}},
  VolleyballStudioActivity:{
    async fetch(path,options={}){
      requests.push(path);
      if(path==="/api/video-edit"&&options.method==="POST"){
        const saved=JSON.parse(options.body);writes.push(saved);
        return {ok:true,json:async()=>({...saved,edit_revision:"saved-revision-2"})};
      }
      if(path==="/api/training/examples"&&options.method==="POST")
        return {ok:true,json:async()=>({ok:true})};
      if(path.startsWith("/api/video-edit?"))
        return {ok:true,json:async()=>({
          trim_start_sec:0,trim_end_sec:null,court:[],edit_revision:"revision-1"
        })};
      if(path.startsWith("/api/video-info?"))
        return {ok:true,json:async()=>({
          codec:"hevc",duration_sec:6679.769,width:2160,height:3840
        })};
      throw Error("Unexpected fetch "+path);
    }
  }
};
const storage=new Map();
const documentHandlers=new Map();
const documentMock={
  hidden:false,
  getElementById:element,
  createElement:()=>element(),
  addEventListener:(type,fn)=>documentHandlers.set(type,[...(documentHandlers.get(type)||[]),fn])
};
element("studio-analysis-editor").open=true;
const dispatchSpace=async(target={tagName:"SUMMARY"},extra={})=>{
  let prevented=false,stopped=false;
  const event={
    code:"Space",key:" ",target,repeat:false,altKey:false,ctrlKey:false,metaKey:false,
    preventDefault(){prevented=true;},stopPropagation(){stopped=true;},...extra
  };
  for(const listener of documentHandlers.get("keydown")||[])await listener(event);
  return {prevented,stopped};
};
vm.runInNewContext(editor,{
  window:win,document:documentMock,
  sessionStorage:{getItem:k=>storage.get("session:"+k)||null,setItem:(k,v)=>storage.set("session:"+k,v)},
  localStorage:{getItem:k=>storage.get("local:"+k)||null,setItem:(k,v)=>storage.set("local:"+k,v),removeItem:k=>storage.delete("local:"+k)},
  URLSearchParams,setTimeout,clearTimeout,setInterval,clearInterval,
  ResizeObserver:win.ResizeObserver,console
});
const source="linked_47983479be9b4e6b86d57005b73aca8e.mp4";
await win.VolleyballEditor.load("6fd50b60022d",source);
await new Promise(resolve=>setTimeout(resolve,160));
const image=element("studio-analysis-still");
assert.equal(image.hidden,false);
assert.equal(video.hidden,true);
assert.match(image.src,/\/api\/video-preview\?/);
assert.ok(image.src.includes("width=960"));
assert.ok(image.src.includes("time=0.000"));
assert.equal(element("studio-analysis-toggle-play").disabled,false);
assert.equal(element("studio-analysis-seek").max,"6679769");
assert.equal(element("studio-analysis-trim-end").value,"6679.77");
assert.equal(element("analysis-media-fixture").style["--frame-ratio"],"0.5625");

await image.fire("load");
await element("studio-analysis-court-mode").fire("click");
const stage=element("studio-analysis-stage");
// A first touch may begin adding a corner; a second touch must cancel that
// pending single-finger edit and turn the gesture into an image-only pinch.
let pinchPrevented=0,pinchStopped=0;
const touch=(pointerId,x,y)=>({pointerId,pointerType:"touch",clientX:x,clientY:y,
  preventDefault(){pinchPrevented++;},stopPropagation(){pinchStopped++;}});
await stage.fire("pointerdown",touch(51,390,200));
await canvas.fire("pointerdown",touch(51,390,200));
await stage.fire("pointerdown",touch(52,590,200));
await stage.fire("pointermove",touch(52,690,200));
assert.ok(Number(element("studio-analysis-zoom-label").textContent.replace("%",""))>84);
await canvas.fire("pointermove",touch(51,420,220)); // No corner motion mid-pinch.
await stage.fire("pointermove",touch(52,590,200));
assert.equal(element("studio-analysis-zoom-label").textContent,"84%");
await stage.fire("pointerup",touch(52,590,200));
await stage.fire("pointerup",touch(51,390,200));
assert.ok(pinchPrevented>0&&pinchStopped>0);
await canvas.fire("pointerdown",{clientX:270,clientY:220,pointerId:1});
await canvas.fire("pointerup",{pointerId:1});
element("studio-analysis-seek").value="31700";
await element("studio-analysis-seek").fire("input");
await element("studio-analysis-set-start").fire("click");
assert.equal(element("studio-analysis-trim-start").value,"31.70");
await new Promise(resolve=>setTimeout(resolve,160));
assert.match(image.src,/time=31.700/);
assert.match(element("studio-analysis-edit-status").textContent,/несохранённые изменения|кадр/);
assert.equal(requests.filter(s=>s.startsWith("/api/video-edit?")).length,1);
assert.equal(requests.filter(s=>s.startsWith("/api/video-info?")).length,1);
console.log("PASS: HEVC frame fallback loads original frames and supports time selection/markup");

// Four corners form a field that can be dragged from anywhere inside the polygon.
const place=async (x,y,id=1)=>{
  await canvas.fire("pointerdown",{clientX:x,clientY:y,pointerId:id});
  await canvas.fire("pointerup",{clientX:x,clientY:y,pointerId:id});
};
for(const [x,y] of [[670,220],[670,450],[270,450]])await place(x,y);
await element("studio-analysis-net-mode").fire("click");
await place(340,270);
await place(580,270);
const draft=()=>{
  const entry=[...storage.entries()].find(([key])=>key.startsWith("local:volley:editor-draft:v1:"));
  assert.ok(entry,"Court changes should be saved as draft metadata");
  return JSON.parse(entry[1]).value;
};
const flush=()=>new Promise(resolve=>setTimeout(resolve,290));
const close=(a,b)=>Math.abs(a-b)<0.00001;
await flush();
const before=draft();
assert.equal(before.court.length,4);
assert.equal(before.net.length,2);
await canvas.fire("pointerdown",{clientX:470,clientY:335,pointerId:7});
await canvas.fire("pointermove",{clientX:524,clientY:371,pointerId:7});
await canvas.fire("pointerup",{clientX:524,clientY:371,pointerId:7});
await flush();
const moved=draft();
const dx=54/(960*.84),dy=36/(540*.84);
for(let i=0;i<4;i++){
  assert.ok(close(moved.court[i][0]-before.court[i][0],dx),"all four corners shift equally in X");
  assert.ok(close(moved.court[i][1]-before.court[i][1],dy),"all four corners shift equally in Y");
}
for(let i=0;i<2;i++){
  assert.ok(close(moved.net[i][0]-before.net[i][0],dx),"net stays aligned with court in X");
  assert.ok(close(moved.net[i][1]-before.net[i][1],dy),"net stays aligned with court in Y");
}
// Outside clicks cannot reposition a completed field.
await canvas.fire("pointerdown",{clientX:900,clientY:500,pointerId:8});
await canvas.fire("pointermove",{clientX:920,clientY:520,pointerId:8});
await canvas.fire("pointerup",{clientX:920,clientY:520,pointerId:8});
await flush();
assert.deepEqual(JSON.stringify(draft().court),JSON.stringify(moved.court));
// Existing individual-corner and net-point editing retain precedence.
await canvas.fire("pointerdown",{clientX:324,clientY:256,pointerId:9});
await canvas.fire("pointermove",{clientX:335,clientY:260,pointerId:9});
await canvas.fire("pointerup",{clientX:335,clientY:260,pointerId:9});
await flush();
const cornerOnly=draft();
assert.ok(!close(cornerOnly.court[0][0],moved.court[0][0]));
for(let i=1;i<4;i++)assert.deepEqual(JSON.stringify(cornerOnly.court[i]),JSON.stringify(moved.court[i]));
assert.deepEqual(JSON.stringify(cornerOnly.net),JSON.stringify(moved.net));
// Net endpoints remain separately draggable without moving the court.
await canvas.fire("pointerdown",{clientX:394,clientY:306,pointerId:11});
await canvas.fire("pointermove",{clientX:409,clientY:310,pointerId:11});
await canvas.fire("pointerup",{clientX:409,clientY:310,pointerId:11});
await flush();
const netOnly=draft();
assert.ok(!close(netOnly.net[0][0],cornerOnly.net[0][0]));
assert.deepEqual(JSON.stringify(netOnly.net[1]),JSON.stringify(cornerOnly.net[1]));
assert.deepEqual(JSON.stringify(netOnly.court),JSON.stringify(cornerOnly.court));
// A large translation is clamped as a single unit, without distorting the polygon.
await canvas.fire("pointerdown",{clientX:524,clientY:371,pointerId:10});
await canvas.fire("pointermove",{clientX:4000,clientY:371,pointerId:10});
await canvas.fire("pointerup",{clientX:4000,clientY:371,pointerId:10});
await flush();
const limited=draft();
const limitedDx=limited.court[0][0]-netOnly.court[0][0];
for(let i=1;i<4;i++)assert.ok(close(limited.court[i][0]-netOnly.court[i][0],limitedDx));
for(const p of [...limited.court,...limited.net])
  assert.ok(p.every(v=>v>=-.5-1e-8&&v<=1.5+1e-8),"Translated field stays within backend bounds");

// Zoom-out must allow significantly more context than the previous 52% floor.
for(let i=0;i<20;i++)await element("studio-analysis-zoom-out").fire("click");
assert.equal(element("studio-analysis-zoom-label").textContent,"10%");
assert.equal(element("studio-analysis-zoom-out").disabled,true);
await element("studio-analysis-zoom-in").fire("click");
assert.equal(element("studio-analysis-zoom-label").textContent,"18%");
await element("studio-analysis-edit-save").fire("click");
assert.equal(writes.length,1,"Field translation must be saved through the existing video-edit API");
assert.equal(writes[0].court.length,4);
assert.equal(writes[0].net.length,2);
assert.deepEqual(JSON.stringify(writes[0].court),JSON.stringify(limited.court));
assert.deepEqual(JSON.stringify(writes[0].net),JSON.stringify(limited.net));
console.log("PASS: entire field/net drag, separate point edits, bounded translation, saved coordinates, 10% zoom");

// Mouse wheels and trackpad two-finger swipes act only over the video stage.
const zoomNow=()=>Number(element("studio-analysis-zoom-label").textContent.replace("%",""));
let wheelCancelled=0;
const wheel=(deltaY,more={})=>({
  deltaY,deltaMode:0,target:{closest:()=>null},
  preventDefault(){wheelCancelled++;},...more
});
await stage.fire("wheel",wheel(-100)); // mouse wheel up
assert.ok(zoomNow()>18);
await stage.fire("wheel",wheel(90,{ctrlKey:true})); // trackpad pinch / two-finger scroll
assert.ok(zoomNow()<23);
assert.equal(wheelCancelled,2);
const toolbarZoom=zoomNow();
await stage.fire("wheel",wheel(-120,{target:{closest:()=>({})}}));
assert.equal(zoomNow(),toolbarZoom,"Zooming over toolbar must leave buttons unaffected");
assert.equal(wheelCancelled,2);
for(let i=0;i<20;i++)await stage.fire("wheel",wheel(150));
assert.equal(zoomNow(),10,"Wheel-out cannot pass the 10% limit");
for(let i=0;i<30;i++)await stage.fire("wheel",wheel(-150));
assert.equal(zoomNow(),124,"Wheel-in cannot pass the 124% limit");
assert.deepEqual(JSON.stringify(writes[0].court),JSON.stringify(limited.court),
  "Zoom gestures must not alter saved court coordinates");
console.log("PASS: mouse wheel, two-finger trackpad gesture and touch pinch zoom without moving court points");



// With focus on <summary>, Space is consumed and must not collapse the editor.
const control=element("studio-analysis-toggle-play");
const summaryKey=await dispatchSpace();
assert.equal(summaryKey.prevented,true);
assert.equal(summaryKey.stopped,true);
assert.equal(element("studio-analysis-editor").open,true);
assert.equal(control.textContent,"Пауза","Space must start HEVC frame playback");
await new Promise(resolve=>setTimeout(resolve,220));
const inFlight=image.src;
await image.fire("load");
await new Promise(resolve=>setTimeout(resolve,160));
assert.notEqual(control.textContent,"▶ Воспроизвести");
assert.notEqual(image.src,inFlight,"HEVC frame playback should request a newer timestamp");
const repeat=await dispatchSpace({tagName:"SUMMARY"},{repeat:true});
assert.equal(repeat.prevented,true);
assert.equal(control.textContent,"Пауза","Key repeat must not toggle playback repeatedly");
const pauseKey=await dispatchSpace();
assert.equal(pauseKey.prevented,true);
assert.equal(control.textContent,"▶ Воспроизвести","Second Space must pause HEVC frame playback");
await image.fire("load");

const ignoredInput=await dispatchSpace({tagName:"INPUT"});
assert.equal(ignoredInput.prevented,false,"Text inputs keep their native keyboard behavior");

// The same shortcut controls native HTML video for supported codecs.
video.canPlayType=()=> "probably";
let started=0,paused=0;
video.play=()=>{started++;video.paused=false;return Promise.resolve();};
video.pause=()=>{paused++;video.paused=true;};
await win.VolleyballEditor.load("6fd50b60022d","different.mp4");
const nativeStart=await dispatchSpace();
assert.equal(nativeStart.prevented,true);
assert.equal(started,1);
await dispatchSpace();
assert.equal(paused,2,"Native video must pause on the second Space (once during load)");
assert.equal(element("studio-analysis-editor").open,true);
const beforeWheel=wheelCancelled;
await stage.fire("wheel",wheel(-120));
assert.equal(wheelCancelled,beforeWheel,"Outside marking, the page keeps its normal wheel scrolling");
assert.equal(zoomNow(),100,"Inactive calibration view remains at 100%");
console.log("PASS: Space starts/pauses HEVC frame playback and native video without collapsing the editor");
