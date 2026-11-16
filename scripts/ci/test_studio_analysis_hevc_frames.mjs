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
const requests=[];
const win={
  devicePixelRatio:1,addEventListener(){},
  ResizeObserver:class {observe(){}},
  VolleyballStudioActivity:{
    async fetch(path){
      requests.push(path);
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
vm.runInNewContext(editor,{
  window:win,document:{
    getElementById:element,
    createElement:()=>element()
  },
  sessionStorage:{getItem:k=>storage.get("session:"+k)||null,setItem:(k,v)=>storage.set("session:"+k,v)},
  localStorage:{getItem:k=>storage.get("local:"+k)||null,setItem:(k,v)=>storage.set("local:"+k,v),removeItem:k=>storage.delete("local:"+k)},
  URLSearchParams,setTimeout,clearTimeout,ResizeObserver:win.ResizeObserver,console
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
assert.equal(element("studio-analysis-toggle-play").disabled,true);
assert.equal(element("studio-analysis-seek").max,"6679769");
assert.equal(element("studio-analysis-trim-end").value,"6679.77");
assert.equal(element("analysis-media-fixture").style["--frame-ratio"],"0.5625");

await image.fire("load");
await element("studio-analysis-court-mode").fire("click");
await canvas.fire("pointerdown",{clientX:270,clientY:220,pointerId:1});
await canvas.fire("pointerup",{pointerId:1});
element("studio-analysis-seek").value="31700";
await element("studio-analysis-seek").fire("input");
await element("studio-analysis-set-start").fire("click");
assert.equal(element("studio-analysis-trim-start").value,"31.70");
await new Promise(resolve=>setTimeout(resolve,160));
assert.match(image.src,/time=31.700/);
assert.match(element("studio-analysis-edit-status").textContent,/несохранённые изменения|покадров/);
assert.equal(requests.filter(s=>s.startsWith("/api/video-edit?")).length,1);
assert.equal(requests.filter(s=>s.startsWith("/api/video-info?")).length,1);
console.log("PASS: HEVC frame fallback loads original frames and supports time selection/markup");
