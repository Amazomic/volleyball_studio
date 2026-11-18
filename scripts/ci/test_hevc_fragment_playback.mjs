import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const code = fs.readFileSync(
  new URL("../../rust/volley-app/static/studio_fragments.js", import.meta.url), "utf8");
const fetched = [];
const source = "/api/video-media?project_id=project123&video=example.mp4";
let currentSrc = source;
const assigned = [];
const listeners = new Map();
const video = {
  tagName: "VIDEO", paused: true, currentTime: 0, buffered: {length: 0},
  getAttribute: name => name === "src" ? currentSrc : null,
  set src(value) {currentSrc = value;assigned.push(value);},
  removeAttribute(name) {if(name==="src")currentSrc="";},
  pause() {this.paused=true;},
  get src() {return currentSrc;},
  load() {},
  addEventListener(name,fn) {listeners.set(name,fn);},
  removeEventListener(name) {listeners.delete(name);},
  play() {return Promise.resolve();}
};
class MockBuffer {
  constructor() {this.buffered = {length: 0};this.handlers = new Map();this.appended=0;}
  addEventListener(name,handler) {this.handlers.set(name,handler);}
  removeEventListener(name) {this.handlers.delete(name);}
  appendBuffer(bytes) {
    this.appended+=bytes.byteLength;
    const end=this.timestampOffset+18.021333;
    video.buffered={length:1,start:()=>0,end:()=>end};
    queueMicrotask(()=>this.handlers.get("updateend")?.());
  }
}
class MockMediaSource {
  constructor() {this.readyState="closed";MockMediaSource.last=this;this.handlers=new Map();}
  static isTypeSupported(mime){return mime.includes("avc1.42C01F");}
  addEventListener(name,handler){this.handlers.set(name,handler);}
  addSourceBuffer(){this.buffer=new MockBuffer();return this.buffer;}
}
class MockObserver {
  constructor(callback){this.callback=callback;}
  observe(){}
}
class MockUrl extends URL {
  static createObjectURL(){return "blob:volley-test";}
  static revokeObjectURL(){}
}
const windowRef={MediaSource:MockMediaSource,dispatchEvent(){}};
const fetch = async path => {
  fetched.push(String(path));
  if(String(path).startsWith("/api/video-info"))
    return {ok:true,json:async()=>({codec:"hevc",duration_sec:50})};
  if(String(path).startsWith("/api/video-fragment"))
    return {ok:true,arrayBuffer:async()=>new ArrayBuffer(1500)};
  throw Error("Unexpected API: "+path);
};
vm.runInNewContext(code,{
  URL:MockUrl,URLSearchParams,MediaSource:MockMediaSource,MutationObserver:MockObserver,
  document:{documentElement:{},querySelectorAll:()=>[video],addEventListener(){}},
  location:{href:"http://127.0.0.1:8875/studio",origin:"http://127.0.0.1:8875"},
  window:windowRef,
  CustomEvent:class {constructor(name,options){this.name=name;this.detail=options.detail;}},
  fetch,console,Number,Promise,Map,WeakMap,WeakSet,Set
});
for(let i=0;i<5;i++)await Promise.resolve();
assert.equal(currentSrc,"blob:volley-test","HEVC media must switch to browser-compatible MSE");
const sourceBuffer=MockMediaSource.last;
sourceBuffer.readyState="open";
await sourceBuffer.handlers.get("sourceopen")();
for(let i=0;i<6;i++)await Promise.resolve();
assert(fetched.some(path=>path.startsWith("/api/video-fragment?") &&
  path.includes("project_id=project123") && path.includes("video=example.mp4") &&
  path.includes("index=0")),"first segment must be fetched for the correct project");
assert(sourceBuffer.buffer.appended>=1500,"fMP4 bytes must reach the SourceBuffer");
assert.equal(typeof windowRef.VolleyballBrowserVideo?.load,"function",
  "explicit preflight loader must be exposed");
assert.equal(await windowRef.VolleyballBrowserVideo.canStream("project123","example.mp4"),
  true,"HEVC must be eligible for short-fragment streaming");
video.currentTime=7;
listeners.get("timeupdate")();
for(let i=0;i<20;i++)await Promise.resolve();
assert(fetched.some(path=>path.startsWith("/api/video-fragment?")&&
  new URL(path,"http://localhost").searchParams.get("index")==="1"),
  "AAC padding beyond 18 seconds must not skip adjacent fragment 1");
assert(!fetched.some(path=>path.startsWith("/api/video-fragment?")&&
  new URL(path,"http://localhost").searchParams.get("index")==="2"),
  "fragment 2 must not replace missing fragment 1");
video.currentTime=25;
listeners.get("timeupdate")();
for(let i=0;i<20;i++)await Promise.resolve();
assert(fetched.some(path=>path.startsWith("/api/video-fragment?")&&
  new URL(path,"http://localhost").searchParams.get("index")==="2"),
  "continuous playback must advance to fragment 2 after fragment 1");
const count=assigned.length;
await windowRef.VolleyballBrowserVideo.load(video,"project123","example.mp4",{seek:21});
assert(currentSrc.startsWith("blob:"),"HEVC preflight must attach MSE directly");
assert(assigned.slice(count).every(src=>src.startsWith("blob:")),
  "HEVC preflight must NEVER assign the unsupported original video URL");
console.log("HEVC MSE preview: preflight avoids original, project-scoped fragment appended");
