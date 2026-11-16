import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const code = fs.readFileSync(
  new URL("../../rust/volley-app/static/studio_fragments.js", import.meta.url), "utf8");
const fetched = [];
const source = "/api/video-media?project_id=project123&video=example.mp4";
let currentSrc = source;
const listeners = new Map();
const video = {
  tagName: "VIDEO", paused: true, currentTime: 0, buffered: {length: 0},
  getAttribute: name => name === "src" ? currentSrc : null,
  set src(value) {currentSrc = value;},
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
  window:{MediaSource:MockMediaSource,dispatchEvent(){}},
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
console.log("HEVC MSE preview: project-scoped segment 0 appended successfully");
