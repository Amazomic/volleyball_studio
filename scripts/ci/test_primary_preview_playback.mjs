import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
const pending=[];
const readyUrl="/api/video-primary-preview?project_id=p&video=a.mp4&media=1&preview_id=abc";
let delegated=0;
const video={tagName:"VIDEO",paused:true,currentTime:0,src:"",load(){},pause(){},removeAttribute(){this.src="";},addEventListener(){},removeEventListener(){},getAttribute(){return this.src;}};
const windowRef={MediaSource:class{},VolleyballPlayback:{primary:()=>new Promise(resolve=>pending.push(resolve)),direct:(target,source)=>{target.src=source;delegated++;}},dispatchEvent(){}};
windowRef.MediaSource.isTypeSupported=()=>true;
vm.runInNewContext(fs.readFileSync(new URL("../../rust/volley-app/static/studio_fragments.js",import.meta.url),"utf8"),{
  window:windowRef,MediaSource:windowRef.MediaSource,document:{documentElement:{},querySelectorAll:()=>[],addEventListener(){}},
  URL,URLSearchParams,location:{href:"http://localhost/studio",origin:"http://localhost"},MutationObserver:class{observe(){}},
  fetch(){throw Error("Ready preview must not fetch original HEVC or fragments");},console,CustomEvent:class{},Event:class{}
});
const first=windowRef.VolleyballBrowserVideo.load(video,"p","a.mp4",{seek:12});
const second=windowRef.VolleyballBrowserVideo.load(video,"p","b.mp4");
pending[1](readyUrl.replace("a.mp4","b.mp4"));await second;
pending[0](readyUrl);await first;
assert.equal(delegated,1);
assert.match(video.src,/video=b.mp4/);
console.log("primary preview uses direct media and rejects a stale per-video load");

const metadata=[];let plays=0,queries=0;
const actualVideo={paused:true,currentTime:0,src:"",pause(){},load(){},getAttribute(){return this.src;},addEventListener(name,fn){if(name==="loadedmetadata")metadata.push(fn);},play(){plays++;return Promise.resolve();}};
const completionHandlers=new Map();
const actualWindow={location:{origin:"http://localhost"},addEventListener(name,fn){completionHandlers.set(name,fn);},VolleyballStudioActivity:{fetch:async path=>{
  assert.match(path,/^\/api\/video-primary-preview\?/);queries++;
  return {ok:true,json:async()=>({project_id:"p",video:"a.mp4",state:"ready",media_url:readyUrl})};
}}};
vm.runInNewContext(fs.readFileSync(new URL("../../rust/volley-app/static/studio_playback.js",import.meta.url),"utf8"),{
 location:{href:"http://localhost/studio",origin:"http://localhost"},window:actualWindow,document:{addEventListener(){},querySelectorAll:()=>[actualVideo],createElement(){throw Error("Ready preview must not probe original");}},URL,URLSearchParams,console
});
assert.equal(await actualWindow.VolleyballPlayback.resolve("p","a.mp4"),readyUrl);
assert.equal(queries,1);
actualWindow.VolleyballPlayback.direct(actualVideo,readyUrl,{seek:10,autoplay:true});
actualWindow.VolleyballPlayback.direct(actualVideo,readyUrl.replace("a.mp4","b.mp4"));
actualWindow.VolleyballPlayback.direct(actualVideo,readyUrl,{seek:20,autoplay:false});
for(const callback of metadata)callback();
assert.equal(plays,0,"A-B-A must honor latest paused selection");
assert.equal(actualVideo.currentTime,20);
console.log("primary preview bypasses original probing and rejects stale A-B-A metadata");

actualVideo.src="/api/video-media?project_id=p&video=a.mp4";
completionHandlers.get("volley:pipeline-complete")({detail:{project_id:"p",video:"a.mp4"}});
await new Promise(resolve=>setImmediate(resolve));
assert.equal(actualVideo.src,readyUrl,"analysis completion must adopt the prepared MP4");
assert.equal(queries,2);
console.log("analysis completion adopts prepared playback without a page refresh");
