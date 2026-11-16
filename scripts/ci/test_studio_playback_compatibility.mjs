import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";

const script=readFileSync(new URL("../../rust/volley-app/static/studio_playback.js",import.meta.url),"utf8");
async function scenario(mode){
  const requests=[],handlers=new Map(),documentEvents=new Map(),notifications=[];
  const video={
    videoWidth:96,readyState:2,error:null,src:"",muted:false,
    addEventListener(type,fn){handlers.set(type,fn);},
    removeEventListener(type){handlers.delete(type);},
    removeAttribute(){this.src="";},pause(){},
    load(){
      if(!this.src)return;
      queueMicrotask(()=>{
        if(mode==="success")handlers.get("loadeddata")?.();
        else if(mode!=="timeout"){
          this.error={code:mode==="network"?2:4};handlers.get("error")?.();
        }
      });
    },
    canPlayType(){return "";}
  };
  const window={location:{origin:"http://127.0.0.1:8875"},addEventListener(){},dispatchEvent:event=>notifications.push(event),
    VolleyballStudioActivity:{async fetch(path,options={}){
      requests.push({path,options});
      const report=options.body?JSON.parse(options.body):{};
      const code=report.browser_error_code;
      return {ok:true,json:async()=>code===4?
        {state:"ready",video:report.video,media_url:"/api/video-media?"+new URLSearchParams({
          project_id:report.project_id,video:report.video,playback:"compatible"
        })}:
        {state:"original"}};
    }}
  };
  const document={createElement:()=>video,addEventListener:(name,handler)=>documentEvents.set(name,handler),getElementById:()=>null};
  const timers=new Map();let timerId=0;
  const context={window,document,URL,URLSearchParams,Map,WeakSet,Promise,Event,
    CustomEvent:class extends Event {constructor(type,options){super(type);this.detail=options.detail;}},
    setTimeout(fn){timers.set(++timerId,fn);if(mode==="timeout")queueMicrotask(fn);return timerId;},
    clearTimeout(id){timers.delete(id);},console};
  vm.runInNewContext(script,context);
  const result=await window.VolleyballPlayback.resolve("123456abcdef","match.mp4");
  return {result,requests,video,documentEvents,notifications};
}
for(const mode of ["success","network","timeout"]){
  const {result,requests,video}=await scenario(mode);
  assert.match(result,/\/api\/video-source\?/);
  assert.ok(requests.every(r=>!r.options.body||JSON.parse(r.options.body).browser_error_code===2));
  if(mode!=="network")assert.equal(requests.length,0);
  assert.equal(video.src,"");
}
const unsupported=await scenario("unsupported");
assert.match(unsupported.result,/playback=compatible/);
assert.equal(JSON.parse(unsupported.requests[0].options.body).browser_error_code,4);
const actual={tagName:"VIDEO",src:"/api/video-source?project_id=123456abcdef&video=match.mp4",
  getAttribute(){return this.src;},error:{code:4},currentTime:5,paused:true};
let intercepted=false;
unsupported.documentEvents.get("error")({target:actual,stopImmediatePropagation(){intercepted=true;}});
actual.src="/api/video-source?project_id=123456abcdef&video=other.mp4";
await new Promise(resolve=>setImmediate(resolve));
assert.equal(intercepted,true);
assert.match(actual.src,/other.mp4/);
const changing={...actual,src:"/api/video-source?project_id=123456abcdef&video=first.mp4",
  addEventListener(){},load(){}};
const before=unsupported.requests.length;
const fail=()=>unsupported.documentEvents.get("error")({target:changing,stopImmediatePropagation(){}});
fail();
changing.src="/api/video-source?project_id=123456abcdef&video=second.mp4";
fail();
await new Promise(resolve=>setImmediate(resolve));
assert.equal(unsupported.requests.length,before+2);
assert.match(changing.src,/video=second.mp4/);
assert.match(changing.src,/playback=compatible/);
assert.equal(unsupported.notifications.at(-1).detail.video,changing);
console.log("Browser playback: actual decode success, unsupported codec, network failure and timeout passed");
