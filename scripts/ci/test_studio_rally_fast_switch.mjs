import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const begin=js.indexOf("  const view=(item,requestedTime=null)=>");
const end=js.indexOf("  const renderReferenceCards=",begin);
const calls=[];
const card=()=>({classList:{add(){calls.push("active");},remove(){calls.push("inactive");}}});
const progress=()=>({style:{setProperty(){}},setAttribute(){}});
const s={project:"p",video:"v",run:"r",videoKey:"p:v",current:"a",playbackMode:"sequence"};
const video={readyState:2,paused:false,set currentTime(value){calls.push(["seek",value]);},play(){calls.push("play");return Promise.resolve();}};
const context={s,ui:{video,viewTitle:{},videoTime:{},emptyVideo:{classList:{add(){}}}},
  rallyCardProgresses:new Map([["a",{card:card(),progress:progress(),start:0}],["b",{card:card(),progress:progress()}]]),
  cards(){calls.push("rebuild");},fps:()=>30,clock:String,second:String,rallyClipFrames:row=>({start:row.start_frame,end:row.end_frame}),
  renderRallyTouches(){calls.push("touches");},setOverlaySource(){},prefetchOverlayAtFrame(){},drawTimeline(){},
  previewIsVisible:()=>true,message(){}};
runInNewContext(js.slice(begin,end)+"\nthis.view=view;",context);
context.view({candidate_id:"b",start_frame:600,end_frame:899});
assert.deepEqual(calls[0],["seek",20],"Media seek must start before card and annotation rendering");
assert.ok(!calls.includes("rebuild"),"Mounted rally cards must not be rebuilt during sequential playback");
assert.ok(calls.includes("inactive")&&calls.includes("active"));
assert.equal(s.current,"b");
console.log("Sequential rally seeks precede UI work and preserve mounted cards");


const prefetchStart=js.indexOf("  let warmRallyKey=null;");
const prefetchEnd=js.indexOf('  for(const name of ["pause","emptied","seeked"])',prefetchStart);
assert.ok(prefetchStart>0&&prefetchEnd>prefetchStart);
const warmed=[],overlays=[];
const state={project:"p",run:"r",video:"v",videoKey:"p:v",current:"a",playbackMode:"sequence",analysisOffset:10,stopAt:30};
const media={paused:false,seeking:false,currentTime:24};
const prefetchContext={s:state,ui:{video:media},training:{active:false},overlay:{raw:false},previewIsVisible:()=>true,
  fps:()=>30,rallyClipFrames:row=>({start:row.start_frame,end:row.end_frame}),nextRally:()=>({start_frame:900,end_frame:1200}),
  overlayWindowStart:frame=>frame-60,requestOverlayWindow:(...args)=>overlays.push(args),
  window:{VolleyballPlayback:{warm:(owner,seek)=>{assert.equal(owner,media);warmed.push(seek);},releaseWarm:()=>warmed.push("release")}}};
runInNewContext(js.slice(prefetchStart,prefetchEnd)+"\nthis.prepare=prepareNextRally;this.release=releaseRallyWarm;",prefetchContext);
prefetchContext.prepare();
assert.equal(warmed.length,0,"Do not preload too early");
media.currentTime=25;
prefetchContext.prepare();prefetchContext.prepare();
assert.deepEqual(warmed,[40],"Use the next detected frame and analysis offset once");
assert.deepEqual(overlays,[["p","r","v",840]],"Prepare the same next rally overlay");
prefetchContext.release();
media.paused=true;prefetchContext.prepare();
assert.deepEqual(warmed,[40,"release"],"Do not load media while paused");
media.paused=false;state.playbackMode="free";prefetchContext.prepare();
assert.deepEqual(warmed,[40,"release"],"Free playback does not preload rallies");
console.log("Sequence prepares only the next rally within five seconds of the boundary");
