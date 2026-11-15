import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(
  new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8"
);
const colorsStart=source.indexOf("  const cpuTouchStyle={");
const colorsEnd=source.indexOf("  };",colorsStart)+4;
const start=source.indexOf("  const rallyClipFrames=item=>");
const end=source.indexOf("  // Event pills and summary",start);
assert.ok(colorsStart>0&&colorsEnd>colorsStart,"CPU overlay palette must exist");
assert.ok(start>0&&end>start,"Rally card rendering helpers must exist");

const make=(tag,className)=>{
  const attrs={};
  return {
    tag,className,attrs,children:[],
    style:{setProperty(key,value){this[key]=value;}},
    setAttribute(name,value){attrs[name]=value;},
    append(...children){this.children.push(...children);},
  };
};
const wipe=node=>{node.children=[];};
const current={
  project:"proj",run:"cpu-a",video:"match.mp4",
  current:"rally-a",videoKey:"proj:match.mp4",seekAt:10,
};
const timeline={data:{
  project_id:"proj",rally_run_id:"cpu-a",video:"match.mp4",
  touch_annotations:[
    {frame_index:95,kind:"serve",touch_attributed:true},
    {frame_index:100,kind:"attack",touch_attributed:true},
    {frame_index:125,kind:"dig",contact_verified:true},
    {frame_index:130,kind:"block",touch_attributed:false,contact_verified:false},
    {frame_index:149,kind:null,touch_attributed:true},
    {frame_index:150,kind:"serve",touch_attributed:true},
  ]
}};
const rallyCardProgresses=new Map();
const video={currentTime:12.5,paused:false};
const overlay={presentedTime:null};
const context={
  make,wipe,s:current,timeline,rallyCardProgresses,overlay,
  ui:{video},fps:()=>10,second:frame=>(frame/10).toFixed(1),
};
runInNewContext(source.slice(colorsStart,colorsEnd)+"\n"+
  source.slice(start,end)+"\n"+
  "this.api={rallyClipFrames,refreshRallyCardTouchTicks,updateRallyCardProgress,cpuTouchStyle};",
context);
const {rallyClipFrames,refreshRallyCardTouchTicks,updateRallyCardProgress,cpuTouchStyle}=context.api;
assert.equal(JSON.stringify(rallyClipFrames({
  detected_start_frame:90,detected_end_frame:160,
  manual_clip:{start_frame:100,end_frame:149},
})),JSON.stringify({start:100,end:149}));

const markers=make("span","rally-card-progress-markers");
const progress=make("div","rally-card-progress");
rallyCardProgresses.set(current.current,{progress,markers,start:100,end:149});
refreshRallyCardTouchTicks();
assert.equal(markers.children.length,3,
  "Only attributed/verified CPU touches within the played clip become marks");
assert.deepEqual(markers.children.map(x=>x.style.left),["0%","50%","98%"]);
assert.equal(markers.children[0].style.backgroundColor,cpuTouchStyle.attack[0]);
assert.equal(markers.children[1].style.backgroundColor,cpuTouchStyle.dig[0]);
assert.equal(markers.children[2].style.backgroundColor,"#c6bf96",
  "Untyped touches have the same color as in the video");
assert.match(progress.attrs["aria-label"],/3 касаний/);
assert.match(markers.children[1].title,/12.5 с/);

updateRallyCardProgress();
assert.equal(progress.style["--rally-progress"],"50%");
video.currentTime=13.9;overlay.presentedTime=12.5;
updateRallyCardProgress();
assert.equal(progress.style["--rally-progress"],"50%",
  "Progress should follow the presented video frame, not a later media clock");
overlay.presentedTime=null;
video.currentTime=6;updateRallyCardProgress();
assert.equal(progress.style["--rally-progress"],"0%");
video.currentTime=99;updateRallyCardProgress();
assert.equal(progress.style["--rally-progress"],"100%");
timeline.data.rally_run_id="cpu-b";
refreshRallyCardTouchTicks();
assert.equal(markers.children.length,0,"Do not show old CPU data after run switch");
assert.match(progress.attrs["aria-label"],/Ожидание касаний CPU/);
assert.match(source,/rallyCardProgresses\.clear\(\)/,
  "Rebuilt pages must not retain stale card elements");
assert.match(source,/refreshRallyCardTouchTicks\(\);updateRallyCardProgress\(\)/,
  "Loaded timeline data must refresh mounted card marks");
// The same handler that responds to a card timeline click must seek and play,
// including when video was paused. It must preserve the chosen timestamp.
const viewStart=source.indexOf("  const view=(item,requestedTime=null)=>");
const viewEnd=source.indexOf("  const cards=()=>{",viewStart);
assert.ok(viewStart>=0&&viewEnd>viewStart,"Rally view handler must exist");
let playCalls=0;
const viewContext={
  s:current,ui:{
    video:{
      currentTime:10,readyState:2,
      play(){playCalls++;return Promise.resolve();},
    },
    viewTitle:{},videoTime:{},
    emptyVideo:{classList:{add(){}}},
  },
  overlay, fps:()=>10,
  clock:frame=>(frame/10).toFixed(1),rallyClipFrames,
  rallyCardProgresses, cards(){throw Error("An active card must not rebuild");},
  renderRallyTouches(){},setOverlaySource(){},prefetchOverlayAtFrame(){},
  drawTimeline(){},previewIsVisible:()=>true,message(){},
  url:path=>path,
};
runInNewContext(source.slice(viewStart,viewEnd)+"\nthis.view=view;",viewContext);
const rally={candidate_id:"rally-a",detected_start_frame:100,detected_end_frame:149};
current.playbackMode="sequence";
viewContext.view(rally,12.5);
assert.equal(viewContext.ui.video.currentTime,12.5);
assert.equal(current.seekAt,12.5);
assert.equal(current.clipStart,10);
assert.equal(current.clipEnd,15);
assert.equal(current.stopAt,15);
assert.equal(playCalls,1,"A timeline seek must resume playback");
viewContext.view(rally,99);
assert.equal(viewContext.ui.video.currentTime,14.9,
  "Seeking near the end must clamp within the rally");
viewContext.view(rally);
assert.equal(viewContext.ui.video.currentTime,10,
  "Clicking the card body should still start at the beginning");
assert.equal(playCalls,3);
assert.ok(source.includes('event.target?.closest?.(".rally-card-progress")'),
  "Global media shortcuts must defer to the focused card slider");
console.log("Rally card markers, palette, manual clip and interactive playback: OK");
