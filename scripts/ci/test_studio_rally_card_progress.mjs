import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import {previewWasmWindow} from "./helpers/preview_wasm.mjs";

const source=readFileSync(
  new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8"
);
const colorsStart=source.indexOf("  const cpuTouchStyle={");
const colorsEnd=source.indexOf("  // The timeline belongs",colorsStart);
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
const overlay={presentedTime:null,raw:false,flags:{touches:true},
  actionFlags:{serve:true,attack:true,dig:true,block:true,unknown_touch:true}};
let formatterCreations=0,formattedTimes=0;
const context={
  Intl:{NumberFormat:function(locale,options){
    formatterCreations++;
    const formatter=new Intl.NumberFormat(locale,options);
    return {format(value){formattedTimes++;return formatter.format(value);}};
  }},
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
})),JSON.stringify({start:90,end:160}),
"Preview must use detected boundaries rather than padded montage boundaries");
assert.equal(JSON.stringify(rallyClipFrames({
  detected_start_frame:174,detected_end_frame:366,
  manual_clip:{start_frame:129,end_frame:411}
})),JSON.stringify({start:174,end:366}),
"Live 30 FPS run: preview start must match timeline at 5.8 s, not montage at 4.3 s");
assert.equal(JSON.stringify(rallyClipFrames({
  start_frame:912,end_frame:1440
})),JSON.stringify({start:912,end:1440}),
"Full CPU timeline items use the same start/end frame fields");

const markers=make("span","rally-card-progress-markers");
const progress=make("div","rally-card-progress");
rallyCardProgresses.set(current.current,{progress,markers,start:100,end:149});
refreshRallyCardTouchTicks();
assert.equal(markers.children.length,4,
  "Every Touch Detector candidate within the played clip becomes a mark, even without a linked player");
assert.deepEqual(markers.children.map(x=>x.style.left),["0%","50%","60%","98%"]);
assert.equal(markers.children[0].style.backgroundColor,cpuTouchStyle.attack[0]);
assert.equal(markers.children[1].style.backgroundColor,cpuTouchStyle.dig[0]);
assert.equal(markers.children[2].style.backgroundColor,cpuTouchStyle.block[0]);
assert.equal(markers.children[3].style.backgroundColor,"#c6bf96",
  "Untyped touches have the same color as in the video");
assert.match(progress.attrs["aria-label"],/4 касаний/);
assert.match(markers.children[1].title,/12.5 с/);
overlay.actionFlags.unknown_touch=false;
refreshRallyCardTouchTicks();
assert.equal(markers.children.length,3,"Unknown-type checkbox filters card timeline");
overlay.actionFlags.unknown_touch=true;
refreshRallyCardTouchTicks();
assert.equal(markers.children.length,4,"Re-enabling unknown restores card markers");
overlay.flags.touches=false;
refreshRallyCardTouchTicks();
assert.equal(markers.children.length,0,"Hiding touches hides all card markers");
overlay.flags.touches=true;overlay.raw=true;
refreshRallyCardTouchTicks();
assert.equal(markers.children.length,0,"Raw mode never renders CPU touch marks");
overlay.raw=false;
refreshRallyCardTouchTicks();
assert.equal(markers.children.length,4);

timeline.data.surface_contact_candidates=[
  {frame_index:140,surface:"floor",outcome:"line_uncertain"},
  {frame_index:145,surface:"floor",outcome:"out"}
];
refreshRallyCardTouchTicks();
assert.equal(markers.children.length,6);
assert.ok(markers.children.some(x=>x.title.includes("Граница площадки")));
assert.ok(markers.children.some(x=>x.title.includes("За пределами площадки")));
delete timeline.data.surface_contact_candidates;
refreshRallyCardTouchTicks();
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
let playCalls=0,drawnPosition=NaN;
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
  drawTimeline(){drawnPosition=viewContext.ui.video.currentTime;},
  previewIsVisible:()=>true,message(){},
  url:path=>path,
};
runInNewContext(source.slice(viewStart,viewEnd)+"\nthis.view=view;",viewContext);
const rally={candidate_id:"rally-a",detected_start_frame:100,detected_end_frame:149,
  manual_clip:{start_frame:85,end_frame:165}};
current.playbackMode="sequence";
viewContext.view(rally,12.5);
assert.equal(viewContext.ui.video.currentTime,12.5);
assert.equal(drawnPosition,12.5,
  "Full-video playhead must redraw after actual seek, not at the prior timestamp");
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
assert.equal(drawnPosition,10,"Click must immediately align the playhead with detected start");
assert.ok(!source.includes('event.target?.closest?.(".rally-card-progress")'),
  "Focused card sliders must not override global arrows");
const nextStart=source.indexOf("  const nextRally=()=>{");
const nextEnd=source.indexOf("  const onRallyBoundary=()=>{",nextStart);
assert.ok(nextStart>0&&nextEnd>nextStart);
const rowA={candidate_id:"a",detected_start_frame:174,detected_end_frame:366,
  manual_clip:{start_frame:129,end_frame:411}};
const rowB={candidate_id:"b",detected_start_frame:533,detected_end_frame:656,
  manual_clip:{start_frame:488,end_frame:701}};
const nextContext={
  window:previewWasmWindow(),
  s:{project:"p",run:"r",current:"",analysisOffset:0,clipEnd:520/30,
    items:[rowA,rowB]},
  timeline:{data:null},ui:{video:{currentTime:520/30}},fps:()=>30,
};
runInNewContext(source.slice(nextStart,nextEnd)+"\nthis.nextRally=nextRally;",nextContext);
assert.equal(nextContext.nextRally().candidate_id,"b",
  "Sequence mode must choose the next detected boundary, not its 45-frame montage lead-in");
console.log("Rally preview: detected boundaries align with timeline, padded manual clips preserved for montage, seek/redraw and sequence: OK");

{
const viewStart=source.indexOf("  const alignActiveRally=");
const viewEnd=source.indexOf("  const renderReferenceCards=",viewStart);
const rail={scrollTop:230,clientTop:0,getBoundingClientRect:()=>({top:100}),
  querySelector:()=>({getBoundingClientRect:()=>({top:280})})};
const selection={project:"p",video:"v",current:"a",videoKey:"p:v",playbackMode:"once"};
const selectedVideo={readyState:2,currentTime:0};
let renders=0;
const selectionContext={s:selection,ui:{list:rail,video:selectedVideo,viewTitle:{},videoTime:{},
  emptyVideo:{classList:{add(){}}}},cards(){renders++;},rallyClipFrames:()=>({start:60,end:99}),
  fps:()=>10,clock:String,renderRallyTouches(){},rallyCardProgresses:new Map(),
  setOverlaySource(){},prefetchOverlayAtFrame(){},drawTimeline(){},previewIsVisible:()=>false};
runInNewContext(source.slice(viewStart,viewEnd)+"\nthis.view=view;",selectionContext);
selectionContext.view({candidate_id:"b"});
assert.equal(selection.current,"b");
assert.equal(renders,1,"Selection must redraw the active highlight");
assert.equal(selectedVideo.currentTime,6);
assert.equal(rail.scrollTop,230,"Clicking a rally must preserve the list scroll position");
selectionContext.view({candidate_id:"c"},8);
assert.equal(selectedVideo.currentTime,8);
assert.equal(rail.scrollTop,230,"Clicking a different card timeline must preserve scroll");

const moreStart=source.indexOf("  const loadMore=async()=>",viewEnd);
const moreEnd=source.indexOf("  const uploadAudio=",moreStart);
const moreState={project:"p",run:"r",items:[],canLoadMore:true,filtered:2,generation:1,limit:1,current:"c"};
rail.getBoundingClientRect=()=>({top:100,bottom:400});
const moreContext={s:moreState,ui:{list:rail},reference:{active:false},training:{active:false},
  previewCache:{load:async()=>({schema:"volley-native-rally-preview/1",authoritative_game_events:false,
    automatic_montage_eligible:false,items:[{candidate_id:"d"}]})},url(){},rallyListQuery:()=>({}),
  validSurfaceContacts:()=>true,cards(){},drawTimeline(){},message(error){throw Error(error);}};
runInNewContext(source.slice(viewStart,source.indexOf("  const view=",viewStart))+
  source.slice(moreStart,moreEnd)+"\nthis.loadMore=loadMore;",moreContext);
await moreContext.loadMore();
assert.equal(moreState.items.length,1);
assert.equal(rail.scrollTop,230,"Loading another page must not move the selected rally to the top");
}

{
  const remoteMarkers=make("span","rally-card-progress-markers");
  const remoteProgress=make("div","rally-card-progress");
  const remoteSource={run:"cpu-b",video:"other.mp4",fps:20,offset:80,timeline:{
    project_id:"proj",rally_run_id:"cpu-b",video:"other.mp4",
    touch_annotations:[{frame_index:125,kind:"attack"}]
  }};
  overlay.raw=false;overlay.flags.touches=true;overlay.actionFlags.attack=true;
  rallyCardProgresses.set("remote",{progress:remoteProgress,markers:remoteMarkers,start:100,end:149,source:remoteSource});
  refreshRallyCardTouchTicks();
  assert.equal(remoteMarkers.children.length,1,"Inactive card has its own video markers");
  assert.equal(remoteMarkers.children[0].style.left,"50%");
  assert.equal(remoteMarkers.children[0].style.backgroundColor,cpuTouchStyle.attack[0]);
  assert.match(remoteMarkers.children[0].title,/86,3/,"Inactive timeline uses source FPS and offset");
  remoteSource.timeline.touch_annotations=[
    {frame_index:100,kind:"attack"},{frame_index:125,kind:"attack"},{frame_index:149,kind:"attack"}
  ];
  remoteSource.offset=1234.5;
  const before=formatterCreations, formattedBefore=formattedTimes;
  refreshRallyCardTouchTicks();
  assert.equal(formatterCreations-before,1,"Create one Intl formatter per marker refresh, not per event");
  assert.equal(formattedTimes-formattedBefore,3,"All source-specific marker times use the reused formatter");
  assert.deepEqual(remoteMarkers.children.map(tick=>tick.title),
    remoteSource.timeline.touch_annotations.map(event=>
      (event.frame_index/remoteSource.fps+remoteSource.offset)
        .toLocaleString("ru-RU",{minimumFractionDigits:1,maximumFractionDigits:1})+" с · Атака"));
}
