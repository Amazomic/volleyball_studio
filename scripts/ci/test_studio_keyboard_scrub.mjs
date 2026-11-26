import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const start=js.indexOf("  // Left/right arrows scrub the one source-wide timeline");
const end=js.indexOf("  // Only text editing owns Space.",start);
assert.ok(start>=0&&end>start,"Global accelerated timeline scrub handler must exist");
assert.match(js,/const steps=\{PageDown:-30,PageUp:30\}/,
  "Focused timeline slider must not also apply an independent arrow seek");

let now=0,visible=true,nextFrame=1,duration=6000;
const queued=new Map(),documentHandlers=new Map(),windowHandlers=new Map(),seeks=[];
const s={project:"p",video:"match.mp4",videoKey:"p:match.mp4",seekAt:100,current:"a",generation:1};
const video={currentTime:100,readyState:2,src:"",getAttribute(){return this.src;},paused:false,pause(){this.paused=true;}};
const visited=[];
let mountedIds=["a","c","b"];
const list={querySelectorAll:()=>mountedIds.map(id=>({
  getAttribute:()=>id,click(){visited.push(id);s.current=id;}
}))};
const document={
  hidden:false,
  addEventListener:(name,handler,capture)=>documentHandlers.set(name+":"+!!capture,handler)
};
// Exercise the real compiled Rust WASM policy, not a JavaScript reimplementation.
const wasmBytes=readFileSync(new URL("../../rust/volley-app/static/volley_web.wasm",import.meta.url));
const engine=new WebAssembly.Instance(new WebAssembly.Module(wasmBytes),{}).exports;

const near=(actual,expected,message)=>assert.ok(Math.abs(actual-expected)<1e-8,message??(actual+" != "+expected));
for(const rate of [24,25,29.97,30,59.94,60,120]){
  const position=Math.round(100.25/rate*1e6)/1e6;
  near(engine.volley_preview_seek_step(position,1,60,1),position+0.1);
  near(engine.volley_preview_seek_step(position,-1,60,1),position-0.1);
  let time=position;
  for(let i=0;i<10;i++)time=engine.volley_preview_seek_step(time,1,60,1);
  near(time,position+1,"Ten steps must advance exactly one second");
  for(let i=0;i<10;i++)time=engine.volley_preview_seek_step(time,-1,60,1);
  near(time,position);
  assert.equal(engine.volley_preview_seek_step(0,-1,60,1),0);
  assert.ok(engine.volley_preview_seek_step(60,1,60,1)<60);
}
assert.equal(engine.volley_preview_seek_hold_delta(1,279,100),0);
assert.equal(engine.volley_preview_seek_hold_delta(1,280,100),0.01);
const window={VolleyballUi:{engine},
  addEventListener:(name,handler)=>windowHandlers.set(name,handler)};
const context={overlay:{presentedTime:null},
  document,window,s,ui:{video,list},performance:{now:()=>now},
  previewIsVisible:()=>visible&&!document.hidden,
  timelineDuration:()=>duration,previewLoad:{sourceFps:60},fps:()=>30,
  loadMore:async()=>{mountedIds.push("d");s.canLoadMore=false;},
  isTypingSpaceTarget:target=>target?.isContentEditable||["TEXTAREA","SELECT"].includes(target?.tagName)||(target?.tagName==="INPUT"&&target.type!=="range"),
  seekTimeline:(position,keyboard)=>{
    assert.equal(keyboard,true,"Keyboard must use source-wide paused seek");
    s.seekAt=Math.min(duration,Math.max(0,position));
    video.currentTime=s.seekAt;
    seeks.push({time:now,position:s.seekAt});
  },
  requestAnimationFrame:callback=>{
    const id=nextFrame++;
    queued.set(id,callback);
    return id;
  },
  cancelAnimationFrame:id=>queued.delete(id)
};
runInNewContext(js.slice(start,end)+"\nthis.arrowScrub=arrowScrub;",context);
const down=documentHandlers.get("keydown:true");
const up=documentHandlers.get("keyup:true");
assert.equal(typeof down,"function");
assert.equal(typeof up,"function");
const key=(value,options={})=>{
  const event={
    key:value,target:{tagName:"DIV",isContentEditable:false},
    shiftKey:false,altKey:false,ctrlKey:false,metaKey:false,
    isComposing:false,repeat:false,prevented:0,stopped:0,
    preventDefault(){this.prevented++;},
    stopPropagation(){this.stopped++;},
    ...options
  };
  down(event);
  return event;
};
const release=value=>up({key:value});
const advanceTo=endTime=>{
  while(now<endTime){
    now=Math.min(now+16,endTime);
    const frames=[...queued.values()];
    queued.clear();
    for(const frame of frames)frame(now);
  }
};

let event=key("ArrowRight");
assert.ok(Math.abs(s.seekAt-100.1)<1e-8,"A quick right-arrow press advances exactly 100 ms");
const firstPosition=s.seekAt;
assert.equal(event.prevented,1);
assert.equal(event.stopped,1);
assert.equal(video.paused,true,"Arrow seek pauses playback immediately");
key("ArrowRight",{repeat:true});
assert.equal(seeks.length,1,"OS key-repeat must not cause repeated base seeks");
advanceTo(240);
assert.equal(s.seekAt,firstPosition,"A tap should not start a continuous seek");
release("ArrowRight");
advanceTo(600);
assert.equal(s.seekAt,firstPosition,"Seeking must stop on keyup");
assert.equal(queued.size,0);

now=700;
key("ArrowRight");
const secondPosition=s.seekAt;
advanceTo(1900);
const afterEarly=s.seekAt;
advanceTo(3400);
const afterLate=s.seekAt;
const earlyPerSecond=(afterEarly-secondPosition)/(1.2);
const latePerSecond=(afterLate-afterEarly)/1.5;
assert.ok(earlyPerSecond>0);
assert.ok(latePerSecond>earlyPerSecond*2,
  "Holding a direction must accelerate exponentially over time");
release("ArrowRight");
const settled=s.seekAt;
advanceTo(3500);
assert.equal(s.seekAt,settled);

key("ArrowLeft",{shiftKey:true});
near(s.seekAt,settled-0.1,
  "Shift+Left also moves exactly 100 ms");
release("ArrowLeft");
const beforeIgnored=s.seekAt;
for(const target of [
  {tagName:"INPUT",type:"text"}, {tagName:"TEXTAREA"}, {tagName:"SELECT"},
  {tagName:"DIV",isContentEditable:true}
]){
  const ignored=key("ArrowRight",{target});
  assert.equal(ignored.prevented,0,"Editable fields keep native arrow behavior");
}
assert.equal(s.seekAt,beforeIgnored);

for(const target of [
  {tagName:"INPUT",type:"range"},
  {tagName:"DIV",closest:selector=>selector===".rally-card-progress"?{}:null},
  {tagName:"BUTTON"}
]){
  const before=s.seekAt;
  const consumed=key("ArrowRight",{target});
  assert.equal(consumed.prevented,1);
  near(s.seekAt,before+0.1);
  release("ArrowRight");
}
video.src="blob:fragment";
const beforeFragment=video.currentTime;
key("ArrowRight");
near(s.seekAt,beforeFragment+0.1,"Fragments use the same time step");
release("ArrowRight");
video.src="/api/video-media?playback=compatible";
const beforeCompatible=video.currentTime;
key("ArrowRight");
near(s.seekAt,beforeCompatible+0.1,"Compatibility copies use the same time step");
release("ArrowRight");
video.src="";
const beforeArrows=s.seekAt;
for(let i=0;i<10;i++){key("ArrowRight");release("ArrowRight");}
near(s.seekAt,beforeArrows+1,"Ten right-arrow presses must advance one second");
for(let i=0;i<10;i++){key("ArrowLeft");release("ArrowLeft");}
near(s.seekAt,beforeArrows,"Ten left-arrow presses must restore the position");
video.seeking=true;
video.currentTime=0;
{key("ArrowRight");release("ArrowRight");}
near(s.seekAt,beforeArrows+0.1,"A pending seek must advance from its requested target");
video.seeking=false;
video.currentTime=s.seekAt;
video.paused=false;
context.overlay.presentedTime=s.seekAt-0.02;
const displayed=context.overlay.presentedTime;
{key("ArrowRight");release("ArrowRight");}
near(s.seekAt,displayed+0.1,"Playing step starts at the displayed video frame");
assert.equal(video.paused,true);
context.overlay.presentedTime=null;
const afterControls=s.seekAt;
key("ArrowLeft",{ctrlKey:true});
assert.equal(s.seekAt,afterControls,"OS modifier combinations remain available");
visible=false;
key("ArrowRight");
assert.equal(s.seekAt,afterControls,"A hidden preview must ignore arrow seeks");
visible=true;

key("ArrowRight");
advanceTo(4000);
windowHandlers.get("blur")();
const onBlur=s.seekAt;
advanceTo(5000);
assert.equal(s.seekAt,onBlur,"Window blur must cancel an active scrub");
key("ArrowLeft");
document.hidden=true;
documentHandlers.get("visibilitychange:false")();
const onHide=s.seekAt;
advanceTo(5600);
assert.equal(s.seekAt,onHide,"Background tabs must cancel an active scrub");

document.hidden=false;
key("ArrowRight");
const beforeSourceSwitch=s.seekAt;
s.generation++;
advanceTo(6100);
assert.equal(s.seekAt,beforeSourceSwitch,"A held key must not seek a different video or analysis");
assert.equal(queued.size,0);
assert.equal(key("ArrowDown").prevented,1);
assert.deepEqual(visited,["c"],"Down follows mounted order, not time or ID sorting");
key("ArrowDown",{repeat:true});
assert.deepEqual(visited,["c"]);
key("ArrowDown");
assert.deepEqual(visited,["c","b"]);
s.canLoadMore=true;
key("ArrowDown");
await new Promise(setImmediate);
assert.deepEqual(visited,["c","b","d"],"Down loads the next page at the list boundary");
key("ArrowUp");
assert.deepEqual(visited,["c","b","d","b"]);
key("ArrowLeft");
release("ArrowLeft");
s.current="";
key("ArrowUp");
assert.equal(visited.at(-1),"c","Navigation retains its anchor after leaving a rally");

const seekStart=js.indexOf("  const seekTimeline=");
const seekEnd=js.indexOf("  const timelinePosition=",seekStart);
const state={project:"p",video:"match.mp4",videoKey:"p:match.mp4",run:"r",
  current:"a",clipStart:5,clipEnd:10,stopAt:10,playbackMode:"once",seekAt:9.999};
const media={readyState:2,currentTime:9.999,paused:true};
const seekContext={s:state,ui:{video:media,viewTitle:{},viewDescription:{},videoTime:{},
  events:{},emptyVideo:{classList:{add(){}}}},overlay:{raw:false,source:{}},
  timeline:{data:{},playhead:{style:{}}},timelineDuration:()=>20,
  cards(){},drawTimeline(){},rallyAtTimelineTime:()=>null,wipe(){},fps:()=>30,
  prefetchOverlayAtFrame(){},timeLabel:String};
runInNewContext(js.slice(seekStart,seekEnd)+"\nthis.seekTimeline=seekTimeline;",seekContext);
seekContext.seekTimeline(9.998,true);
assert.equal(state.current,"a","Within the selected rally, preserve selection");
assert.equal(state.stopAt,10);
seekContext.seekTimeline(10.001,true);
assert.equal(media.currentTime,10.001,"Keyboard can leave a rally in once mode");
assert.equal(state.stopAt,null);
assert.equal(state.current,"");
seekContext.seekTimeline(20,true);
assert.equal(media.currentTime,20,"No artificial 50 ms gap at the video end");
seekContext.seekTimeline(-1,true);
assert.equal(media.currentTime,0);
seekContext.timeline.data=null;state.run="";
seekContext.seekTimeline(0.001,true);
assert.equal(media.currentTime,0.001,"Keyboard works without CPU results");

const alignStart=js.indexOf("  const alignActiveRally=");
const alignEnd=js.indexOf("  const view=",alignStart);
const rail={scrollTop:50,clientTop:1,getBoundingClientRect:()=>({top:100}),
  querySelector:()=>({getBoundingClientRect:()=>({top:350})})};
const railContext={ui:{list:rail}};
runInNewContext(js.slice(alignStart,alignEnd)+"\nthis.align=alignActiveRally;",railContext);
railContext.align();
assert.equal(rail.scrollTop,299,"Active card is aligned to the top of its own scrolling rail");
console.log("Studio preview keyboard: 100 ms steps, acceleration, pause, cross-rally seek, ordered navigation and active-card alignment: OK");
