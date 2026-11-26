import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {join} from "node:path";
import {fileURLToPath} from "node:url";
import {runInNewContext} from "node:vm";

const root=fileURLToPath(new URL("../..",import.meta.url));
const html=readFileSync(join(root,"rust/volley-app/static/studio.html"),"utf8");
const css=readFileSync(join(root,"rust/volley-app/static/studio.css"),"utf8");
const js=readFileSync(join(root,"rust/volley-app/static/studio.js"),"utf8");
const wrap=html.slice(html.indexOf('<div class="video-wrap">'),
  html.indexOf('<div class="viewer-details"'));
for(const id of ["studio-video-play","studio-video-mute",
  "studio-video-fullscreen","studio-video-mode"]){
  assert.equal((html.match(new RegExp('id="'+id+'"','g'))||[]).length,1,
    id+" must appear exactly once");
  assert.ok(wrap.includes('id="'+id+'"'),id+" must be inside video");
}
assert.doesNotMatch(html,/id="studio-video-step-(?:backward|forward)"/,"Step buttons must not appear in preview");
assert.doesNotMatch(js,/playback\.(?:backward|forward)\b/,"Removed controls must not be accessed");
for(const mode of ["sequence","loop","free","once"])
  assert.match(wrap,new RegExp('<option value="'+mode+'">'));
assert.match(css,/#studio-preview-view \.viewer-playback-controls\{\s*position:absolute/);
assert.match(css,/#studio-preview-view \.video-playback-mode\{\s*position:absolute/);
assert.match(css,/grid-template-areas:"player list" "timeline timeline"/);
assert.match(js,/ui\.video\.play\(\)\.catch\(error=>\s*message/);
assert.match(js,/document\.addEventListener\("keydown",event=>\{[\s\S]*?event\.stopPropagation\(\);[\s\S]*?toggleVideoPlayback\(\);\s*\},true\)/);
assert.match(js,/isTypingSpaceTarget\(event\.target\)/);

const start=js.indexOf('  const nextRally=()=>{');
const end=js.indexOf('  ui.video.addEventListener("timeupdate"',start);
assert.ok(start>=0&&end>start,"Rally boundary logic must be present");
const script=js.slice(start,end)+"\nthis.nextRally=nextRally;this.onRallyBoundary=onRallyBoundary;";
const rallies=[
  {candidate_id:"a",start_frame:0,end_frame:299},
  {candidate_id:"b",start_frame:450,end_frame:749},
  {candidate_id:"c",start_frame:900,end_frame:1199}
];
const s={project:"p",run:"r",current:"a",clipStart:0,clipEnd:10,
  stopAt:10,seekAt:0,playbackMode:"sequence",
  items:[{candidate_id:"a",detected_start_frame:0,detected_end_frame:299}]};
let paused=0;const played=[],followed=[];
const ui={video:{currentTime:10,paused:false,play:()=>Promise.resolve(),
  pause:()=>{paused++;}}};
const timeline={data:{project_id:"p",rally_run_id:"r",rallies}};
const ctx={fps:()=>30,s,ui,timeline,view:item=>{played.push(item.candidate_id);s.current=item.candidate_id;},followSequenceRally:()=>followed.push(s.current),message:()=>{}};
runInNewContext(script,ctx,{filename:"studio-rally-boundary.js"});
ctx.onRallyBoundary();
assert.deepEqual(played,["b"],"Sequence must use full timeline, not paginated cards");
assert.deepEqual(followed,["b"],"Only an automatic sequence transition follows the next card");
s.current="c";s.clipEnd=40;ctx.onRallyBoundary();
assert.equal(paused,1,"Sequence must stop after last rally");
s.playbackMode="loop";s.clipStart=30;s.clipEnd=40;
s.stopAt=40;ui.video.currentTime=40;
ctx.onRallyBoundary();
assert.equal(ui.video.currentTime,30,"Loop must return to selected rally start");
assert.equal(s.stopAt,40);
s.playbackMode="once";ctx.onRallyBoundary();
assert.equal(paused,2,"Once must pause at rally boundary");
assert.equal(s.stopAt,null);
s.playbackMode="free";s.stopAt=40;ctx.onRallyBoundary();
assert.equal(paused,2,"Free mode must not pause at rally boundary");
assert.equal(s.stopAt,null);
assert.deepEqual(followed,["b"],"Other playback modes must not scroll");
console.log("Studio playback modes and controls: OK");


const followStart=js.indexOf("  const followSequenceRally=async()=>");
const followEnd=js.indexOf("  const view=",followStart);
assert.ok(followStart>=0&&followEnd>followStart);
let mounted=true,cardTop=210,cardHeight=100,loads=0;
const rail={scrollTop:230,clientTop:1,clientHeight:320,
  getBoundingClientRect:()=>({top:100}),
  querySelector:()=>mounted?{getBoundingClientRect:()=>({top:cardTop,bottom:cardTop+cardHeight})}:null};
const state={current:"b",generation:1,videoKey:"p:v",playbackMode:"sequence",canLoadMore:false,items:[]};
const scrollContext={s:state,ui:{list:rail,filter:{value:"all"}},training:{active:false},
  previewIsVisible:()=>true,loadMore:async()=>{loads++;state.items.push({candidate_id:"b"});mounted=true;}};
runInNewContext(js.slice(followStart,followEnd)+"\nthis.follow=followSequenceRally;",scrollContext);
await scrollContext.follow();
assert.equal(rail.scrollTop,230,"Fully visible next card must not jump");
cardTop=380;
await scrollContext.follow();
assert.equal(rail.scrollTop,289,"Partially hidden card: scroll only the bottom overflow");
cardTop=460;
await scrollContext.follow();
assert.equal(rail.scrollTop,428,"Below viewport: reveal bottom of card, not align to top");
cardTop=75;
await scrollContext.follow();
assert.equal(rail.scrollTop,402,"Above viewport: scroll upward only as far as necessary");
mounted=false;state.canLoadMore=true;cardTop=450;
await scrollContext.follow();
assert.equal(loads,1,"Load another page when the playing card is not mounted");
assert.equal(rail.scrollTop,531,"Reveal the card after pagination");
mounted=false;
scrollContext.loadMore=async()=>{loads++;state.current="manual";mounted=true;state.items.push({candidate_id:"manual"});};
await scrollContext.follow();
assert.equal(rail.scrollTop,531,"Do not scroll after a different manual selection during pagination");
mounted=false;state.current="b";
scrollContext.loadMore=async()=>{loads++;state.sequenceFollow=null;mounted=true;state.items.push({candidate_id:"b"});};
await scrollContext.follow();
assert.equal(rail.scrollTop,531,"A repeated manual click cancels an outstanding follow");
mounted=false;
scrollContext.loadMore=async()=>{loads++;};
await scrollContext.follow();
assert.equal(loads,4,"No endless pagination if the request makes no progress");
assert.equal(rail.scrollTop,531);
state.marksLoaded=true;state.gold=new Set();scrollContext.ui.filter.value="marked";
await scrollContext.follow();
assert.equal(loads,4,"Preserve an explicit filter that excludes the playing card");
console.log("Sequence follow: nearest-edge scrolling, pagination, cancellation and filter preservation: OK");
