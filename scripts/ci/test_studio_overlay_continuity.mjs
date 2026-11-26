import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
assert.doesNotMatch(html,/id="studio-overlay-stabilizer"/);
assert.match(html,/id="studio-debug-stabilizer-compare"[^>]*aria-pressed="false"[^>]*>Оба<\/button>/);
for(const id of ["studio-ball-stabilizer-controls","studio-debug-stabilizer-gap",
  "studio-debug-stabilizer-step","studio-debug-stabilizer-stats"]){
  assert.ok(!html.includes('id="'+id+'"'),
    "Both must not render any Ball Stabilizer settings panel: "+id);
}
assert.ok(!js.includes("stabilizer_gap:Number("));
assert.ok(!js.includes("stabilizer_max_step:Number("));
assert.match(js,/stabilizer:true/,"Both must request filtering from the Rust algorithm without custom settings");
assert.ok(!css.includes("#studio-ball-stabilizer-controls"),
  "Both must have no dedicated settings layout");
const trajectoryModes=html.match(/<div id="studio-debug-trajectory-switch"[\s\S]*?<\/div>/)?.[0]||"";
assert.deepEqual([...trajectoryModes.matchAll(/<button[^>]*id="([^"]+)"[^>]*>([^<]+)<\/button>/g)]
  .map(match=>[match[1],match[2]]),[
    ["studio-debug-trajectory-before","Сырые"],
    ["studio-debug-trajectory-after","После Ball Stabilizer"],
    ["studio-debug-stabilizer-compare","Оба"]
  ],"The Debug trajectory control has exactly three exclusive display modes");
assert.match(trajectoryModes,/id="studio-debug-trajectory-legend"[^>]*hidden/);
assert.match(css,/\.trajectory-raw\{background:#47e2ff\}/);
assert.match(css,/\.trajectory-stabilized\{background:#f1bd51\}/);
assert.match(css,/#studio-preview-view:not\(\.training-workspace\) #studio-debug-stabilizer-compare\{display:none\}/);
assert.match(js,/\["ball_stabilizer_v3","ball_stabilizer_v4","ball_stabilizer_v5","ball_stabilizer_v6","ball_stabilizer_v7","ball_stabilizer_v8","ball_stabilizer_v9","ball_stabilizer_v10","ball_stabilizer_v11","ball_stabilizer_v12","ball_stabilizer_v13"\]\.includes\(stats\?\.version\)/,
  "Debug comparison must read historical V3/V4/V5 results alongside V6/V7/V8/V9/V10");
assert.match(html,/id="studio-overlay-processed"[^>]*role="tab"/);
assert.match(html,/id="studio-overlay-raw"[^>]*role="tab"/);
assert.match(html,/id="studio-raw-people-visible"/);
const start=js.indexOf("  const selectedFusion=video=>");
const end=js.indexOf("  const setOverlaySource=",start);
assert.ok(start>=0&&end>start,"Preview overlay window helpers must exist");
const colorStart=js.indexOf("  const playerTeamColor=");
const colorEnd=js.indexOf(";",colorStart)+1;
assert.ok(colorStart>=0&&colorEnd>colorStart);
const script=js.slice(colorStart,colorEnd)+js.slice(start,end)+
  "\nthis.drawOverlay=drawOverlay;this.overlayWindowStart=overlayWindowStart;this.prefetchOverlayAtFrame=prefetchOverlayAtFrame;";

const oldRecord={
  start_frame:0,end_frame:180,
  frames:[{frame:90,players:[{id:1,box:[0.1,0.2,0.3,0.8]}],ball:null}]
};
const timers=[];
const calls=[];
let outlines=0;
const renderedLabels=[];
const labelPositions=[];
const actionColors=[];
const ballColors=[];
const poseFills=[];
const ctx={
  save(){},restore(){},setLineDash(){},strokeRect(){outlines++;},
  beginPath(){},arc(){},moveTo(){},lineTo(){},fill(){
    if(["#47e2ff","#ff8f55"].includes(this.fillStyle))ballColors.push(this.fillStyle);
    if(this.fillStyle==="#f9d977")poseFills.push(this.globalAlpha);
  },stroke(){actionColors.push(this.strokeStyle);},
  measureText(){return {width:20};},
  fillRect(){},fillText(value,x,y){renderedLabels.push(value);labelPositions.push({value,x,y});}
};
const s={project:"p",analysis:"a",chain:{choices:[{value:"a",kind:"fusion",video:"v",id:"f"}]},run:"r",video:"v",videoKey:"p:v"};
const ui={video:{currentTime:90/30,src:"/api/video-media"}};
const overlayLabels={key:"",names:new Map(),loaded:false};
const overlay={
  source:{events:[]},flags:{players:true,touches:false,trajectory:false},
  cache:new Map([["p:cpu:r:v:0",oldRecord]]),rawFrames:new Map(),
  fetching:new Set(),failed:new Map()
};
const timeline={data:{project_id:"p",rally_run_id:"r",video:"v",total_frames:300}};
const scope={s,ui,window:{},overlay,overlayLabels,timeline,setTimeout:fn=>{timers.push(fn);return timers.length;},clearTimeout(){},
  playerLabelKey:(project,run,video,sha)=>[project,run,video,sha].join(":"),
  requestOverlayLabels:()=>{},
  overlayContext:()=>({ctx,x:0,y:0,width:100,height:100}),
  fps:()=>30,url:(path,args)=>path+"?start_frame="+args.start_frame,
  request:resource=>{calls.push(resource);return new Promise(()=>{});},
  overlayMessage:()=>{}
};
runInNewContext(script,scope,{filename:"studio-overlay-window.js"});
assert.equal(scope.overlayWindowStart(89),0);
assert.equal(scope.overlayWindowStart(90),30);
assert.equal(scope.overlayWindowStart(180),120);

// At the 3-second boundary, the new request is outstanding but the
// previous window still contains the frame. Its boxes must not disappear.
// Prefetch 1.5 seconds before the new window becomes active.
ui.video.currentTime=45/30;
scope.drawOverlay();
assert.equal(calls.filter(s=>s.includes("start_frame=30")).length,1,
  "Next window must be prefetched before the 3-second boundary");

// A slow request must not erase already available player boxes.
ui.video.currentTime=90/30;
scope.drawOverlay();
scope.drawOverlay();
assert.equal(outlines,2,"Prior window must render while the next is fetching");
assert.equal(calls.filter(s=>s.includes("start_frame=30")).length,1,
  "Overlapping request must be deduplicated");

// No fabricated contours when saved evidence does not cover the frame.
ui.video.currentTime=180/30;
scope.drawOverlay();
assert.equal(outlines,2,"No stale boxes outside source window");
assert.equal(calls.filter(s=>s.includes("start_frame=120")).length,1);
// GPU overlays must remain useful without a CPU run or saved Touchs.
overlay.raw=true;s.run="";
overlay.rawBall={value:"70"};
overlay.rawVballVisible={checked:true};overlay.rawPeopleVisible={checked:true};
overlay.rawPlayer={value:"60"};overlay.rawBall={value:"0"};
overlay.rawPoseVisible={checked:false};overlay.rawPose={value:"35"};
overlay.flags.touches=true;
overlay.cache.set("p:gpu:f:v:0",{
  start_frame:0,end_frame:180,frames:[{frame:45,raw:{
    balls:[{source:"vballnet",point:[0.2,0.2],score:0.8},
      {source:"unsupported",point:[0.3,0.3],score:0.7}],
    players:[{box:[0.1,0.2,0.3,0.8],score:0.9,touch_probability:0.9}],poses:[]
  }}]
});
ui.video.currentTime=45/30;
scope.drawOverlay();
assert.deepEqual(ballColors,["#47e2ff"],"Only VBallNet supplies the ball");
assert.equal(outlines,3,"Raw player boxes are independent of a CPU replay");
assert.equal(actionColors.length,0,"Raw contact probabilities cannot draw CPU Touchs");
assert.ok(!renderedLabels.some(label=>String(label).includes("legacy p(contact)")),
  "legacy probabilities are not displayed in the new player overlay");
ballColors.length=0;overlay.rawPeopleVisible.checked=false;
scope.drawOverlay();
assert.deepEqual(ballColors,["#47e2ff"],"People overlays can be hidden independently");
overlay.rawPeopleVisible.checked=true;overlay.rawVballVisible.checked=false;
ballColors.length=0;scope.drawOverlay();
assert.deepEqual(ballColors,[],"VBallNet can be hidden independently");
overlay.rawVballVisible.checked=true;overlay.rawBall.value="90";
ballColors.length=0;scope.drawOverlay();
assert.deepEqual(ballColors,[],"Ball confidence is filtered locally");
// Interpolated raw poses are visually distinct and never overwrite actual
// model samples on the six-frame RTMPose cadence.
const rawFrame=overlay.cache.get("p:gpu:f:v:0").frames[0].raw;
const joints=Array.from({length:17},()=>[0.3,0.4,0.95]);
rawFrame.poses_interpolated=[{keypoints:joints}];
rawFrame.poses=[];
overlay.rawPoseVisible.checked=true;
scope.drawOverlay();
assert.equal(poseFills.length,17);
assert.ok(poseFills.every(value=>value===0.78));
poseFills.length=0;
rawFrame.poses=[{keypoints:joints}];
scope.drawOverlay();
assert.equal(poseFills.length,17);
assert.ok(poseFills.every(value=>value===1));
overlay.rawPoseVisible.checked=false;
const comparisonRecord={
  width:100,height:100,start_frame:0,end_frame:180,
  frames:[
    {frame:44,raw:{balls:[{source:"vballnet",point:[0.2,0.25],score:0.2}]},
      stabilized_ball:[0.3,0.3],ball_interpolated:false,ball_rejected:false},
    {frame:45,raw:{balls:[{source:"vballnet",point:[0.92,0.86],score:0.2}]},
      stabilized_ball:[0.32,0.35],ball_interpolated:true,ball_rejected:false}
  ]
};
overlay.cache.set("p:gpu-stabilizer:f:v:0",comparisonRecord);
overlay.stabilizerView=true;
overlay.rawBallTrail={checked:false};
overlay.rawBall.value="99";
overlay.rawVballVisible.checked=false;
overlay.rawPoseVisible.checked=true;
ui.video.currentTime=45/30;
actionColors.length=0;ballColors.length=0;poseFills.length=0;
scope.drawOverlay();
assert.equal(actionColors.filter(color=>color==="#47e2ff").length,0,
  "Both must not connect raw VBallNet detections, even across consecutive frames or a teleport");
assert.equal(ballColors.filter(color=>color==="#47e2ff").length,2,
  "Both displays exactly the two measured VBallNet observations as cyan dots");
assert.ok(actionColors.includes("#f1bd51"),"Both draws the gold stabilized trajectory in the same frame");
assert.ok(ballColors.includes("#47e2ff"),"Both displays unfiltered raw detections below the raw confidence slider");
assert.deepEqual(poseFills,[],"Both compares ball trajectories without unrelated RTMPose skeletons");
overlay.stabilizerView=false;overlay.rawPoseVisible.checked=false;
actionColors.length=0;
overlay.raw=false;s.run="r";
timeline.data.schema="volley-native-rally-timeline/1";
timeline.data.source_replay_sha256="a".repeat(64);
timeline.data.touch_annotations=[
  {frame_index:45,kind:"attack",ball_position_px:[20,20],contact_verified:true},
  {frame_index:45,kind:"block",ball_position_px:[30,30],contact_verified:true},
  {frame_index:45,kind:null,ball_position_px:[40,40],contact_verified:true}
];
overlay.actionFlags={attack:true,block:true,unknown_touch:true};
oldRecord.width=100;oldRecord.height=100;
oldRecord.replay_sha256=timeline.data.source_replay_sha256;
actionColors.length=0;
scope.drawOverlay();
assert.deepEqual(actionColors,["#ff6b6b","#c38cff","#c6bf96"],
  "Processed mode alone shows classified CPU Touchs");
overlay.flags.players=false;overlay.actionFlags={attack:false,block:true,unknown_touch:true};
oldRecord.width=100;oldRecord.height=100;
oldRecord.replay_sha256=timeline.data.source_replay_sha256;
overlay.source.events=[
  {frame_index:45,ball_position_px:[20,20],action_prediction:{kind:"attack"}},
  {frame_index:45,ball_position_px:[30,30],rally_context:{estimated_kind:"block"}},
  {frame_index:45,ball_position_px:[40,40]}
];
actionColors.length=0;
renderedLabels.length=0;
scope.drawOverlay();
assert.deepEqual(actionColors,["#c38cff","#c6bf96"],
  "A subset preserves explicitly enabled unknown touches");
overlay.actionFlags.unknown_touch=false;
actionColors.length=0;scope.drawOverlay();
assert.deepEqual(actionColors,["#c38cff"],
  "Unknown touches disappear only when their own filter is disabled");
overlay.actionFlags.unknown_touch=true;
assert.ok(renderedLabels.some(label=>label.startsWith("≈ Блок · ")),"Block must be labeled as such with its provisional ID");
assert.ok(!renderedLabels.includes("Касание?"),"Legacy card touch labels are forbidden");
timeline.data.touch_annotations=[];
actionColors.length=0;
scope.drawOverlay();
assert.equal(actionColors.length,0,
  "Legacy card motion_events alone cannot create a Touch annotation");
const initStart=js.indexOf("  if(overlay.mode){");
const initEnd=js.indexOf("  const startOverlayFrameLoop=",initStart);
let changed,redraws=0,timelineRedraws=0,cardRedraws=0;
const checkbox={dataset:{overlayAction:"attack"},checked:false,addEventListener:(name,fn)=>{if(name==="change")changed=fn;}};
const input={addEventListener(){}};
const controls={querySelectorAll:()=>[checkbox],classList:{toggle(){},remove(){},add(){}}};
const rawClasses=new Set(["hidden"]);
const rawControls={classList:{
  toggle(name,force){if(force===false)rawClasses.delete(name);else if(force===true)rawClasses.add(name);else if(rawClasses.has(name))rawClasses.delete(name);else rawClasses.add(name);},
  add(name){rawClasses.add(name);},remove(name){rawClasses.delete(name);}
}};
const tab=()=>({attributes:{},closest:()=>controls,addEventListener(name,fn){this[name]=fn;},setAttribute(name,value){this.attributes[name]=value;}});
const trainingMode={active:false,panel:{classList:{add(){}}}};
const initOverlay={mode:tab(),rawMode:tab(),stabilizerMode:tab(),
  debugTrajectorySwitch:controls,debugTrajectoryLegend:controls,debugBefore:tab(),debugAfter:tab(),
  rawControls,actionControls:controls,canvas:{closest:()=>controls},
  rawBall:input,rawVballVisible:input,rawPeopleVisible:input,
  rawPlayer:input,rawPose:input,rawPoseVisible:input,
  flags:{touches:true},actionFlags:{attack:true,unknown_touch:true}};
runInNewContext(js.slice(initStart,initEnd),{
  overlay:initOverlay,$:()=>({classList:{remove(){}},disabled:false,addEventListener(){},value:"0.55"}),window:{addEventListener(){}},
  training:trainingMode,reference:{active:false,epoch:0,entries:[],picker:{classList:{add(){}}},analysis:{setAttribute(){}},toggle:{setAttribute(){}}},comparison:{epoch:0},
  annotation:{lane:{querySelectorAll:()=>[]}},hideAnnotationContext(){},trainingEditorReset(){},
  updateRawDetectorLabel(){},updateReferenceModeUi(){},placeTrainingTimeline(){},
  drawOverlay:()=>{redraws++;},drawTimeline:()=>{timelineRedraws++;},
  refreshRallyCardTouchTicks:()=>{cardRedraws++;},updateOverlayToggles:()=>{},
  setOverlaySource:()=>{},rawFusion:()=>({id:"f"}),s:{run:"",video:"",current:""},
  cards:()=>{},
  timeline:{status:{textContent:""},data:null},
  ensurePreviewVideo:()=>{},prefetchOverlayAtFrame:()=>{},overlayMessage:()=>{}
});
changed();
assert.equal(initOverlay.actionFlags.attack,false,"Checkbox disables the matching action");
assert.equal(redraws,1,"Changing the checkbox redraws the player");
assert.equal(timelineRedraws,1,"Changing the checkbox redraws timeline markers");
assert.equal(cardRedraws,1,"Changing the checkbox redraws rally card markers");
initOverlay.rawMode.click();
assert.equal(initOverlay.raw,true);
assert.equal(initOverlay.rawMode.attributes["aria-selected"],"true");
initOverlay.mode.click();
assert.equal(initOverlay.raw,false);
assert.equal(initOverlay.mode.attributes["aria-selected"],"true");
initOverlay.stabilizerMode.click();
assert.notEqual(initOverlay.stabilizerView,true,
  "The normal preview must not enter Ball Stabilizer comparison");
trainingMode.active=true;
initOverlay.stabilizerMode.click();
assert.equal(trainingMode.active,true,"Debug comparison must not leave Debug");
assert.equal(initOverlay.raw,true,"Debug comparison must request raw GPU observations");
assert.equal(initOverlay.stabilizerView,true);
assert.equal(initOverlay.stabilizerMode.attributes["aria-pressed"],"true");
assert.equal(rawClasses.has("hidden"),true,
  "Both must keep the GPU controls panel hidden");
assert.equal(initOverlay.debugBefore.attributes["aria-pressed"],"false");
assert.equal(initOverlay.debugAfter.attributes["aria-pressed"],"false");
initOverlay.debugAfter.click();
assert.equal(trainingMode.active,true);
assert.equal(initOverlay.raw,false,"After stabilization must restore CPU observations");
assert.equal(initOverlay.stabilizerView,false);
assert.equal(initOverlay.stabilizerMode.attributes["aria-pressed"],"false");
assert.equal(initOverlay.debugAfter.attributes["aria-pressed"],"true");
trainingMode.active=false;
assert.match(html,/id="studio-debug-trajectory-before"[^>]*aria-pressed="false"/);
assert.match(html,/id="studio-debug-trajectory-after"[^>]*aria-pressed="true"/);
initOverlay.debugBefore.click();
assert.equal(initOverlay.trajectoryVariant,"before");
assert.equal(initOverlay.debugBefore.attributes["aria-pressed"],"true");
initOverlay.debugAfter.click();
assert.equal(initOverlay.trajectoryVariant,"after");
assert.equal(initOverlay.debugAfter.attributes["aria-pressed"],"true");
// The actual presented timestamp wins over currentTime while decoding.
overlay.presentedTime=90/30-0.0007;
ui.video.currentTime=180/30;
overlay.flags.players=true;overlay.flags.touches=false;
const beforePresented=outlines;
scope.drawOverlay();
assert.equal(outlines,beforePresented+1,"Presentation timestamp must control boxes during decode lag");

const savedBallFrames=oldRecord.frames,oldArc=ctx.arc,oldLineTo=ctx.lineTo;
const marker=[],trail=[];
oldRecord.frames=[
  {frame:89,ball:[0.2,0.2],players:[]},
  {...savedBallFrames.find(row=>row.frame===90),ball:[0.4,0.5]}
];
ctx.arc=(px,py)=>marker.push([px,py]);
ctx.lineTo=(px,py)=>trail.push([px,py]);
overlay.flags.players=false;overlay.flags.trajectory=true;
scope.drawOverlay();
assert.deepEqual(marker,[[40,50]],"Ball marker must use the nearest analyzed sample of the actual presented frame");
assert.deepEqual(trail,[[40,50]],"Trajectory endpoint must use the same sample as the ball marker");
overlay.flags.players=true;overlay.flags.trajectory=false;
oldRecord.frames=savedBallFrames;ctx.arc=oldArc;ctx.lineTo=oldLineTo;
overlay.presentedTime=null;
ui.video.currentTime=90/30;ui.video.paused=true;ui.video.requestVideoFrameCallback=()=>{};
scope.drawOverlay();
assert.equal(outlines,beforePresented+1,"Do not draw from currentTime while waiting for a paused frame");
delete ui.video.requestVideoFrameCallback;delete ui.video.paused;
ui.video.currentTime=180/30;

overlay.presentedTime=null;
s.videoKey="p:other";
scope.drawOverlay();
assert.equal(outlines,beforePresented+1,"Never draw a new CPU version on media from another source");
s.videoKey="p:v";
overlay.source.events=[];
timeline.data.touch_annotations=[
  {frame_index:45,kind:"block",ball_position_px:[30,30],contact_verified:true}
];
overlay.flags.players=false;overlay.flags.touches=true;
ui.video.currentTime=45/30;
actionColors.length=0;
scope.drawOverlay();
assert.deepEqual(actionColors,["#c38cff"],
  "CPU timeline block survives changing the rally card and keeps its classification");
// Contacts are independently available while the large replay index is still
// loading. They must not wait for a player-box response.
const savedWindow=overlay.cache.get("p:cpu:r:v:0");
overlay.cache.delete("p:cpu:r:v:0");
ui.video.videoWidth=100;ui.video.videoHeight=100;
actionColors.length=0;
scope.drawOverlay();
assert.deepEqual(actionColors,["#c38cff"],
  "CPU contacts must render while the replay-window request is still outstanding");
overlay.cache.set("p:cpu:r:v:0",savedWindow);
// Opening a rally/seek primes both the immediate frame and the following
// overlapping window, rather than waiting for the first decoded video frame.
overlay.fetching.clear();
const requestsBefore=calls.length;
scope.prefetchOverlayAtFrame(180);
assert.equal(calls.length,requestsBefore,"Seeking coalesces overlay requests until debounce expires");
timers.pop()();
const early=calls.slice(requestsBefore);
assert.ok(early.some(path=>path.includes("start_frame=120")),
  "Seek should request the target player window after the debounce");
assert.ok(!early.some(path=>path.includes("start_frame=210")),
  "Seek should not immediately prefetch the next player window");
timeline.data.rally_run_id="wrong";
actionColors.length=0;
scope.drawOverlay();
assert.equal(actionColors.length,0,
  "Touch annotations from a foreign CPU version must never be displayed");
timeline.data.rally_run_id="r";
overlay.flags.touches=false;overlay.flags.players=true;
ui.video.currentTime=90/30;
timeline.data.source_replay_sha256="a".repeat(64);
oldRecord.replay_sha256="b".repeat(64);
const beforeDigest=outlines;
scope.drawOverlay();
assert.equal(outlines,beforeDigest,
  "A cached overlay from another replay SHA must never be drawn");
oldRecord.replay_sha256=timeline.data.source_replay_sha256;
scope.drawOverlay();
assert.equal(outlines,beforeDigest+1,
  "Matching source and replay SHA restores correct overlay display");

// Manually assigned names replace numbers, and stale CPU identities never leak.
s.videoKey="p:v";
timeline.data.rally_run_id="r";
timeline.data.source_replay_sha256="a".repeat(64);
oldRecord.replay_sha256=timeline.data.source_replay_sha256;
overlay.flags.players=true;
overlay.flags.touches=false;
ui.video.currentTime=90/30;
overlayLabels.key=scope.playerLabelKey("p","r","v",oldRecord.replay_sha256);
overlayLabels.loaded=true;
overlayLabels.names.set("1","Ильнур");
renderedLabels.length=0;
scope.drawOverlay();
assert.ok(renderedLabels.includes("Ильнур"),"Reviewed Track ID displays player name");
assert.ok(labelPositions.some(label=>label.value==="Ильнур"&&label.y>=80),"Player name appears below the box");
overlayLabels.names.clear();
renderedLabels.length=0;
scope.drawOverlay();
assert.ok(renderedLabels.includes("1"),"Unassigned Track ID keeps numeric fallback");
overlayLabels.names.set("1","Ильнур");
overlayLabels.key=scope.playerLabelKey("p","old-run","v",oldRecord.replay_sha256);
renderedLabels.length=0;
scope.drawOverlay();
assert.ok(renderedLabels.includes("1"),"Previous CPU run names cannot appear in new run");
overlayLabels.key=scope.playerLabelKey("p","r","v",oldRecord.replay_sha256);
overlayLabels.names.clear();

// A mismatched evidence duration must not let the seek bar run past EOF.
const durationStart=js.indexOf("  const timelineDuration=()=>{");
const durationEnd=js.indexOf("  const updateTimelineProgress=",durationStart);
assert.ok(durationStart>=0&&durationEnd>durationStart);
const durationScope={
  s:{project:"p",video:"v",videoKey:"p:v"},
  ui:{video:{duration:8.5}},
  timeline:{data:{total_frames:300,video_duration_sec:10}},
  fps:()=>30
};
runInNewContext(js.slice(durationStart,durationEnd)+
  "\nthis.timelineDuration=timelineDuration;",durationScope);
assert.equal(durationScope.timelineDuration(),8.5,
  "Decoded media duration, not maximum of estimates, must bound seeking");
durationScope.timeline.data=null;
assert.equal(durationScope.timelineDuration(),8.5,
  "Raw GPU-only preview must remain seekable even without a CPU timeline");

// Player labels are an independent, validated, source-bound read. A manual edit
// should refresh them without reloading or reprocessing the player boxes.
const labelsStart=js.indexOf("  const overlayLabels={");
const labelsEnd=js.indexOf("  const overlayWindowStart=",labelsStart);
assert.ok(labelsStart>=0&&labelsEnd>labelsStart);
let labelResponses=[];
let renderedOnNameChange=0;
const labelRuntime={
  request:()=>new Promise(resolve=>labelResponses.push(resolve)),
  url:(route)=>route,
  AbortController,
  s:{project:"p",run:"r",video:"v"},
  drawOverlay:()=>{renderedOnNameChange++;},
  overlayMessage:()=>{}
};
runInNewContext(js.slice(labelsStart,labelsEnd)+
  "\nthis.labelsApi={overlayLabels,requestOverlayLabels,invalidateOverlayLabels,playerLabelKey};",
  labelRuntime);
const labelsApi=labelRuntime.labelsApi;
const labelSha="a".repeat(64);
const labelResponse=(name,run="r",sha=labelSha)=>({
  schema:"volley-native-preview-player-labels/1",
  project_id:"p",rally_run_id:run,video:"v",
  source_replay_sha256:sha,identity_source:"manual",manual_track_ids:["1"],
  names:{"1":name}
});
const flush=()=>new Promise(resolve=>setImmediate(resolve));
labelsApi.requestOverlayLabels("p","r","v",labelSha);
labelsApi.requestOverlayLabels("p","r","v",labelSha);
assert.equal(labelResponses.length,1,"Name lookup must be deduplicated");
labelResponses.shift()(labelResponse("Ильнур"));
await flush();
assert.equal(labelsApi.overlayLabels.names.get("1"),"Ильнур");
assert.equal(renderedOnNameChange,1,"Loaded labels repaint existing frames");

labelsApi.invalidateOverlayLabels();
labelsApi.requestOverlayLabels("p","r","v",labelSha);
labelResponses.shift()(labelResponse("Ирек"));
await flush();
assert.equal(labelsApi.overlayLabels.names.get("1"),"Ирек",
  "A reviewed assignment change must refresh the name");

labelsApi.invalidateOverlayLabels();
labelsApi.requestOverlayLabels("p","r","v",labelSha);
const stale=labelResponses.shift();
labelsApi.invalidateOverlayLabels();
labelsApi.requestOverlayLabels("p","r","v",labelSha);
const current=labelResponses.shift();
stale(labelResponse("Неверный игрок"));
current(labelResponse("Ильнур"));
await flush();
assert.equal(labelsApi.overlayLabels.names.get("1"),"Ильнур",
  "A slow response from an invalidated assignment cannot overwrite a new one");

labelsApi.invalidateOverlayLabels();
labelsApi.requestOverlayLabels("p","r","v",labelSha);
labelResponses.shift()(labelResponse("Ильнур","r","b".repeat(64)));
await flush();
assert.equal(labelsApi.overlayLabels.names.size,0,
  "Foreign replay hashes must never produce athlete names");
console.log("Studio preview overlay frame continuity: OK");

const loopStart=js.indexOf("  const startOverlayFrameLoop=");
const loopEnd=js.indexOf('  window.addEventListener("resize",drawOverlay);',loopStart);
assert.ok(loopStart>=0&&loopEnd>loopStart);
const handlers=new Map(),callbacks=new Map();
let callbackId=0;
const media={paused:true,seeking:false,currentTime:1.04,
  addEventListener(name,fn){handlers.set(name,fn);},
  requestVideoFrameCallback(fn){callbacks.set(++callbackId,fn);return callbackId;},
  cancelVideoFrameCallback(id){callbacks.delete(id);}
};
const clockOverlay={loop:0,presentedTime:null};
const rendered=[];
runInNewContext(js.slice(loopStart,loopEnd),{
  ui:{video:media},overlay:clockOverlay,
  drawOverlay(){rendered.push(clockOverlay.presentedTime);},
  updateRallyCardProgress(){},overlayContext(){}
});
const emit=name=>handlers.get(name)?.();
const present=seconds=>{
  assert.equal(callbacks.size,1,"Exactly one video frame callback must be armed");
  const [id,callback]=callbacks.entries().next().value;
  callbacks.delete(id);callback(0,{mediaTime:seconds});
};
emit("loadedmetadata");present(1);
media.paused=false;emit("playing");present(1.033333);
media.currentTime=1.09;media.paused=true;emit("pause");
assert.equal(clockOverlay.presentedTime,1.033333,"Pause must preserve the frame actually displayed");
media.seeking=true;media.currentTime=2.09;emit("seeking");
assert.equal(clockOverlay.presentedTime,null);
media.seeking=false;emit("seeked");
assert.equal(clockOverlay.presentedTime,null,"Do not guess a frame from the seek target");
present(2.066667);
assert.equal(clockOverlay.presentedTime,2.066667,"Paused seek must capture presented mediaTime");
media.seeking=true;emit("seeking");present(3.0);
media.seeking=false;emit("seeked");
assert.equal(clockOverlay.presentedTime,3,"Seeked must not discard a frame already presented");
