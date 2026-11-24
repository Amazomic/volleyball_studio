import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const part=(start,end)=>{
  const from=js.indexOf(start),to=js.indexOf(end,from);
  assert.ok(from>=0&&to>from,"Missing executable preview section "+start);
  return js.slice(from,to);
};

const actionMarkup=html.match(/<fieldset id="studio-overlay-actions"[\s\S]*?<\/fieldset>/)?.[0]??"";
assert.equal((actionMarkup.match(/data-overlay-action=/g)||[]).length,6);
assert.match(css,/\.video-overlay-actions\{[^}]*justify-content:flex-start;[^}]*align-self:flex-start/);

const buttons=Object.fromEntries(["rallies","touches","trajectory","players","torso","pose","court"]
  .map(key=>[key,{setAttribute(){}}]));
let filterHidden=false;
const overlay={
  ...buttons,raw:false,
  flags:{rallies:true,touches:true,trajectory:true,players:true,torso:false,pose:true,court:true},
  actionControls:{classList:{toggle(_name,hidden){filterHidden=hidden;}}}
};
const toggleCtx={overlay};
runInNewContext(part("  const updateOverlayToggles=()=>{","  updateOverlayToggles();")+
  "\nthis.sync=updateOverlayToggles;",toggleCtx);
toggleCtx.sync();
assert.equal(filterHidden,false,"Filters appear for enabled processed touches");
overlay.flags.touches=false;toggleCtx.sync();
assert.equal(filterHidden,true,"Filters disappear when touches are disabled");
overlay.flags.touches=true;overlay.raw=true;toggleCtx.sync();
assert.equal(filterHidden,true,"Raw data never shows CPU touch filters");

const filterCtx={overlay};
runInNewContext(part("  const cpuTouchStyle={","  // The timeline belongs")+
  "\nthis.visible=kind=>overlay.actionFlags[cpuTouchKind(kind)]===true;this.cpuTouchStyle=cpuTouchStyle;this.cpuTouchKind=cpuTouchKind;",filterCtx);
overlay.raw=false;
overlay.actionFlags={unknown_touch:true,attack:true,block:true,serve:true,overhead_pass:true,dig:true};
assert.equal(filterCtx.visible("attack"),true);
assert.equal(filterCtx.visible(null),true,"Unclassified events appear with all types selected");
overlay.actionFlags.attack=false;
assert.equal(filterCtx.visible("attack"),false);
assert.equal(filterCtx.visible("serve"),true);
assert.equal(filterCtx.visible(null),true,"Unknown touches have an independent checkbox");
overlay.actionFlags.unknown_touch=false;
assert.equal(filterCtx.visible(null),false,"Disabling unknown hides only unclassified touches");

const paints=[];
const ctx={setTransform(){},clearRect(){},fillRect(x,y,w,h){paints.push(this.fillStyle);}};
const canvas={clientWidth:400,clientHeight:18,getContext:()=>ctx};
const events=[
  {frame_index:10,kind:"attack",contact_verified:true},
  {frame_index:20,kind:"serve",touch_attributed:true},
  {frame_index:30,kind:null,contact_verified:true}
];
const timeline={canvas,data:{rallies:[],touch_annotations:events}};
const dummy={hidden:false};
const timelineCtx={
  timeline,overlay,window:{devicePixelRatio:1},
  s:{analysisOffset:0},training:{active:false},fps:()=>10,
  timelineDuration:()=>10,updateTimelineProgress(){},showBallCandidates:()=>false,
  cpuTouchStyle:filterCtx.cpuTouchStyle,cpuTouchKind:filterCtx.cpuTouchKind,
  $:()=>dummy
};
runInNewContext(part("  const drawTimeline=()=>{","  const clearTimeline=()=>{")+
  "\nthis.draw=drawTimeline;",timelineCtx);
overlay.flags.touches=true;
paints.length=0;timelineCtx.draw();
assert.ok(!paints.includes("#ff6b6b"),"Attack markers obey the unchecked filter");
assert.ok(paints.includes("#f29d49"),"Selected serve remains on timeline");
assert.ok(!paints.includes("#c6bf96"),"Unclassified markers follow their own disabled filter");
overlay.actionFlags.attack=true;overlay.actionFlags.unknown_touch=true;
paints.length=0;timelineCtx.draw();
assert.ok(paints.includes("#ff6b6b"));
assert.ok(paints.includes("#c6bf96"),"All selected restores every touch type");
overlay.flags.touches=false;
paints.length=0;timelineCtx.draw();
assert.equal(paints.length,1,"Disabling touches removes all timeline markers");

assert.match(js,/overlay\.actionFlags\[input\.dataset\.overlayAction\]=input\.checked;\s*drawOverlay\(\);drawTimeline\(\);refreshRallyCardTouchTicks\(\);/);
assert.ok(js.includes('validTimeline&&(typeof training==="undefined"||!training.active)&&'),
  "CPU card ticks must be suppressed only in training, not in the standard preview");
console.log("Preview touch filter visibility and timeline/card redraw: OK");
