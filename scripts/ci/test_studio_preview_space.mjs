import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(
  new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const playbackStart=source.indexOf("  const toggleVideoPlayback=()=>{");
const playbackEnd=source.indexOf("  playback.mode.addEventListener",playbackStart);
const shortcutStart=source.indexOf("  // Only text editing owns Space.");
const shortcutEnd=source.indexOf('  ui.video.addEventListener("keydown"',shortcutStart);
assert.ok(playbackStart>0&&playbackEnd>playbackStart);
assert.ok(shortcutStart>playbackEnd&&shortcutEnd>shortcutStart);

let visible=true,plays=0,pauses=0,scrubStops=0;
const s={videoKey:"p:match.mp4",playbackMode:"free",clipStart:null,clipEnd:null};
const video={
  paused:true,error:null,ended:false,currentTime:12,
  play(){this.paused=false;plays++;return Promise.resolve();},
  pause(){this.paused=true;pauses++;}
};
const handlers=[];
const context={
  s,ui:{video},overlay:{presentedTime:null},previewIsVisible:()=>visible,
  stopArrowScrub(){scrubStops++;},
  loadPreviewVideo(){throw Error("Unexpected media recovery");},
  message(error){throw Error(error);},
  document:{addEventListener:(type,handler,capture)=>handlers.push({type,handler,capture})}
};
runInNewContext(
  source.slice(playbackStart,playbackEnd)+source.slice(shortcutStart,shortcutEnd),
  context);
assert.equal(handlers.length,1);
assert.equal(handlers[0].type,"keydown");
assert.equal(handlers[0].capture,true,"Space must be intercepted before a focused button acts");

function target(tagName="DIV",type="",closestMatch=""){
  return {
    tagName,type,isContentEditable:false,
    closest(selector){return selector===closestMatch?{}:null;}
  };
}
function press(focus=target(),options={}){
  const event={
    code:"Space",key:" ",target:focus,
    altKey:false,ctrlKey:false,metaKey:false,isComposing:false,repeat:false,
    prevented:0,stopped:0,
    preventDefault(){this.prevented++;},
    stopPropagation(){this.stopped++;},
    ...options
  };
  handlers[0].handler(event);
  return event;
}

for(const focus of [
  target(),target("BUTTON"),target("SUMMARY"),
  target("INPUT","range"),target("INPUT","checkbox"),
  target("DIV","",".rally-card-progress"),target("VIDEO")
]){
  const before=video.paused;
  const result=press(focus);
  assert.equal(video.paused,!before,focus.tagName+" must toggle playback");
  assert.equal(result.prevented,1,"Space must not scroll or activate focused controls");
  assert.equal(result.stopped,1);
}
assert.equal(plays,4);
assert.equal(pauses,3);

const held=press(target("BUTTON"),{repeat:true});
assert.equal(held.prevented,1,"Repeated Space must suppress native button action");
assert.equal(held.stopped,1);
assert.equal(plays,4);
assert.equal(pauses,3,"Holding Space must not toggle video repeatedly");

for(const focus of [
  target("INPUT","text"),target("INPUT","search"),target("INPUT","number"),
  target("TEXTAREA"),target("SELECT"),
  {...target(),isContentEditable:true},
  target("DIV","","[contenteditable], [role=\"textbox\"]"),
  target("BUTTON","","dialog[open]")
]){
  const result=press(focus);
  assert.equal(result.prevented,0,"Text editing and dialogs retain Space");
}
for(const options of [
  {altKey:true},{ctrlKey:true},{metaKey:true},{isComposing:true},
  {code:"Escape",key:"Escape"}
]){
  assert.equal(press(target(),options).prevented,0);
}

visible=false;
assert.equal(press().prevented,0,"Hidden preview must not take the keyboard");
visible=true;
s.videoKey="";
assert.equal(press().prevented,0,"No source means no playback shortcut");
s.videoKey="p:match.mp4";
assert.equal(press(target(),{code:"Space",key:"Spacebar"}).prevented,1,
  "Code must work with alternate keyboard-layout key strings");
assert.equal(video.paused,true,"The next distinct press pauses playback");
assert.equal(plays,4);
assert.equal(pauses,4);
assert.equal(press().prevented,1);
assert.equal(video.paused,false,"Another Space press resumes playback");
assert.equal(plays,5);

s.playbackMode="once";s.clipStart=10;s.clipEnd=20;s.stopAt=20;
video.paused=true;video.currentTime=19.999;
press();
assert.equal(video.currentTime,19.999,"Space resumes a paused millisecond seek near the rally end");
assert.equal(video.paused,false);
assert.ok(scrubStops>0,"Playback toggle cancels any active arrow hold");
video.paused=true;s.stopAt=null;video.currentTime=20;
press();
assert.equal(video.currentTime,10,"A completed rally still replays from the start");

console.log("Preview Space: pause/resume over focused controls, text/dialog guards and key-repeat suppression: OK");
