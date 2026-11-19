import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const reidUnlock=()=>{");
const end=source.indexOf("  const refreshPhotoMatches=",start);
assert.ok(start>0&&end>start);
const state={
  project:"p",video:"match.mp4",replay:"r",rallyRun:"run",sourceSha:"a".repeat(64),
  token:1,busy:false,writable:true,total:3,roster:[{id:"playerA"}],
  photoScoreLoading:true,autoAssignPending:null,
  unassignedIds:new Set([1]),comparisonIds:new Set([1]),selected:new Set()
};
const controls=Object.fromEntries([
  "reviewThreshold","groupButton","reviewRead","reviewExtract","galleryExtract",
  "galleryRepeat","galleryAutoThreshold","singlesExtract","reviewMerge","reviewSplit"
].map(key=>[key,{disabled:true,value:key==="galleryAutoThreshold"?"92":"30",title:""}]));
const feedback={textContent:"",dataset:{},hidden:true};
const posted=[];
const button={disabled:true,title:"",classList:{add(){},remove(){}},setAttribute(){},removeAttribute(){}};
const context={
  autoAssignInFlight:false,galleryAuto:button,
  ...controls,state,autoAssignStatus:feedback,queueMicrotask,
  reviewReady:()=>true,reviewScope:()=>({
    project_id:state.project,video:state.video,replay_id:state.replay,
    rally_run_id:state.rallyRun,expected_source_sha256:state.sourceSha,threshold_percent:30
  }),
  autoAssignScope:()=>JSON.stringify([
    state.project,state.video,state.replay,state.rallyRun,state.sourceSha
  ]),
  autoAssignMessage(message,kind="action"){
    feedback.textContent=message;feedback.dataset.kind=kind;feedback.hidden=!message;
  },
  galleryStatus:{textContent:""},unlock(){},reviewNote(){},loadPage:async()=>{},
  window:{dispatchEvent(){}},CustomEvent:class {constructor(name){this.type=name}},
  post:async(_,payload)=>{
    posted.push(payload);
    return {automatic_count:1,automatic_assignments:{"101":"playerA"},
      threshold_percent:payload.auto_threshold_percent};
  }
};
runInNewContext(source.slice(start,end)+"\nthis.unlockReid=reidUnlock;this.run=autoAssignSimilar;",context);
context.unlockReid();
assert.equal(context.galleryAuto.disabled,false,"running background comparison must not block button");
assert.equal(context.galleryAutoThreshold.disabled,false,"slider remains interactive");
await context.run();
assert.equal(posted.length,0,"click queues instead of starting concurrent comparison");
assert.equal(state.autoAssignPending.threshold,92);
assert.match(feedback.textContent,/очередь/);
assert.equal(context.galleryAuto.disabled,true,"queued button cannot be clicked twice");
state.photoScoreLoading=false;
context.unlockReid();
await new Promise(resolve=>setImmediate(resolve));
assert.equal(posted.length,1);
assert.equal(posted[0].auto_threshold_percent,92);
assert.equal(state.autoAssignPending,null);
assert.equal(state.busy,false);
assert.match(feedback.textContent,/Привязано ID: 1/);
state.photoScoreLoading=true;
await context.run(87);
assert.equal(state.autoAssignPending.threshold,87);
state.video="unrelated.mp4";
state.photoScoreLoading=false;
context.unlockReid();
await new Promise(resolve=>setImmediate(resolve));
assert.equal(posted.length,1,"queued action must not run on another video");
console.log("Auto-assign enabled during background ReID; queue, duplicate prevention and scope safety: OK");
