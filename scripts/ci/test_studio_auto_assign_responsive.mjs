import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
assert.match(css,/\.track-auto-assign\.is-working/);
assert.match(js,/galleryAuto\.disabled=.*autoAssignInFlight/);
const start=js.indexOf("  const autoAssignSimilar=async(forcedThreshold)=>{");
const end=js.indexOf("  const refreshPhotoMatches=async(scope,token)=>{",start);
assert.ok(start>=0&&end>start);
const handler=js.slice(start,end);
assert.ok(!handler.includes("state.busy=true"),"Auto-link must not lock the whole player UI");
assert.ok(handler.includes('window.VolleyballTasks?.set("auto-reid"'));
assert.ok(handler.includes('window.VolleyballTasks?.clear("auto-reid"'));

let resolveRequest,rejectRequest;
let calls=0,refreshes=0,disabledUpdates=0;
const events=[],tasks=[],notices=[];
const state={busy:false,writable:true,total:2,roster:[{id:"athlete"}],project:"p",
  video:"v",sourceSha:"a".repeat(64),comparisonIds:new Set([7,8]),replay:"r",token:3,
  photoScoreLoading:false,photoScoresPromise:null,autoAssignPending:null,
  bindingEpoch:0,unassignedIds:new Set([7,8]),rows:[{track_id:7},{track_id:8}],
  filteredTotal:2,offset:0,limit:24,selected:new Set()};
const working=new Set();
const attrs=new Map();
const galleryAuto={classList:{add:v=>working.add(v),remove:v=>working.delete(v)},
  setAttribute:(key,val)=>attrs.set(key,val),removeAttribute:key=>attrs.delete(key)};
const galleryStatus={textContent:""};
const context={
  state,galleryAuto,galleryStatus,galleryAutoThreshold:{value:"80"},
  autoAssignScope:()=>JSON.stringify([state.project,state.video,state.replay,state.rallyRun,state.sourceSha]),
  autoAssignMessage:(message)=>{notices.push({msg:message})},
  reviewReady:()=>true,
  reviewScope:()=>({project_id:"p",video:"v",replay_id:"r"}),
  reidUnlock:()=>{disabledUpdates++},
  unlock:()=>{},
  reviewNote:(msg,error)=>notices.push({msg,error}),
  animateTrackCards:fn=>fn(),renderRows:()=>{},renderGroupCards:()=>{},updateUnassignedLabel:()=>{},
  loadPage:async()=>{refreshes++;assert.equal(state.unassignedIds.has(7),false,"Cards removed before refresh")},
  post:async(_,body)=>{calls++;assert.equal(body.auto_threshold_percent,80);return new Promise((resolve,reject)=>{
    resolveRequest=resolve;rejectRequest=reject;
  })},
  CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail}},
  window:{VolleyballTasks:{set:(id,task)=>tasks.push({id,task}),
    clear:id=>tasks.push({id,clear:true})},
    dispatchEvent:event=>events.push(event)}
};
runInNewContext("let autoAssignInFlight=false;\n"+handler+"\nthis.run=autoAssignSimilar;",context);
const running=context.run();
assert.equal(calls,1);
assert.equal(state.busy,false,"Unrelated player tools remain usable");
assert.equal(galleryAuto.disabled,undefined);
assert.equal(working.has("is-working"),true);
assert.equal(attrs.get("aria-busy"),"true");
assert.equal(tasks[0].id,"auto-reid");
assert.equal(tasks[0].task.progress,null,"Do not fabricate percentage");
assert.match(galleryStatus.textContent,/Можно продолжать работу/);
await context.run();
assert.equal(calls,1,"Double click must not launch duplicate requests");
state.token++; // A concurrent gallery read must not discard the committed mutation.
resolveRequest({automatic_count:2,threshold_percent:80,automatic_assignments:{"7":"athlete","8":"athlete"},unassigned_track_ids:[],identity_revision:"new"});
await running;
assert.equal(refreshes,1);
assert.equal(state.rows.length,0);
assert.equal(state.filteredTotal,0);
assert.equal(events.length,1);
assert.equal(tasks.at(-1).clear,true);
assert.equal(working.has("is-working"),false);
assert.equal(attrs.has("aria-busy"),false);
assert.ok(disabledUpdates>=2);
assert.match(galleryStatus.textContent,/привязано ID: 2/);

const failure=context.run();
rejectRequest(new Error("stale evidence"));
await failure;
assert.equal(state.busy,false);
assert.equal(tasks.at(-1).clear,true);
assert.equal(notices.at(-1).error,true);
assert.match(galleryStatus.textContent,/stale evidence/);

state.photoScoreLoading=true;
const waiting=context.run();
await waiting;
assert.equal(calls,2,"Autolink must reuse already running photo comparison");
assert.equal(state.autoAssignPending.threshold,80,"The threshold is preserved while queued");
assert.equal(state.busy,false);
state.photoScoreLoading=false;
const queued=state.autoAssignPending;
state.autoAssignPending=null;
const resumed=context.run(queued.threshold);
assert.equal(calls,3,"Only after existing photo comparison is done, assign");
resolveRequest({automatic_count:1,threshold_percent:80,automatic_assignments:{"7":"athlete"},unassigned_track_ids:[8],identity_revision:"newer"});
await resumed;
assert.equal(refreshes,2);
assert.equal(tasks.at(-1).clear,true);
console.log("Auto-link remains interactive, reuses ongoing comparison and reports sidebar lifecycle: OK");
