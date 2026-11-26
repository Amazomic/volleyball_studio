import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const photoEnrollmentQueues=new Map();");
const end=source.indexOf("  // Native appearance matches",start);
assert.ok(start>=0&&end>start);
const calls=[],messages=[],events=[];
const sha="a".repeat(64);
const state={project:"p",video:"match.mp4",replay:"r",rallyRun:"run",sourceSha:sha,unassignedIds:new Set(),
  busy:false,rosterBusy:false,projectPlayerIds:new Set(["player"]),assignmentPending:new Set(),
  photoIndices:new Map([[7,2],[8,4]]),selected:new Set()};
const savedPhoto=(track_id,photo_index)=>({
  project_id:"p",video:"match.mp4",replay_id:"r",rally_run_id:"run",
  track_id,photo_index,source_replay_sha256:sha
});
let photos=[savedPhoto(8,4)],revision=0;
let rendered=0;
const context={state,unlock:()=>{},showBindingStatus:()=>{},
  animateTrackCards:update=>update(),renderRows:()=>{rendered++;},renderGroupCards(){},updateUnassignedLabel(){},
  edit:async(action,ids,player)=>{
    calls.push({action,ids,player});
    // The selected gallery disappears after assignment.
    state.photoIndices.clear();
    return true;
  },
  post:async(url,body)=>{
    calls.push({url,body});
    if(body.action==="read")return {revision:"rev"+revision,photos:[...photos]};
    assert.equal(body.expected_revision,"rev"+revision);
    photos.push(savedPhoto(body.track_id,body.photo_index));
    return {revision:"rev"+ ++revision,photos:[...photos]};
  },
  note:(message,error)=>messages.push({message,error}),
  window:{dispatchEvent:event=>events.push(event.type)},
  Event:class{constructor(type){this.type=type;}}};
runInNewContext(source.slice(start,end)+"\nthis.bindSelected=bindSelected;",context);
const firstAssignment=context.bindSelected([7,8,9],"player");
assert.equal(rendered,1,"Manual assignment immediately redraws cards before saving");
await firstAssignment;
const enrolls=calls.filter(call=>call.body?.action==="enroll");
assert.equal(calls[0].action,"assign");
assert.deepEqual(Array.from(calls[0].ids),[7,8,9]);
assert.equal(calls[1].body.action,"read");
assert.deepEqual(enrolls.map(call=>call.body.track_id),[7,9],
  "Every unsaved group member must contribute an image; existing ID 8 is skipped");
assert.deepEqual(enrolls.map(call=>call.body.photo_index),[2,0],
  "Use the picture selected for each distinct Track ID");
assert.deepEqual(enrolls.map(call=>call.body.expected_revision),["rev0","rev1"],
  "Refresh the album revision after each append");
assert.ok(enrolls.every(call=>call.body.rally_run_id==="run"&&
  call.body.expected_source_sha256===sha));
assert.deepEqual(events,["volley:player-photo-saved"]);
assert.match(messages.at(-1).message,/Эталоны: 2/);
assert.match(messages.at(-1).message,/уже были: 1/);
assert.equal(state.busy,false);

calls.length=0;
await context.bindSelected([9],"other");
assert.equal(calls.length,0,"Only players in this project may be assigned");
context.edit=async()=>false;
await context.bindSelected([9],"player");
assert.equal(calls.length,0,"A failed assignment must not save references");

context.edit=async()=>true;
context.post=async(url,body)=>{
  calls.push({url,body});
  if(body.action==="read")return {revision:"rev0",photos:[]};
  if(body.track_id===7)throw Error("crop unavailable");
  return {revision:"rev1",photos:[savedPhoto(8,0)]};
};
calls.length=0;
await context.bindSelected([7,8],"player");
assert.deepEqual(calls.filter(call=>call.body?.action==="enroll")
  .map(call=>call.body.track_id),[7,8],
  "A failed photo must not abort saving the remaining group");
assert.equal(messages.at(-1).error,false,
  "A failed reference crop must not label a successful assignment as failed");
assert.match(messages.at(-1).message,/ID 7: crop unavailable/);
assert.match(messages.at(-1).message,/Эталоны: 1/);

context.post=async(url,body)=>{
  if(body.action==="read")throw Error("album unavailable");
  throw Error("unexpected");
};
await context.bindSelected([7],"player");
assert.equal(messages.at(-1).error,undefined);
assert.match(messages.at(-1).message,/Привязка ID сохранена. Эталонные фото не добавлены/);
context.post=async(url,body)=>{
  calls.push({url,body});
  if(body.action==="read")return {revision:"rev0",photos:[]};
  return {revision:"rev"+calls.length,photos:[]};
};
calls.length=0;
await context.bindSelected([7,8,9,10,11,12],"player");
assert.equal(calls.filter(call=>call.body?.action==="enroll").length,3,
  "Group of many fragments must not create a reference photo for every ID");
context.post=async(url,body)=>{
  calls.push({url,body});
  if(body.action==="read")return {revision:"rev0",photos:Array.from({length:12},(_,i)=>savedPhoto(i+100,0))};
  throw Error("Automatic reference ceiling was ignored");
};
calls.length=0;
await context.bindSelected([7,8,9],"player");
assert.equal(calls.filter(call=>call.body?.action==="enroll").length,0,
  "Completed album must not be expanded on every manual Track ID binding");
const refreshStart=source.indexOf('    refreshReferences.addEventListener("click",async()=>{');
const refreshEnd=source.indexOf('    const resetLinks=node(',refreshStart);
assert.ok(refreshStart>0&&refreshEnd>refreshStart);
let releasePhotos;
const pendingPhotos=new Promise(resolve=>{releasePhotos=resolve;});
const queued=new Map([["player",pendingPhotos]]);
const refreshRequests=[];
const refreshTasks=[];
let onRefresh;
const refreshButton={disabled:false,addEventListener(name,callback){if(name==="click")onRefresh=callback;}};
const refreshStatus={hidden:true,textContent:""};
const refreshState={busy:false,rosterBusy:false,project:"p",video:"match.mp4",replay:"r",
  rallyRun:"run",sourceSha:sha,token:11,writable:true};
runInNewContext(source.slice(refreshStart,refreshEnd),{
  refreshReferences:refreshButton,refreshStatus,state:refreshState,
  photoEnrollmentQueues:queued,
  post:async url=>{refreshRequests.push(url);return {added:1,confirmed_tracks:1};},
  window:{dispatchEvent(){},crypto:{getRandomValues:a=>a.fill(1)},setTimeout(){return 1;},clearTimeout(){},VolleyballTasks:{
    set(key,task){refreshTasks.push({key,...task});},
    clear(key){refreshTasks.push({key,cleared:true});}
  }},Event:class{},
  loadLibraryRoster:async()=>{},
  refreshPhotoMatches:async()=>{refreshRequests.push("matches");return {trackCount:3};},
  autoAssignSimilar:async()=>{refreshRequests.push("auto_assign");}
});
const refreshing=onRefresh();
await new Promise(resolve=>setImmediate(resolve));
assert.deepEqual(refreshRequests,[],"Feature refresh must wait for pending manual photo saves");
assert.equal(refreshStatus.hidden,true,"Processing status is shown in the shared task panel");
assert.ok(refreshTasks.some(task=>/Ожидание сохранения фотографий/.test(task.stage||"")));
releasePhotos();
await refreshing;
assert.deepEqual(refreshRequests,["/api/players/native/references/refresh","matches","auto_assign"],
  "Refreshing reference photos must finish comparison before automatic assignment");
assert.match(refreshStatus.textContent,/Сопоставлено ID: 3/);
assert.ok(refreshTasks.some(task=>task.key==="player-references-refresh"&&task.cleared));

console.log("Group assignment preserves links, bounds photo enrollment and handles reference failures: OK");

const editStart=source.indexOf("  const edit=async");
const editEnd=source.indexOf("  // Multiple photo saves",editStart);
assert.ok(editStart>0&&editEnd>editStart);
const editState={project:"p",video:"v",replay:"r",rallyRun:"run",revision:"old",
  busy:false,selected:new Set(),offset:0,limit:24,pendingEditFailure:""};
const editCalls=[],editNotes=[];
let editReloads=0;
const editContext={state:editState,ui:{player:{value:"player"}},
  read:async()=>({revision:"fresh"}),
  path:(_,args)=>JSON.stringify(args),
  post:async(_,payload)=>{
    editCalls.push({...payload});
    if(payload.expected_revision==="old")throw Error("Track bindings changed; refresh the revision");
    return {revision:"saved",unassigned_track_ids:[],displaced_automatic_track_ids:[]};
  },
  note:(message,error)=>editNotes.push({message,error}),
  unlock:()=>{},applyLocalTrackEdit:()=>{},
  loadPage:async()=>{editReloads++;},
  window:{dispatchEvent(){}},CustomEvent:class{constructor(){}}
};
runInNewContext(source.slice(editStart,editEnd)+"\nthis.edit=edit;",editContext);
assert.equal(await editContext.edit("assign",[13],"player"),true);
assert.deepEqual(editCalls.map(call=>call.expected_revision),["old","fresh"],
  "A concurrent auto-assignment must not reject manual confirmation on a stale revision");
assert.equal(editState.revision,"saved");
assert.equal(editState.pendingEditFailure,"");
assert.equal(editReloads,1);

editContext.post=async()=>{throw Error("Identity constraint: ID 7 conflicts with ID 3");};
assert.equal(await editContext.edit("assign",[7],"player"),false);
assert.match(editState.pendingEditFailure,/ID 7: Identity constraint: ID 7 conflicts with ID 3/);
const noteStart=source.indexOf("  const note=(value,error=false)=>{");
const noteEnd=source.indexOf("  const options=(",noteStart);
const stickyStatus={textContent:"",classList:{toggle(_,error){this.error=error;}}};
const stickyContext={state:editState,ui:{status:stickyStatus}};
runInNewContext(source.slice(noteStart,noteEnd)+"\nthis.note=note;",stickyContext);
stickyContext.note("New gallery results");
assert.match(stickyStatus.textContent,/ID 7: Identity constraint/);
assert.equal(stickyStatus.classList.error,true,
  "The gallery refresh must not erase the last manual assignment failure");
