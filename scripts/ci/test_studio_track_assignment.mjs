import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const bindSelected=async (ids,playerId)=>{");
const end=source.indexOf("  // Native appearance matches",start);
assert.ok(start>=0&&end>start);
const calls=[],messages=[],events=[];
const sha="a".repeat(64);
const state={project:"p",video:"match.mp4",replay:"r",rallyRun:"run",sourceSha:sha,
  busy:false,rosterBusy:false,projectPlayerIds:new Set(["player"]),
  photoIndices:new Map([[7,2],[8,4]])};
const savedPhoto=(track_id,photo_index)=>({
  project_id:"p",video:"match.mp4",replay_id:"r",rally_run_id:"run",
  track_id,photo_index,source_replay_sha256:sha
});
let photos=[savedPhoto(8,4)],revision=0;
const context={state,unlock:()=>{},
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
await context.bindSelected([7,8,9],"player");
const enrolls=calls.filter(call=>call.body?.action==="enroll");
assert.equal(calls[0].action,"assign");
assert.deepEqual(calls[0].ids,[7,8,9]);
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
assert.match(messages.at(-1).message,/Сохранено фото: 2/);
assert.match(messages.at(-1).message,/уже было: 1/);
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
assert.equal(messages.at(-1).error,true);
assert.match(messages.at(-1).message,/ID 7: crop unavailable/);
assert.match(messages.at(-1).message,/Сохранено фото: 1/);

context.post=async(url,body)=>{
  if(body.action==="read")throw Error("album unavailable");
  throw Error("unexpected");
};
await context.bindSelected([7],"player");
assert.equal(messages.at(-1).error,true);
assert.match(messages.at(-1).message,/Track ID привязаны, но фото не сохранены/);
console.log("Group assignment enrolls distinct Track IDs with revision-safe dedup and partial failure recovery: OK");
