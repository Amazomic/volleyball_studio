import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const applyLocalTrackEdit=");
const end=source.indexOf("  // Native appearance matches",start);
assert.ok(start>0&&end>start,"Edit and enrollment must be owned by one handler");
let releaseAlbum;
const pausedAlbum=new Promise(resolve=>{releaseAlbum=resolve;});
const sha="a".repeat(64), assignments=[],referenceRequests=[],notes=[];
const state={
  project:"project",video:"match.mp4",replay:"source",rallyRun:"run",
  sourceSha:sha,revision:"rev0",busy:false,rosterBusy:false,writable:true,
  selected:new Set(),projectPlayerIds:new Set(["player"]),
  photoIndices:new Map(),roster:[{id:"player",name:"Игрок"}],
  rows:[{track_id:7,assigned_player_id:null},{track_id:8,assigned_player_id:null}],
  unassignedIds:new Set([7,8]),comparisonIds:new Set([7,8]),
  viewPlayer:"",viewConflicts:false,filteredTotal:2,offset:0,limit:24
};
let refreshes=0,enrolled=0;
const context={
  state,ui:{player:{value:"player"},assigned:{},excluded:{}},
  updateUnassignedLabel:()=>{},renderRows:()=>{},renderGroupCards:()=>{},
  animateTrackCards:callback=>callback(),unlock:()=>{},
  note:(message,error)=>notes.push([message,error]),
  loadPage:()=>{refreshes++;return new Promise(()=>{});},
  post:async(url,body)=>{
    if(url==="/api/players/native/tracks"){
      assert.equal(body.expected_revision,"rev"+assignments.length,
        "Next assignment must use the server-confirmed revision");
      assignments.push([...body.track_ids]);
      return {revision:"rev"+assignments.length,assigned_count:assignments.length,
        excluded_count:0,
        unassigned_track_ids:[7,8].filter(id=>!assignments.flat().includes(id)),
        comparison_track_ids:[7,8].filter(id=>!assignments.flat().includes(id))};
    }
    assert.equal(url,"/api/players/native/references");
    referenceRequests.push({...body});
    if(body.action==="read"){
      if(referenceRequests.filter(r=>r.action==="read").length===1)await pausedAlbum;
      return {revision:"album"+enrolled,photos:[]};
    }
    assert.equal(body.expected_revision,"album"+enrolled);
    enrolled++;
    return {revision:"album"+enrolled,photos:[]};
  },
  window:{dispatchEvent(){}},
  CustomEvent:class{constructor(type,opts){this.type=type;this.detail=opts?.detail;}},
  Event:class{constructor(type){this.type=type;}}
};
runInNewContext(source.slice(start,end)+"\nthis.bindSelected=bindSelected;",context);
const first=context.bindSelected([7],"player");
for(let i=0;i<15;i++)await Promise.resolve();
assert.deepEqual(assignments,[[7]]);
assert.equal(state.busy,false,"Photo enrollment must not hold the global assignment lock");
assert.equal(state.rows.some(row=>row.track_id===7),false,"Assigned card disappears immediately");
assert.equal(refreshes,1,"Gallery reconciliation is scheduled, never awaited");

const second=context.bindSelected([8],"player");
for(let i=0;i<15;i++)await Promise.resolve();
assert.deepEqual(assignments,[[7],[8]],
  "A second assignment must work before first photo enrollment completes");
assert.equal(state.busy,false);
assert.equal(state.rows.length,0,"Second card disappears without waiting for gallery reload");
assert.equal(referenceRequests.filter(row=>row.action==="read").length,1,
  "Album writes for one player must be serialized");

releaseAlbum();
await Promise.all([first,second]);
assert.equal(enrolled,2,"Both photos remain preserved");
assert.deepEqual(referenceRequests.filter(row=>row.action==="enroll")
  .map(row=>row.expected_revision),["album0","album1"]);
assert.ok(notes.some(row=>row[0].includes("можно продолжать работу")));
console.log("Manual bindings are nonblocking, revision-safe and retain all photos: OK");
