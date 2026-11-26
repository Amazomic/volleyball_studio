import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const refreshPhotoMatches=async(scope,token,progressId)=>{");
const end=source.indexOf("  const reviewRender=",start);
assert.ok(start>=0&&end>start,"photo analysis action is defined");
const calls=[];
const state={project:"p",video:"v",replay:"r",rallyRun:"run",sourceSha:"a".repeat(64),
  token:1,busy:false,writable:true,photoScoreLoading:false,comparisonIds:new Set(Array.from({length:10},(_,i)=>i+1)),playerMatches:{}};
const context={state,reviewReady:()=>true,reviewScope:()=>({project_id:"p",video:"v",replay_id:"r",rally_run_id:state.rallyRun,expected_source_sha256:state.sourceSha}),
  unlock(){},reidUnlock(){},reviewNote(){},renderRows(){},renderGroupCards(){},
  galleryStatus:{textContent:""},
  post:async(url,body)=>{calls.push({url,body});return body.action==="refresh_matches"?
    {schema:"volley-native-player-photo-comparison/1",source_replay_sha256:state.sourceSha,auto_assign_enabled:false,confirmed:false,saved:true,identity_revision:"fresh-revision",player_matches:{"1":[]},photo_count:1}:{};}};
assert.ok(!source.includes("loadPhotoScoresOnFirstView"),"gallery must not infer ReID on opening");
runInNewContext(source.slice(start,end)+"\nthis.repeat=repeatPhotoAnalysis;",context);
await context.repeat();
assert.deepEqual(calls.map(call=>call.body.action),["refresh_matches"]);
assert.equal(calls[0].body.rally_run_id,"run");
assert.ok("1" in state.playerMatches);
assert.equal(state.busy,false);
assert.equal(state.similarityStatus,"calculated");
assert.equal(state.revision,"fresh-revision",
  "Saved photo matches must advance local revision before manual assignment");
assert.deepEqual(calls.map(call=>call.body.action),["refresh_matches"]);
state.comparisonIds=new Set(Array.from({length:2086},(_,i)=>i+1));
await context.repeat();
assert.deepEqual(calls.map(call=>call.body.action),["refresh_matches","refresh_matches"],
  "Large rally scope must reach the Rust photo matcher rather than a stale 1024-ID UI cap");
assert.equal(state.similarityStatus,"calculated");
assert.equal(state.revision,"fresh-revision",
  "Saved photo matches must advance local revision before manual assignment");
console.log("Photo similarity refresh also supports more than 1024 Track IDs: OK");
