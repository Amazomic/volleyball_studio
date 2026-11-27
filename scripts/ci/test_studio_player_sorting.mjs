import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {previewWasmWindow} from "./helpers/preview_wasm.mjs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const playerCandidates=ids=>{");
const end=source.indexOf("  const attachSuggestions=",start);
assert.ok(start>0&&end>start,"similarity ranking helpers exist");
const state={roster:[{id:"a",name:"Аня"},{id:"b",name:"Борис"}],
  playerMatches:{"2":[{player_id:"a",cosine_similarity:.84}],
    "4":[{player_id:"b",cosine_similarity:.92}],
    "5":[{player_id:"b",cosine_similarity:.89}],
    "7":[{player_id:"a",cosine_similarity:.70}]}};
const context={state,window:previewWasmWindow()};
runInNewContext(source.slice(start,end)+
  "\nthis.sortBySimilarity=sortBySimilarity;this.visibleUnassigned=visibleUnassigned;this.visibleUnassignedCount=visibleUnassignedCount;",context);
const single=[{track_id:7},{track_id:9},{track_id:2},{track_id:4}];
assert.deepEqual(Array.from(context.sortBySimilarity(single,item=>[item.track_id]),x=>x.track_id),
  [4,2,7,9],"unassigned cards must be sorted by best player similarity, unknown scores last");
const grouped=[{ids:[2,7]},{ids:[4,5]},{ids:[8,9]}];
assert.deepEqual(Array.from(context.sortBySimilarity(grouped,item=>item.ids),x=>x.ids[0]),
  [4,2,8],"unassigned group cards must also rank by best player similarity");
state.assignmentPending=new Set([2,4,5]);
state.unassignedIds=new Set([2,4,5,7,8,9]);
assert.equal(context.visibleUnassigned([2]).length,0,"a clicked Track ID disappears before network response");
assert.equal(context.visibleUnassigned([7])[0],7,"unrelated Track IDs remain visible");
assert.equal(context.visibleUnassignedCount(),3,"pending assignments disappear from the unassigned counter");
state.assignmentPending.clear();
assert.equal(context.visibleUnassigned([2])[0],2,"a failed assignment can restore the card");
assert.equal(context.visibleUnassignedCount(),6,"failed assignments restore the unassigned counter");
assert.match(source,/animateTrackCards\(\(\)=>\{renderRows\(\);renderGroupCards\(\);unlock\(\);\}\);/);
console.log("Player similarity sorting and optimistic hide/restore: OK");
