import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {join} from "node:path";
import {fileURLToPath} from "node:url";

const root=join(fileURLToPath(new URL("../..",import.meta.url)));
const source=readFileSync(join(root,"rust/volley-app/static/studio_tracks.js"),"utf8");
const selector=source.match(/  const playerCandidates=ids=>\{[\s\S]*?\n  \};/)?.[0];
assert.ok(selector,"Track ID suggestion function must remain testable");

const roster=Array.from({length:6},(_,i)=>({id:"player-"+i,name:"Игрок "+i}));
const scores=[0.58,0.55,0.43,0.40,0.71,0.13];
const state={roster,playerMatches:{"206":scores.map((score,i)=>({
  player_id:roster[i].id,cosine_similarity:score
}))}};
const candidates=Function("state",selector+"; return playerCandidates;")(state)([206]);
assert.equal(candidates.length,6,"All athletes with measured scores must be visible");
assert.equal(candidates[0].playerId,"player-4","Best match is always selected first");
assert.deepEqual(candidates.map(row=>row.score),[0.71,0.58,0.55,0.43,0.40,0.13]);
assert.match(source,/action:state\.writable\?"refresh_matches":"compare_photos"/,
  "Writable refresh must persist scores; read-only review stays available");
console.log("Track ID candidate ranking and persistence action: OK");
