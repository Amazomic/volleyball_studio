import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const loadVideos=async()=>{");
const end=source.indexOf("  const publishTrackCovers=",start);
assert.ok(start>=0&&end>start);

const project={value:"project"},run={value:"a".repeat(32)},sourceText={};
const state={project:"",video:"",replay:"",rallyRun:"",fallback:false,rows:[],
  token:0,tracksLoaded:false,loadingKey:null};
const pending=[],notes=[],preserved=[];
let requests=0,pages=0;
const context={
  state,ui:{project,video:{},replay:{}},
  get:id=>id==="studio-run"?run:sourceText,
  reset:()=>{state.token++;state.rows=[];state.tracksLoaded=false;state.loadingKey=null;},
  loadLibraryRoster:async()=>{},
  options(){},
  note:(message,error=false)=>notes.push({message,error}),
  path:()=>"runs",
  read:()=>{requests++;return new Promise((resolve,reject)=>pending.push({resolve,reject}));},
  loadPage:async opts=>{
    pages++;preserved.push(Boolean(opts?.preserveCards));
    state.tracksLoaded=true;
    // The actual loadPage advances the same token and used to prevent cleanup.
    state.token++;
    notes.push({message:"Track ID загружены",error:false});
  },
  active:()=>true,
  queueMicrotask
};
runInNewContext(source.slice(start,end)+"\nthis.loadVideos=loadVideos;",context);
const cpu=runId=>({runs:[{
  video:"match.mp4",source_replay_id:"b".repeat(32),rally_run_id:runId
}]});
const tick=()=>new Promise(resolve=>setImmediate(resolve));

// A selection can change while the CPU index request is in flight.
// If the header doesn't dispatch another event, the old code left "Проверка CPU-источника…"
// on screen forever. The stale read must schedule one load for the current selection.
const first=context.loadVideos();
const duplicate=context.loadVideos();
assert.equal(requests,1,"duplicate in-flight requests are coalesced");
run.value="c".repeat(32);
pending[0].resolve(cpu("a".repeat(32)));
await Promise.all([first,duplicate]);
await tick();
assert.equal(requests,2,"stale selection starts a request for the new header state");
assert.equal(pages,0,"stale CPU index must never render old Track IDs");
pending[1].resolve(cpu(run.value));
await tick();
assert.equal(pages,1);
assert.equal(state.rallyRun,run.value);
assert.equal(state.loadingKey,null,"loadPage token changes must not leave loadingKey set");

// Re-entering the same tab should refresh current assignments without rereading the CPU index.
await context.loadVideos();
assert.equal(requests,2);
assert.equal(pages,2);
assert.equal(preserved.at(-1),true,"same CPU source refreshes current cards");
assert.equal(state.loadingKey,null);

// New CPU run does require a new source lookup.
run.value="d".repeat(32);
const changed=context.loadVideos();
assert.equal(requests,3);
pending[2].resolve(cpu(run.value));
await changed;
assert.equal(pages,3);
assert.equal(state.loadingKey,null);

// A transient API failure must not permanently lock out the Refresh button.
run.value="e".repeat(32);
const failed=context.loadVideos();
pending[3].reject(Error("CPU index temporarily unavailable"));
await failed;
assert.equal(state.loadingKey,null);
assert.equal(notes.at(-1).error,true);
const retried=context.loadVideos();
assert.equal(requests,5,"failed requests can be retried");
pending[4].resolve(cpu(run.value));
await retried;
assert.equal(pages,4);
console.log("Track CPU source selection races, refresh, and recovery: OK");
