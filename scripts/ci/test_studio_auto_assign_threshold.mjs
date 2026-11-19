import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
for(const snippet of [
  'galleryAutoThreshold.type="range";galleryAutoThreshold.min="1";galleryAutoThreshold.max="99"',
  'galleryAutoThreshold.value="80"',
  'autoAssignControls.append(galleryAutoLabel,galleryAuto,autoAssignStatus)',
  'ui.list.before(autoAssignControls)',
  'autoAssignControls.hidden=mode==="player"'
])assert.ok(source.includes(snippet),"Missing slider UI: "+snippet);
const start=source.indexOf("  const autoAssignSimilar=async(forcedThreshold)=>{");
const end=source.indexOf("  const refreshPhotoMatches=",start);
assert.ok(start>0&&end>start);

const requests=[];
const events=[];
const state={
  project:"p",video:"video.mp4",replay:"r",token:4,
  sourceSha:"a".repeat(64),rallyRun:"run",busy:false,writable:true,total:3,
  roster:[{id:"playerA"}],photoScoreLoading:false
};
const slider={value:"92"};
const status={textContent:""};
const button={disabled:true,title:"",classList:{add(){},remove(){}},setAttribute(){},removeAttribute(){}};
const context={
  autoAssignInFlight:false,galleryAuto:button,
  state,galleryAutoThreshold:slider,galleryStatus:status,
  autoAssignScope:()=>JSON.stringify([state.project,state.video,state.replay,state.rallyRun,state.sourceSha]),
  autoAssignMessage:()=>{},
  reviewReady:()=>true,reviewScope:()=>({project_id:"p",video:"video.mp4",replay_id:"r",
    expected_source_sha256:state.sourceSha,threshold_percent:30}),
  window:{dispatchEvent(event){events.push(event)},CustomEvent:function(name,options){
    this.type=name;this.detail=options.detail;}},
  CustomEvent:function(name,options){this.type=name;this.detail=options.detail;},
  unlock(){},reidUnlock(){},reviewNote(){},loadPage:async()=>{},
  post:async(url,body)=>{
    requests.push({url,body});
    return {automatic_count:1,automatic_assignments:{"101":"playerA"},
      threshold_percent:body.auto_threshold_percent,unassigned_track_ids:[102,103]};
  }
};
runInNewContext(source.slice(start,end)+"\nthis.run=autoAssignSimilar;",context);
await context.run();
assert.equal(requests.length,1);
assert.equal(requests[0].url,"/api/players/native/reid-review");
assert.equal(requests[0].body.action,"auto_assign");
assert.equal(requests[0].body.auto_threshold_percent,92);
assert.equal(state.busy,false);
assert.match(status.textContent,/92%/);
assert.match(status.textContent,/остальные остались непривязанными/);
assert.equal(events.length,1);

slider.value="35";
await context.run();
assert.equal(requests[1].body.auto_threshold_percent,35);
slider.value="0";
await context.run();
assert.equal(requests.length,2,"invalid threshold is rejected before saving");
console.log("Adjustable auto-assign slider, API payload, result, and unassigned behavior: OK");
