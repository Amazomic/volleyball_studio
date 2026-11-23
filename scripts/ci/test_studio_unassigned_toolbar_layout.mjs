import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const tracks=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=tracks.indexOf("  const reidUnlock=()=>{");
const end=tracks.indexOf("  const autoAssignSimilar=async",start);
assert.ok(start>=0&&end>start,"Auto-assign controls remain testable");
const control=()=>({disabled:false,title:""});
const state={
  busy:false,assignmentPending:new Set([7]),writable:true,
  project:"project",video:"video.mp4",replay:"replay",sourceSha:"a".repeat(64),
  total:2,roster:[{id:"player"}],unassignedIds:new Set([7,8]),
  viewPlayer:"",selected:new Set(),comparisonIds:new Set([7,8]),
  photoScoreLoading:false,groupRevision:"revision",autoAssignPending:null
};
const autoAssignStatus={dataset:{kind:"blocked"}};
const messages=[];
const context={state,autoAssignInFlight:false,reviewReady:()=>true,
  reviewThreshold:control(),groupButton:control(),reviewRead:control(),
  reviewExtract:control(),galleryExtract:control(),galleryRepeat:control(),
  galleryAuto:control(),galleryAutoThreshold:control(),singlesExtract:control(),
  reviewMerge:control(),reviewSplit:control(),autoAssignStatus,
  autoAssignMessage:(text,kind="action")=>{
    messages.push([text,kind]);
    autoAssignStatus.dataset.kind=kind;
  }
};
runInNewContext(tracks.slice(start,end)+"\nthis.refresh=reidUnlock;",context);
context.refresh();
assert.equal(context.galleryAuto.disabled,true,"Pending manual assignment disables auto-assign");
assert.deepEqual(messages,[["","action"]],
  "Pending manual assignment clears prior notice instead of expanding the toolbar");
messages.length=0;
state.assignmentPending.clear();
context.refresh();
assert.deepEqual(messages,[],"Finishing assignment does not reintroduce a notice");
state.writable=false;
context.refresh();
assert.deepEqual(messages,[["Привязки недоступны для записи.","blocked"]],
  "Persistent write blockers still remain visible");

const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
const toolbar=css.match(/#studio-players-view \.track-auto-assign-controls\{([^}]*)\}/)?.[1];
assert.ok(toolbar,"Unassigned ID toolbar has a scoped layout rule");
for(const rule of ["width:100%","max-width:100%","min-width:0","box-sizing:border-box"])
  assert.ok(toolbar.includes(rule),`Toolbar must retain ${rule} while bindings change`);
assert.match(css,/#studio-players-view \.track-auto-assign-status\{[^}]*overflow-wrap:anywhere/,
  "Status messages cannot expand the toolbar with long text");
console.log("Track ID toolbar stays fixed during manual assignment: OK");
