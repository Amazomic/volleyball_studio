import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {previewWasmWindow} from "./helpers/preview_wasm.mjs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
assert.doesNotMatch(css,/native-track-photo-card\[data-bind-status/,"No saved-status badges on player cards");
const animationStart=js.indexOf("  const animateTrackCards=update=>{");
const animationEnd=js.indexOf("  const loadPage=async",animationStart);
const localStart=js.indexOf("  const applyLocalTrackEdit=");
const localEnd=js.indexOf("  const edit=async",localStart);
assert.ok(animationStart>0&&animationEnd>animationStart&&localStart>0&&localEnd>localStart);

const moves=[],ghosts=[],removed=[];
const makeCard=(id,index)=>({
  id,
  dataset:{trackCardKey:"single:"+id},
  style:{},
  getBoundingClientRect:()=>({left:index*232,top:100,width:220,height:280}),
  querySelectorAll:()=>[],
  animate:(frames,opts)=>{
    moves.push({id,frames,opts});
    return {finished:Promise.resolve()};
  },
  remove(){removed.push(id);}
});
let cards=[makeCard(7,0),makeCard(8,1),makeCard(9,2)];
const list={querySelectorAll:()=>cards};
const gallery={querySelectorAll:()=>[]};
const state={
  roster:[{id:"player",name:"Игрок"}],
  rows:[7,8,9].map(track_id=>({track_id,assigned_player_id:null})),
  unassignedIds:new Set([7,8,9]),excludedIds:new Set(),comparisonIds:new Set(),
  unassignedReasons:{},viewPlayer:"",viewConflicts:false,viewExcluded:false,
  filteredTotal:3,offset:0,limit:24
};
const context={
  state,ui:{list,assigned:{textContent:""},excluded:{textContent:""}},
  galleryItems:gallery,
  document:{
    body:{dataset:{}},
    getElementById:()=>({append:card=>ghosts.push(card.id)})
  },
  window:{...previewWasmWindow(),matchMedia:()=>({matches:false})},
  updateUnassignedLabel:()=>{},
  renderRows:()=>{cards=state.rows.map((row,index)=>makeCard(row.track_id,index));},
  renderGroupCards:()=>{},unlock:()=>{}
};
runInNewContext(js.slice(animationStart,animationEnd)+js.slice(localStart,localEnd)+
  "\nthis.update=applyLocalTrackEdit;",context);
context.update("assign",[7],"player",{
  assigned_count:1,excluded_count:0,unassigned_track_ids:[8,9],
  comparison_track_ids:[8,9]
});
assert.deepEqual(state.rows.map(row=>row.track_id),[8,9],"Confirmed card must disappear immediately");
assert.equal(state.filteredTotal,2);
assert.equal(ghosts.includes(7),true,"Departing card fades out in a detached overlay");
assert.ok(moves.some(item=>item.id===8&&
  item.frames[0].transform==="translate(232px, 0px)"),
  "Next card animates from previous grid position instead of jumping");
await Promise.resolve();
assert.ok(removed.includes(7),"Outgoing card is cleaned up after animation");

context.update("assign",[8],"player",{
  assigned_count:2,excluded_count:0,unassigned_track_ids:[9],
  comparison_track_ids:[9]
});
assert.deepEqual(state.rows.map(row=>row.track_id),[9]);
assert.equal(state.filteredTotal,1);
await Promise.resolve();
assert.ok(removed.includes(8));

context.document.body.dataset.uiMotion="reduce";
const beforeGhosts=ghosts.length;
context.update("assign",[9],"player",{
  assigned_count:3,excluded_count:0,unassigned_track_ids:[],
  comparison_track_ids:[]
});
assert.deepEqual(state.rows.map(row=>row.track_id),[]);
assert.equal(ghosts.length,beforeGhosts,"Reduced-motion mode must not animate ghosts");
console.log("Manual assignments remove cards, reflow the grid with FLIP and respect reduced motion: OK");
