import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const from=source.indexOf("  const renderGroupCards=()=>{");
const to=source.indexOf('  galleryMore.addEventListener',from);
assert.ok(from>=0&&to>from);

const makeNode=(tag,klass="",text="")=>({
  tag,klass,text,children:[],dataset:{},events:{},
  append(...children){this.children.push(...children);},
  addEventListener(name,listener){this.events[name]=listener;},
  setAttribute(name,value){this[name]=value;}
});
const state={
  grouping:false,groupLoaded:true,groupShown:24,busy:false,writable:true,groupRevision:"revision",
  groupNavigation:new Map(),photoIndices:new Map(),groups:[{ids:[101,102,103],source:"auto",averageSimilarity:.83}],
  unassignedIds:new Set([101,102,103]),project:"project",video:"video",replay:"replay",rallyRun:"rally",
  selected:new Set()
};
const items=makeNode("div"),more=makeNode("button"),splits=[];
const context={
  state,galleryItems:items,galleryMore:more,
  clear:element=>{element.children=[];},
  node:makeNode,
  path:(url,args)=>url+"?"+new URLSearchParams(args),
  attachSuggestions(){},
  reviewEdit:async(action,ids)=>{splits.push([action,...ids]);}
};
runInNewContext(source.slice(from,to)+"\nthis.renderGroupCards=renderGroupCards;",context);
context.renderGroupCards();
const card=items.children[0];
const mosaic=card.children[0],photo=mosaic.children.find(x=>x.klass==="track-group-main-photo");
const find=klass=>mosaic.children.find(x=>x.klass.includes(klass));
assert.equal(mosaic.children.filter(x=>x.klass.startsWith("track-group-arrow ")).length,4);
assert.ok(photo.src.includes("track_id=101"));
assert.ok(photo.src.includes("photo_index=0"));

find("track-group-arrow-right").events.click();
assert.ok(photo.src.includes("track_id=102"),"right moves to the next Track ID");
find("track-group-arrow-bottom").events.click();
assert.ok(photo.src.includes("photo_index=1"),"down moves to the next photo of this ID");
find("track-group-arrow-left").events.click();
assert.ok(photo.src.includes("track_id=101"));
find("track-group-arrow-right").events.click();
assert.ok(photo.src.includes("photo_index=1"),"each Track ID retains its own photo index");
find("track-group-detach").events.click();
assert.deepEqual(splits, [["split",102]],"detach splits only the visible ID");
context.renderGroupCards();
const preserved=items.children[0].children[0].children.find(x=>x.klass==="track-group-main-photo");
assert.ok(preserved.src.includes("track_id=102"),"card rendering preserves selected ID");

const helperStart=source.indexOf("  const refreshPhotoMatches=async(");
const helperEnd=source.indexOf("  const repeatPhotoAnalysis",helperStart);
assert.ok(helperStart>=0&&helperEnd>helperStart);
const requests=[];
const helperState={
  sourceSha:"sha",replay:"replay",token:9,comparisonIds:new Set([101,102]),
  playerMatches:{"101":[{player_id:"old"}]}
};
const helper={
  state:helperState,
  post:async(url,body)=>{
    requests.push({url,body});
    return {schema:"volley-native-player-photo-comparison/1",source_replay_sha256:"sha",
      auto_assign_enabled:false,confirmed:false,photo_count:7,
      player_matches:{"102":[{player_id:"new",cosine_similarity:.92}]}};
  },
  renderRows(){},renderGroupCards(){}
};
runInNewContext(source.slice(helperStart,helperEnd)+"\nthis.refreshPhotoMatches=refreshPhotoMatches;",helper);
const summary=await helper.refreshPhotoMatches({
  project_id:"project",video:"video",replay_id:"replay",
  rally_run_id:"rally",expected_source_sha256:"sha"},9);
assert.equal(requests.length,1);
assert.equal(requests[0].body.action,"compare_photos");
assert.equal(summary.photoCount,7);
assert.equal(helperState.playerMatches["101"],undefined,"old suggestions are replaced");
assert.equal(helperState.playerMatches["102"][0].cosine_similarity,.92);
const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
assert.match(css,/#studio-players-view \.track-group-arrow-top\{left:8px;top:8px\}/,
  "previous photo control stays in the upper-left corner");
assert.match(css,/#studio-players-view \.track-group-arrow-bottom\{left:8px;top:52px\}/,
  "next photo control stacks below it, away from the face");
assert.match(css,/#studio-players-view \.track-group-arrow::before\{/,
  "arrows use centered CSS chevrons instead of font glyph positioning");
assert.match(css,/transition:scale \.18s ease/,
  "photo navigation has a smooth hover/press animation");
assert.match(css,/@media\(prefers-reduced-motion:reduce\)/,
  "button animation respects reduced-motion preferences");
console.log("ReID photo/ID navigation, controls layout, and refreshed photo matches: OK");
