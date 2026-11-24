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
  unassignedIds:new Set([101,102,103]),assignmentPending:new Set(),project:"project",video:"video",replay:"replay",rallyRun:"rally",
  selected:new Set()
};
const items=makeNode("div"),more=makeNode("button"),splits=[];
const context={
  state,galleryItems:items,galleryMore:more,
  clear:element=>{element.children=[];},
  node:makeNode,
  path:(url,args)=>url+"?"+new URLSearchParams(args),
  attachSuggestions(){},
  visibleUnassigned:id=>state.unassignedIds.has(id)&&!state.assignmentPending.has(id),
  sortBySimilarity:groups=>groups,
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
for(let i=0;i<4;i++)find("track-group-arrow-bottom").events.click();
assert.ok(photo.src.includes("photo_index=0"),"five photos per Track ID wrap to the first");
find("track-group-arrow-left").events.click();
assert.ok(photo.src.includes("track_id=101"));
find("track-group-arrow-right").events.click();
assert.ok(photo.src.includes("photo_index=0"),"each Track ID retains its own photo index");
find("track-group-detach").events.click();
assert.deepEqual(splits, [["split",102]],"detach splits only the visible ID");
context.renderGroupCards();
const preserved=items.children[0].children[0].children.find(x=>x.klass==="track-group-main-photo");
assert.ok(preserved.src.includes("track_id=102"),"card rendering preserves selected ID");
state.assignmentPending=new Set([101,102,103]);
context.renderGroupCards();
assert.match(items.children[0].text,/Нет групп с непривязанными/,
  "Pending group assignment removes its card without waiting for a response");

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
helperState.writable=true;
helperState.revision="outdated";
helper.post=async(url,body)=>{
  requests.push({url,body});
  return {schema:"volley-native-player-photo-comparison/1",source_replay_sha256:"sha",
    auto_assign_enabled:false,confirmed:false,saved:true,identity_revision:"new-revision",
    photo_count:4,player_matches:{"101":[]}};
};
await helper.refreshPhotoMatches({
  project_id:"project",video:"video",replay_id:"replay",
  rally_run_id:"rally",expected_source_sha256:"sha"},9);
assert.equal(requests.at(-1).body.action,"refresh_matches");
assert.equal(helperState.revision,"new-revision",
  "Matching refresh must synchronize the revision used by manual Track ID binding");

const candidateStart=source.indexOf("  const playerCandidates=ids=>{");
const candidateEnd=source.indexOf("  const renderRows=()=>{",candidateStart);
assert.ok(candidateStart>0&&candidateEnd>candidateStart);
const candidateState={
  roster:[{id:"athlete",name:"Player A"}],
  rows:[],unassignedReasons:{
    "101":"identity_conflict_or_ambiguity","102":"insufficient_observations",
    "103":"no_photo_match"},
  appliedAutoThreshold:75,
  playerMatches:{
    "101":[{player_id:"athlete",cosine_similarity:0.91,reference_kind:"photo"}],
    "102":[{player_id:"athlete",cosine_similarity:0.83,reference_kind:"photo"}]
  }
};
const cardWithReasons=makeNode("article");
const candidateContext={state:candidateState,node:makeNode,updateCardLink(){},bindSelected(){}};
runInNewContext(source.slice(candidateStart,candidateEnd)+
  "\nthis.attachSuggestions=attachSuggestions;",candidateContext);
candidateContext.attachSuggestions(cardWithReasons,[101,102,103]);
const panel=cardWithReasons.children[0];
const warning=panel.children.find(item=>item.klass.includes("track-auto-warning"));
assert.match(warning.text,/ID 101: конфликт/);
assert.match(warning.text,/ID 102: недостаточно кадров/);
assert.match(warning.text,/ID 103: нет совпадения/);
const controls=panel.children.find(item=>item.klass==="track-card-link-actions");
const select=controls.children[0];
assert.equal(select.children[0].text,"Выберите игрока",
  "A group with only two matching ID out of three must not silently select the athlete");
assert.match(select.children[1].text,/по 2 из 3 ID/);

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
