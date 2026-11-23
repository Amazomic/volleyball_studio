// Report UI contract: initial saved calculation, cached revisit and deliberate
// reassignment refresh. No synthetic Track IDs may become visible players.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import assert from "node:assert/strict";

const root=join(fileURLToPath(new URL("../..",import.meta.url)));
const source=readFileSync(join(root,"rust/volley-app/static/studio_stats.js"),"utf8");
const markup=readFileSync(join(root,"rust/volley-app/static/studio.html"),"utf8");
assert.match(markup,/id="studio-stats-recalculate"[^>]*>Пересчитать статистику<\/button>/);
class Element {
  constructor(tag="div"){
    this.tagName=tag;this.children=[];this.listeners=new Map();
    this.className="";this.classList={
      values:new Set(),add:s=>this.classList.values.add(s),
      remove:s=>this.classList.values.delete(s),
      toggle:(s,on)=>on?this.classList.values.add(s):this.classList.values.delete(s),
      contains:s=>this.classList.values.has(s)
    };
    this.value="";this.disabled=false;this.textContent="";
    this.style={setProperty:(name,value)=>{this.style[name]=String(value);}};
  }
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=[...items];}
  setAttribute(name,value){this[name]=String(value);if(name==="class")this.className=String(value);}
  addEventListener(name,fn){this.listeners.set(name,fn);}
  dispatch(name){return this.listeners.get(name)?.();}
}
const ids=["studio-project","studio-stats-recalculate","studio-stats-status",
  "studio-stats-summary","studio-stats-total","studio-stats-rallies",
  "studio-stats-attacks","studio-stats-blocks","studio-stats-serves","studio-stats-players","studio-stats-player-detail","studio-activity-legend"];
const nodes=new Map(ids.map(id=>[id,new Element()]));
nodes.get("studio-project").value="1234abcdef01";
const sha=c=>c.repeat(64);
const video="match.mp4", run="a".repeat(32);
const snapshot=(refreshed=false)=>({
  schema:"volley-native-player-statistics/1",
  algorithm:"verified-rally-roster-pose-contacts-v2",
  project_id:"1234abcdef01",source_fingerprint:sha("f"),
  video_count:1,roster_player_ids:["p1","p2"],
  sources:[{
    video,rally_run_id:run,source_replay_id:"b".repeat(32),
    source_replay_sha256:sha("c"),source_touch_id:"d".repeat(32),
    source_touch_sha256:sha("e"),source_rally_sha256:sha("1"),
    identity_revision:"absent",total_touch_candidates:8,initial_serve_candidates:2,unassigned_initial_serve_candidates:refreshed?0:1,
    unassigned_touch_candidates:refreshed?0:2
  }],
  total_touch_candidates:8,assigned_touch_candidates:refreshed?8:6,
  rally_candidates:2,initial_serve_candidates:2,unassigned_initial_serve_candidates:refreshed?0:1,mean_player_touches:refreshed?4:3,
  estimated_hits:refreshed?2:1,
  unassigned_touch_candidates:refreshed?0:2,
  players:[{
    player_id:"p1",name:"Player One",touches:6,activity:1,unclassified_touches:0,initial_serve_candidates:1,
    identity_source:"manually_reviewed_native_track_ids",
    estimated_actions:{attack:1,block:1,dig:1,serve:1,overhead_pass:2}
  },{
    player_id:"p2",name:"Player Two",touches:refreshed?2:0,
    activity:refreshed?0.8:0,unclassified_touches:refreshed?1:0,initial_serve_candidates:refreshed?1:0,
    identity_source:"photo_auto_or_reviewed_native_track_ids",
    estimated_actions:refreshed?{block:1}:{}
  }],
  raw_tracks:[{track_id:14,touches:2,confirmed_player_id:refreshed?"p2":null}],
  auto_player_identity_enabled:true,
  automatic_action_classification:true,
  automatic_action_classification_verified:false,
  estimated_actions:refreshed?{attack:1,block:2,dig:1}:{attack:1,block:1,dig:1},
  all_contacts_physically_verified:true,
  snapshot_persisted:true,saved_revision:sha(refreshed?"4":"3"),
  reused_saved_snapshot:!refreshed
});
let reassigned=false,saved=false,invalidCounts=false,classificationDisabled=false,albumUnavailable=false,onePhoto=false,loaded=0,recalculations=0;
const calls=[];
const fetch=async(url,options={})=>{
  calls.push({url,options});
  if(url==="/api/projects")return {ok:true,json:async()=>({
    projects:[{id:"1234abcdef01",videos:[video]}]
  })};
  if(url.startsWith("/api/studio/analysis-overview?")){
    return {ok:true,json:async()=>({
      schema:"volley-native-analysis-overview/1",
      metadata_only:true,project_id:"1234abcdef01",video,
      stages:[{key:"rallies",items:[{id:run,sha256:sha("2"),count:2}],truncated:false}]
    })};
  }
  if(url==="/api/players/library/native")return {ok:true,json:async()=>({players:[
    {id:"p1",name:"Player One",cover_reference_id:"a".repeat(24),saved_photo_count:3},
    {id:"p2",name:"Player Two",cover_reference_id:onePhoto?"d".repeat(24):null,saved_photo_count:onePhoto?1:0}
  ]})};
  if(url==="/api/players/native/references"){
    assert.deepEqual(JSON.parse(options.body),{project_id:"1234abcdef01",player_id:"p1",action:"read"});
    return albumUnavailable?{ok:false,status:503,json:async()=>({error:"album unavailable"})}:
      {ok:true,json:async()=>({photos:[{id:"b".repeat(24)},{id:"c".repeat(24)},{id:"a".repeat(24)}]})};
  }
  assert.equal(url,"/api/players/native/statistics");
  const payload=JSON.parse(options.body);
  assert.deepEqual(payload.sources,[{rally_run_id:run}]);
  let data;
  if(payload.mode==="open"){
    loaded++;
    assert.equal(payload.expected_revision,undefined);
    data=snapshot(saved);
    data.reused_saved_snapshot=loaded>1;
  }else if(payload.mode==="recalculate"){
    recalculations++;
    assert.equal(payload.expected_revision,sha("3"));
    assert.equal(payload.expected_source_fingerprint,undefined);
    assert.equal(reassigned,true);
    saved=true;
    data=snapshot(true);
  }else throw Error("Unexpected statistics mode: "+payload.mode);
  if(invalidCounts)data.players[0].touches=7;
  if(classificationDisabled)data.automatic_action_classification=false;
  return {ok:true,json:async()=>data};
};
const callbacks=new Map();
const overviewSource=readFileSync(join(root,"rust/volley-app/static/studio_overview.js"),"utf8");
const context={
  document:{getElementById:id=>nodes.get(id),createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)},
  window:{addEventListener:(name,cb)=>callbacks.set(name,cb),
    VolleyballStudioActivity:{fetch:(...args)=>fetch(...args)}},
  location:{hash:"#report"},fetch,URLSearchParams,Set,Map,JSON,Number,String,
  Array,Boolean,Error,Math,RegExp
};
vm.runInNewContext(overviewSource,context,{filename:"studio_overview.js"});
vm.runInNewContext(source,context,{filename:"studio_stats.js"});
const flush=async()=>{for(let i=0;i<6;i++)await new Promise(resolve=>setTimeout(resolve,0));};
const $=id=>nodes.get("studio-stats-"+id);
await flush();
assert.equal(loaded,1,"first report visit calculates and saves");
assert.equal($("recalculate").disabled,false);
assert.equal($("total").textContent,"6","only named players appear in summary");
assert.equal($("rallies").textContent,"2");
assert.equal($("serves").textContent,"2","serve proposals from preview must appear in report summary");
assert.equal($("players").children.length,2);
const first=$("players").children[0], second=$("players").children[1];
const find=(node,klass)=>node.className.split(/\s+/).includes(klass)?node:
  node.children.map(child=>find(child,klass)).find(Boolean);
const content=node=>[node.textContent,...node.children.map(content)].join(" ");
assert.match(content(first),/Player One/);
assert.match(content(second),/Player Two/);
assert.match(find(first,"report-player-avatar").children[0].src,/player_id=p1&reference_id=a{24}/);
assert.equal(find(second,"report-player-avatar").textContent,"P");
const avatar=find(first,"report-player-avatar");
assert.equal(avatar.tagName,"button","player photo must be a keyboard-accessible button");
await avatar.dispatch("click");
assert.match(avatar.children[0].src,/reference_id=b{24}/,"first click shows the next saved photo");
await avatar.dispatch("click");
assert.match(avatar.children[0].src,/reference_id=c{24}/);
await avatar.dispatch("click");
assert.match(avatar.children[0].src,/reference_id=a{24}/,"the album wraps to the first photo");
albumUnavailable=true;
await avatar.dispatch("click");
assert.match(avatar.children[0].src,/reference_id=a{24}/,"a failed album request preserves the visible photo");
albumUnavailable=false;
await avatar.dispatch("click");
assert.match(avatar.children[0].src,/reference_id=b{24}/,"photo switching recovers after an album error");
assert.match($("status").textContent,/обновлена/);
assert.ok(find(first,"report-action-attack"),"each player card shows attack statistics without a click");
assert.match(content(find(first,"report-action-attack")),/Атаки\s+1/);
assert.match(content(find(first,"report-action-block")),/Блоки\s+1/);
assert.match(content(find(first,"report-action-serve")),/≈ Подачи\s+1/);
assert.match(content(find(first,"report-action-dig")),/Приёмы\s+1/);
assert.match(content(find(first,"report-action-overhead_pass")),/Передачи\s+2/);
assert.match(content(find(second,"report-player-touch-count")),/0/);
assert.match(content(find(second,"report-action-serve")),/≈ Подачи\s+0/);
const firstActivity=find(first,"report-player-activity");
assert.ok(firstActivity,"player card displays an activity indicator");
assert.equal(firstActivity.style["--report-activity"],"1");
for(const tier of ["very-low","low","medium","high","very-high"]){
  const icon=find(firstActivity,"report-activity-"+tier);
  assert.ok(icon,"activity displays a distinct figure for "+tier);
  assert.equal(icon.tagName,"svg");
}
assert.match(content(firstActivity),/1×/);

// Manual Track ID assignment does not silently change saved report figures.
reassigned=true;onePhoto=true;
await callbacks.get("volley:tab-change")({detail:{tab:"report"}});
await flush();
assert.equal(loaded,2);
assert.equal(find($("players").children[1],"report-player-avatar").tagName,"div");
assert.match(find($("players").children[1],"report-player-avatar").children[0].src,/reference_id=d{24}/);
assert.equal(recalculations,0);
assert.equal($("total").textContent,"6");
await $("recalculate").dispatch("click");
await flush();
assert.equal(recalculations,1);
assert.equal($("total").textContent,"8");
assert.match(content(find($("players").children[1],"report-player-touch-count")),/2/);
assert.match(content(find($("players").children[1],"report-action-block")),/Блоки\s+1/);
assert.match(content(find($("players").children[1],"report-action-serve")),/≈ Подачи\s+1/);
assert.match($("status").textContent,/пересчитана/);
assert.equal($("recalculate").disabled,false);

// Reload reuses saved results, and tampered totals fail closed.
await callbacks.get("volley:tab-change")({detail:{tab:"report"}});
await flush();
assert.equal(loaded,3);
assert.equal($("total").textContent,"8");
classificationDisabled=true;
await callbacks.get("volley:tab-change")({detail:{tab:"report"}});
await flush();
assert.match(content(find($("players").children[0],"report-action-attack")),/Атаки\s+—/);
assert.match(content(find($("players").children[1],"report-player-touch-count")),/2/);
invalidCounts=true;
await callbacks.get("volley:tab-change")({detail:{tab:"report"}});
await flush();
assert.match($("status").textContent,/Количество касаний/);
assert.equal($("players").children.length,0);
assert.equal($("recalculate").disabled,true);
console.log("Native Report: roster-only, first-open save, cached revisits, manual ID refresh: OK");
