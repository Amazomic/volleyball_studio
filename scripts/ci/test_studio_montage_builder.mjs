import {readFileSync} from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";

const text=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const html=text("rust/volley-app/static/studio.html");
const source=text("rust/volley-app/static/studio_montage_builder.js");
assert.ok(html.includes('src="/studio/montage-builder.js"'));
assert.ok(text("rust/volley-app/src/studio_web.rs").includes('include_str!("../static/studio_montage_builder.js")'));
class Element {
  constructor(tag="div") {
    this.tagName=tag;this.children=[];this.listeners=new Map();
    this.value="";this.disabled=false;this.checked=false;this.files=[];
    this.textContent="";this.className="";this.currentTime=0;this.readyState=1;
    this.attributes=new Map();this.paused=true;
    const classes=new Set();
    this.classList={add:name=>classes.add(name),remove:name=>classes.delete(name),
      contains:name=>classes.has(name),toggle:(name,on)=>on?classes.add(name):classes.delete(name)};
  }
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=[...items];this.textContent="";}
  addEventListener(name,fn){this.listeners.set(name,fn);}
  async fire(name,event={target:this}){await this.listeners.get(name)?.(event);await tick();}
  querySelectorAll(){return [];}
  setAttribute(name,value){this.attributes.set(name,String(value));}
  getAttribute(name){return this.attributes.get(name)||null;}
  removeAttribute(name){this.attributes.delete(name);}
  load(){this.loads=(this.loads||0)+1;}
  pause(){this.paused=true;}
  async play(){this.paused=false;this.plays=(this.plays||0)+1;}
  click(){this.clicks=(this.clicks||0)+1;}
}
const nodes=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element()]));
const get=id=>{assert.ok(nodes.has(id),"Missing actual DOM node "+id);return nodes.get(id);};
const montage=id=>get("studio-montage-"+id);
const tick=async()=>{for(let i=0;i<6;i++)await new Promise(resolve=>setImmediate(resolve));};
get("studio-project").value="1234abcdef01";
get("studio-encoder").value="cpu";get("studio-volume").value="40";
montage("zoom").value="1.25";
const run="a".repeat(32),otherRun="b".repeat(32),job="montage_"+"1".repeat(32),audio="track_"+"d".repeat(32)+".wav";
get("studio-run").value=run;
let requests=[],jobState="running",clips=2,invalidFingerprint=false,wrongProject=false,failPreview=false;
let expectedRun=run,creativeSourcesExpanded=false;
const reply=(data,ok=true)=>({ok,status:ok?200:409,json:async()=>data});
const fetch=async(url,options={})=>{
  const body=typeof options.body==="string"?JSON.parse(options.body):null;
  requests.push({url,body});
  if(url==="/api/projects")return reply({projects:[{
    id:"1234abcdef01",videos:creativeSourcesExpanded?["match.mp4","second.mp4"]:["match.mp4"],player_ids:["athlete1"]
  }]});
  if(url==="/api/players/library/native")return reply({players:[
    {id:"athlete1",name:"Athlete 1"},{id:"other",name:"Other athlete"}
  ]});
  if(url.startsWith("/api/studio/analysis-overview?"))return reply({
    schema:"volley-native-analysis-overview/1",project_id:"1234abcdef01",video:new URLSearchParams(url.split("?")[1]).get("video"),metadata_only:true,
    stages:[{key:"rallies",items:[{id:otherRun,count:6},{id:run,count:4}],truncated:false}]
  });
  if(url.startsWith("/api/cpu/native/rallies/timeline?")&&new URLSearchParams(url.split("?")[1]).get("rally_run_id")===otherRun)return reply({
    schema:"volley-native-rally-timeline/1",project_id:"1234abcdef01",rally_run_id:otherRun,video:"second.mp4",fps:30,total_frames:900,analysis_offset_sec:4,
    rallies:[{candidate_id:"c",start_frame:30,end_frame:89}],touch_annotations:[]
  });
  if(url.startsWith("/api/cpu/native/rallies/timeline?"))return reply({schema:"volley-native-rally-timeline/1",project_id:"1234abcdef01",rally_run_id:run,video:"match.mp4",fps:30,total_frames:900,analysis_offset_sec:0,
    rallies:[{candidate_id:"a",start_frame:60,end_frame:149},{candidate_id:"b",start_frame:240,end_frame:329}],
    touch_annotations:[{frame_index:90,touch_attributed:true,kind:"attack",player_id:"athlete1",player_name:"Athlete 1"},{frame_index:270,touch_attributed:true,kind:"block",player_id:"other",player_name:"Other athlete"}]});
  if(url.startsWith("/api/cpu/native/rallies/marks?"))return reply({candidate_ids:["a"]});
  if(url==="/api/montage/native/start")return reply({job_id:job,status:"queued"});
  if(url==="/api/montage/native/audio")return reply({audio_file:audio,duration_sec:31});
  if(url==="/api/montage/native/player-highlights"||url==="/api/montage/native/game-report") {
    assert.deepEqual(body.rally_run_ids,[expectedRun]);
    if(url==="/api/montage/native/player-highlights")assert.equal(body.include_statistics,undefined);
    assert.equal(body.project_id,"1234abcdef01");
    assert.equal(body.start_frame,undefined);assert.equal(body.end_frame,undefined);
    if(body.operation==="start") {
      assert.equal(body.expected_source_fingerprint,"f".repeat(64));
      jobState="running";
      return reply({job_id:job,status:"queued"});
    }
    assert.equal(body.expected_source_fingerprint,undefined);
    if(failPreview)return reply({message:"Unverified contacts"},false);
    return reply({project_id:wrongProject?"other":"1234abcdef01",
      source_fingerprint:invalidFingerprint?"invalid":"f".repeat(64),
      selected_clips:clips,selected_rallies:clips,selected_verified_player_contacts:3,
      selected_manually_reviewed_actions:1,statistics:["Athlete 1: 3"],selected_content_sec:25});
  }
  if(url==="/api/montage/native/status")return reply({id:job,state:jobState,
    project_id:"1234abcdef01",overall_percent:45,media_url:"/api/montage/native/media?project_id=1234abcdef01&job_id="+job,
    archive_url:jobState==="complete"?"/api/montage/native/media?job_id="+job+"&archive=1":null});
  if(url==="/api/montage/native/cancel") {
    assert.equal(body.job_id,job);jobState="cancelled";return reply({cancel_requested:true});
  }
  throw Error("Unexpected request "+url);
};
const windowListeners=new Map();
const events=[];
let pollTimer=null;
const window={addEventListener:(event,fn)=>windowListeners.set(event,fn),
  dispatchEvent:event=>{events.push(event);windowListeners.get(event.type)?.(event);},
  VolleyballStudioActivity:{fetch}};
const context={document:{getElementById:get,createElement:tag=>new Element(tag)},
  window,Event:class {constructor(type){this.type=type;}},
  CustomEvent:class {constructor(type,opts={}){this.type=type;this.detail=opts.detail;}},
  URLSearchParams,setInterval:fn=>{pollTimer=fn;return 1;},clearInterval(){} ,
  URL:{createObjectURL:()=>"blob:test",revokeObjectURL(){}},
  FormData:class {append(){}}};
vm.runInNewContext(text("rust/volley-app/static/studio_overview.js"),context);
vm.runInNewContext(source,context);await tick();
assert.equal(montage("player").value,"athlete1");
assert.equal(montage("player").children.length,2);
assert.ok(!html.includes('id="studio-montage-mode"'));
assert.ok(!html.includes('id="studio-montage-presets"'));
assert.equal(montage("statistics").checked,false);
assert.equal(montage("recognition").checked,false);
assert.equal(montage("source-list").children.length,1);
assert.ok(montage("source-list").children[0].children.every(child=>child.tagName!=="select"));
const choose=async profile=>{
  montage("profile").value=profile;
  await montage("profile").fire("change");
};
const starts=()=>requests.filter(r=>r.body?.operation==="start");
await choose("attacks");
assert.equal(montage("settings-label").textContent,"Лучшие атаки");
assert.equal(montage("action").value,"attack");
assert.equal(montage("action-field").classList.contains("hidden"),true);
assert.equal(montage("player-field").classList.contains("hidden"),false);
get("studio-run").value=otherRun;expectedRun=otherRun;
windowListeners.get("volley:postprocessing-change")({
  detail:{project_id:"1234abcdef01",rally_run_id:otherRun}
});
await montage("preview").fire("click");
assert.deepEqual(requests.at(-1).body.rally_run_ids,[otherRun]);
get("studio-run").value=run;expectedRun=run;
windowListeners.get("volley:postprocessing-change")({
  detail:{project_id:"1234abcdef01",rally_run_id:run}
});
requests=[];
await choose("attacks");await montage("preview").fire("click");
let request=requests.at(-1);
assert.equal(request.url,"/api/montage/native/player-highlights");
assert.equal(request.body.player_id,"athlete1");assert.equal(request.body.action_kind,"attack");
assert.equal(request.body.clip_context_sec,0.5);
assert.equal(montage("preview-list").children[0].textContent,"Коротких фрагментов: 2");
get("studio-volume").value="60";await get("studio-volume").fire("change");
get("studio-music").files=[{name:"music.wav",size:2048}];
await montage("start").fire("click");
request=starts().at(-1);
assert.equal(request.body.layout,"vertical");assert.equal(request.body.audio_volume,60);
assert.equal(request.body.audio_file,audio);assert.equal(request.body.selection_mode,"auto");
assert.equal(request.body.include_recognition_data,false);
assert.equal(get("studio-cancel").classList.contains("hidden"),false);
const startIndex=requests.indexOf(request);
assert.equal(requests.slice(0,startIndex).filter(r=>r.body?.operation==="preview").length,2);
await get("studio-cancel").fire("click");
assert.equal(get("studio-job").textContent,"Монтаж отменён");
assert.equal(get("studio-cancel").classList.contains("hidden"),true);
assert.equal(get("studio-download").classList.contains("hidden"),true);
for(const failure of ["empty","fingerprint","project","server"]) {
  clips=failure==="empty"?0:2;invalidFingerprint=failure==="fingerprint";
  wrongProject=failure==="project";failPreview=failure==="server";
  await montage("action").fire("change");const before=starts().length;
  await montage("start").fire("click");assert.equal(starts().length,before,failure+" must prevent launch");
  assert.ok(get("studio-job").textContent,"Failed launch must explain why");
  if(failure==="empty"){await montage("start").fire("click");assert.equal(starts().length,before,"Empty selection cannot launch on retry");}
}
clips=2;invalidFingerprint=false;wrongProject=false;failPreview=false;
await choose("report");
assert.equal(montage("settings-label").textContent,"Лучшие и длинные розыгрыши");
assert.equal(montage("statistics-field").classList.contains("hidden"),false);
assert.equal(montage("player-field").classList.contains("hidden"),true);
await montage("start").fire("click");
assert.equal(starts().at(-1).url,"/api/montage/native/game-report");
assert.equal(starts().at(-1).body.player_id,undefined);
assert.equal(starts().at(-1).body.include_statistics,false);
assert.equal(starts().at(-1).body.target_duration_sec,30);
assert.ok(!montage("preview-list").children.some(row=>row.textContent.includes("финальная статистика")));
montage("statistics").checked=true;await montage("statistics").fire("change");
await montage("start").fire("click");
assert.equal(starts().at(-1).body.include_statistics,true);
jobState="cancelled";await get("studio-cancel").fire("click");
montage("recognition").checked=true;
await montage("recognition").fire("change");
await montage("start").fire("click");
assert.equal(starts().at(-1).body.include_recognition_data,true);
jobState="complete";await pollTimer();
const completed=events.filter(e=>e.type==="volley:montage-complete").at(-1);
assert.equal(completed.detail.media_url,"/api/montage/native/media?project_id=1234abcdef01&job_id="+job);
assert.equal(completed.detail.project_id,"1234abcdef01");
assert.equal(completed.detail.id,job);
assert.equal(get("studio-download").classList.contains("hidden"),false);
assert.equal(get("studio-download-archive").classList.contains("hidden"),false);
assert.ok(get("studio-download-archive").href.endsWith("&archive=1"));
await choose("favorites");
assert.equal(montage("favorites").classList.contains("hidden"),false);
assert.equal(montage("sources").classList.contains("hidden"),true);
await montage("start").fire("click");
assert.equal(get("studio-render").clicks,1);
const a="a".repeat(24),b="b".repeat(24);
assert.ok(html.includes('id="studio-montage-assembly"'));
assert.equal(montage("play").disabled,true,"empty montage cannot preview");
assert.equal(windowListeners.has("volley:montage-favorites"),true);
window.dispatchEvent(new context.CustomEvent("volley:montage-favorites",{detail:{
  project_id:"1234abcdef01",rally_run_id:run,ready:true,video:"match.mp4",
  items:[{id:a,video:"match.mp4",start:2,end:4,time:"00:02",duration:"2 с"},
    {id:b,video:"match.mp4",start:8,end:11,time:"00:08",duration:"3 с"}]
}}));
assert.equal(montage("library").children.length,2);
assert.equal(montage("sequence").children.length,1);
const add=(index)=>montage("library").children[index].children[1].children[2].children[0].fire("click");
await add(0);await add(1);
assert.deepEqual([...window.VolleyballMontageDraft.ids("1234abcdef01",run)],[a,b]);
assert.equal(montage("sequence").children.length,2);
assert.equal(montage("play").disabled,false);
await montage("sequence").children[0].children[1].children[2].children[1].fire("click");
assert.deepEqual([...window.VolleyballMontageDraft.ids("1234abcdef01",run)],[b,a],
  "Export must receive the manual order");
await montage("play").fire("click");
assert.equal(get("studio-montage-video").currentTime,8);
assert.equal(montage("play-position").textContent,"1 / 2");
get("studio-montage-video").currentTime=11;
await get("studio-montage-video").fire("timeupdate");
assert.equal(get("studio-montage-video").currentTime,2);
assert.equal(montage("play-position").textContent,"2 / 2");
get("studio-montage-video").currentTime=4;
await get("studio-montage-video").fire("timeupdate");
assert.match(montage("play-status").textContent,/завершён/);
await montage("sequence").children[1].children[1].children[2].children[2].fire("click");
assert.deepEqual([...window.VolleyballMontageDraft.ids("1234abcdef01",run)],[b],
  "Removing a clip must affect export");
assert.match(text("rust/volley-app/static/studio.js"),
  /const ids=window\.VolleyballMontageDraft\?\.ids\(s\.project,s\.run\)\|\|\[\]/);
get("studio-run").value=otherRun;
window.dispatchEvent(new context.CustomEvent("volley:montage-favorites",{detail:{
  project_id:"1234abcdef01",rally_run_id:otherRun,ready:true,video:"match.mp4",
  items:[{id:a,video:"match.mp4",start:2,end:4,time:"00:02",duration:"2 с"}]
}}));
assert.deepEqual([...window.VolleyballMontageDraft.ids("1234abcdef01",run)],[],
  "Draft order must be scoped to a specific CPU run");
window.VolleyballMontageSources=()=>({project_id:"1234abcdef01",sources:[{video:"match.mp4",run}]});
await choose("creative");
assert.equal(montage("creative-catalog").children.length,2);
const card=montage("creative-catalog").children[0];
montage("creative-video").duration=30;
await card.children[2].children[0].fire("click");
assert.equal(montage("creative-video").currentTime,2);
montage("creative-video").currentTime=5;await montage("creative-video").fire("timeupdate");
assert.equal(montage("creative-video").paused,true);
const dragEvent={preventDefault(){},stopPropagation(){},dataTransfer:{setData(){}}};
await card.fire("dragstart",dragEvent);await montage("creative-drop").fire("drop",dragEvent);
await montage("creative-catalog").children[1].children[2].children[1].fire("click");
assert.equal(montage("creative-list").children.length,2);
await montage("creative-list").children[1].fire("dragstart",dragEvent);
await montage("creative-list").children[0].fire("drop",dragEvent);
await montage("creative-play").fire("click");
assert.equal(montage("creative-video").currentTime,8);
montage("creative-video").currentTime=11;await montage("creative-video").fire("timeupdate");
assert.equal(montage("creative-video").currentTime,2);
montage("creative-video").currentTime=5;await montage("creative-video").fire("timeupdate");
assert.equal(montage("creative-video").paused,true);
assert.match(montage("creative-play-status").textContent,/завершён/);
await montage("creative-list").children[0].children[2].fire("click");
await montage("creative-list").children[0].children[2].fire("click");
assert.equal(montage("creative-list").children.length,0);
montage("creative-kind").value="favorites";await montage("creative-kind").fire("change");
assert.equal(montage("creative-catalog").children.length,1);
montage("creative-kind").value="touches";await montage("creative-kind").fire("change");
assert.equal(montage("creative-catalog").children.length,2);
montage("creative-touch-kind").value="attack";await montage("creative-touch-kind").fire("change");
assert.equal(montage("creative-catalog").children.length,1);
montage("creative-touch-player").value="other";await montage("creative-touch-player").fire("change");
assert.equal(montage("creative-catalog").children.length,0);
montage("creative-kind").value="manual";await montage("creative-kind").fire("change");
assert.equal(montage("settings-label").textContent,"Творчество");
assert.equal(montage("creative").classList.contains("hidden"),false);
assert.equal(montage("sources").classList.contains("hidden"),true);
montage("creative-source").value="match.mp4";await montage("creative-source").fire("change");
const creativeVideo=montage("creative-video");creativeVideo.duration=30;
const initialCreativeRequests=requests.filter(r=>r.url==="/api/montage/native/start").length;
await montage("start").fire("click");
assert.equal(requests.filter(r=>r.url==="/api/montage/native/start").length,initialCreativeRequests);
creativeVideo.currentTime=6;await montage("creative-in").fire("click");
creativeVideo.currentTime=3;await montage("creative-out").fire("click");
assert.equal(montage("creative-add").disabled,true,"Reversed bounds cannot be added");
creativeVideo.currentTime=2;await montage("creative-in").fire("click");
creativeVideo.currentTime=5;await montage("creative-out").fire("click");
await montage("creative-add").fire("click");
assert.equal(montage("creative-list").children.length,1);
montage("recognition").checked=false;get("studio-music").files=[];
await montage("start").fire("click");
const creativeRequest=requests.filter(r=>r.url==="/api/montage/native/start").at(-1);
assert.deepEqual(creativeRequest.body.clips,[{video:"match.mp4",start_frame:60,end_frame:149}]);
creativeSourcesExpanded=true;
window.VolleyballMontageSources=()=>({project_id:"1234abcdef01",sources:[{video:"match.mp4",run},{video:"second.mp4",run:otherRun}]});
await get("studio-project").fire("change");
montage("creative-kind").value="rallies";await montage("creative-kind").fire("change");
assert.equal(montage("creative-catalog").children.length,3,"Every source in the selected analysis contributes rallies");
await montage("creative-catalog").children[2].children[2].children[1].fire("click");
await montage("creative-play").fire("click");
assert.equal(creativeVideo.currentTime,2);
creativeVideo.currentTime=5;await creativeVideo.fire("timeupdate");
assert.match(creativeVideo.src,/second.mp4/);
assert.equal(creativeVideo.currentTime,5,"Source analysis offset must be included");
await montage("start").fire("click");
assert.deepEqual(requests.filter(r=>r.url==="/api/montage/native/start").at(-1).body.clips,[
  {video:"match.mp4",start_frame:60,end_frame:149},
  {video:"second.mp4",start_frame:30,end_frame:89,analysis_offset_sec:4}
],"Cross-video playback and export must preserve clip order and offsets");
await choose("report");assert.equal(montage("creative").classList.contains("hidden"),true);
console.log("Live montage builder: profiles, fingerprint, audio, sequenced preview, reordered export IDs, cleanup and source isolation passed");

const libListeners=new Map();
const dialog=get("studio-video-library-player");
dialog.showModal=function(){this.open=true;};
dialog.close=function(){this.open=false;};
const libContext={document:{getElementById:get,addEventListener(){}},
  window:{addEventListener:(name,fn)=>libListeners.set(name,fn)},URLSearchParams};
vm.runInNewContext(text("rust/volley-app/static/studio_video_library.js"),libContext);
libListeners.get("volley:montage-complete")({detail:completed.detail});
assert.equal(dialog.open,true,"Completed montage opens the video player");
assert.equal(get("studio-video-player-media").src,completed.detail.media_url);
dialog.close();
libListeners.get("volley:montage-complete")({detail:{...completed.detail,project_id:"abcdef123456"}});
assert.equal(dialog.open,false,"Other project must not open a player");
libListeners.get("volley:montage-complete")({detail:{...completed.detail,media_url:"/api/montage/native/media?project_id=1234abcdef01&job_id=montage_"+"2".repeat(32)}});
assert.equal(dialog.open,false,"Mismatched media URL must not open");