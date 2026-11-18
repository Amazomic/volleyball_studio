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
    this.textContent="";this.className="";
    const classes=new Set();
    this.classList={add:name=>classes.add(name),remove:name=>classes.delete(name),
      contains:name=>classes.has(name),toggle:(name,on)=>on?classes.add(name):classes.delete(name)};
  }
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=[...items];this.textContent="";}
  addEventListener(name,fn){this.listeners.set(name,fn);}
  async fire(name,event={target:this}){await this.listeners.get(name)?.(event);await tick();}
  querySelectorAll(){return [];}
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
let expectedRun=run;
const reply=(data,ok=true)=>({ok,status:ok?200:409,json:async()=>data});
const fetch=async(url,options={})=>{
  const body=typeof options.body==="string"?JSON.parse(options.body):null;
  requests.push({url,body});
  if(url==="/api/projects")return reply({projects:[{
    id:"1234abcdef01",videos:["match.mp4"],player_ids:["athlete1"]
  }]});
  if(url==="/api/players/library/native")return reply({players:[
    {id:"athlete1",name:"Athlete 1"},{id:"other",name:"Other athlete"}
  ]});
  if(url.startsWith("/api/studio/analysis-overview?"))return reply({
    schema:"volley-native-analysis-overview/1",project_id:"1234abcdef01",video:"match.mp4",metadata_only:true,
    stages:[{key:"rallies",items:[{id:otherRun,count:6},{id:run,count:4}],truncated:false}]
  });
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
    overall_percent:45,media_url:"/api/montage/native/media?job_id="+job,
    archive_url:jobState==="complete"?"/api/montage/native/media?job_id="+job+"&archive=1":null});
  if(url==="/api/montage/native/cancel") {
    assert.equal(body.job_id,job);jobState="cancelled";return reply({cancel_requested:true});
  }
  throw Error("Unexpected request "+url);
};
const windowListeners=new Map();
let pollTimer=null;
const context={document:{getElementById:get,createElement:tag=>new Element(tag)},
  window:{addEventListener:(event,fn)=>windowListeners.set(event,fn),VolleyballStudioActivity:{fetch}},
  URLSearchParams,setInterval:fn=>{pollTimer=fn;return 1;},clearInterval(){} ,
  URL:{createObjectURL:()=>"blob:test",revokeObjectURL(){}},
  FormData:class {append(){}}};
vm.runInNewContext(text("rust/volley-app/static/studio_overview.js"),context);
vm.runInNewContext(source,context);await tick();
assert.equal(montage("player").value,"athlete1");
assert.equal(montage("player").children.length,2);
assert.ok(!html.includes('data-profile="report"'));
assert.equal(montage("statistics").checked,false);
assert.equal(montage("recognition").checked,false);
assert.equal(montage("source-list").children.length,1);
assert.ok(montage("source-list").children[0].children.every(child=>child.tagName!=="select"));
const choose=async profile=>montage("presets").fire("click",{
  target:{closest:()=>({dataset:{profile}})}
});
const starts=()=>requests.filter(r=>r.body?.operation==="start");
await choose("attacks");
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
}
clips=2;invalidFingerprint=false;wrongProject=false;failPreview=false;
montage("mode").value="report";await montage("mode").fire("change");
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
assert.equal(get("studio-download").classList.contains("hidden"),false);
assert.equal(get("studio-download-archive").classList.contains("hidden"),false);
assert.ok(get("studio-download-archive").href.endsWith("&archive=1"));
await choose("favorites");await montage("start").fire("click");
assert.equal(get("studio-render").clicks,1);
console.log("Live montage builder: roster, attack/report/favorite profiles, source fingerprint, audio, start, cancel and invalid-preview veto passed");
