import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
assert.ok(!html.includes('id="studio-preview-video"'));
const from=js.indexOf("  const projectRallyRuns=");
const to=js.indexOf("  const projectRallyCard=",from);
assert.ok(from>=0&&to>from);

const itemA={candidate_id:"a".repeat(24),start_frame:60,end_frame:90,motion_event_count:3};
const itemB={candidate_id:"b".repeat(24),start_frame:30,end_frame:60,motion_event_count:4};
const run=(video,id,created_at)=>({video,rally_run_id:id,created_at});
const chain={byAnalysis:new Map([
  ["a1",[run("a.mp4","run-a",10)]],
  ["b1",[run("b.mp4","run-b-old",15)]],
  ["b2",[run("b.mp4","run-b",20)]]
])};
const s={project:"project",projectVideos:["a.mp4","b.mp4","unprocessed.mp4"],chain,
  run:"run-a",projectRallies:new Map()};
const ui={analysis:{value:"a1"},list:{scrollTop:24}};
const opened=[],requests=[],messages=[];
const context={s,ui,cards(){},valid:id=>/^[a-f0-9]{24}$/.test(id),message:value=>messages.push(value),
  rallyClipFrames:row=>({start:row.start_frame,end:row.end_frame}),
  url:(path,query)=>({path,...query}),
  previewCache:{load:async query=>{
    requests.push(query);
    const first=query.rally_run_id==="run-a",item=first?itemA:itemB;
    if(query.path.endsWith("/marks"))return {candidate_ids:first?[item.candidate_id]:[]};
    return {schema:"volley-native-rally-timeline/1",project_id:query.project_id,
      rally_run_id:query.rally_run_id,video:first?"a.mp4":"b.mp4",
      fps:30,analysis_offset_sec:first?17:80,rallies:[item]};
  }},
  view:item=>opened.push(item),refreshPicker(){},alignActiveRally(){},
  setAnalysis:async (id,exact,_video,rally)=>{
    s.run=id;s.current=rally.item.candidate_id;
    opened.push({analysis:ui.analysis.value,exact,...rally});
  }
};
const api=runInNewContext(js.slice(from,to)+
  "\n({projectRallyRuns,refreshProjectRallies,projectRallyEntries,openProjectRally})",context);
await api.refreshProjectRallies();
assert.deepEqual([...s.projectRallies.keys()],["run-a","run-b"]);
assert.equal(requests.length,4);
assert.equal(messages.length,0);
await api.refreshProjectRallies();
assert.equal(requests.length,4,"Repainting does not reload unchanged sources");
const entries=(filter="all")=>Array.from(api.projectRallyEntries(
  s.projectRallies.values(),s.run,[{...itemA,detail:true}],null,filter));
assert.deepEqual(entries().map(row=>row.item.candidate_id),[itemA.candidate_id,itemB.candidate_id]);
assert.deepEqual(entries("marked").map(row=>row.item.candidate_id),[itemA.candidate_id]);
assert.deepEqual(entries("unmarked").map(row=>row.item.candidate_id),[itemB.candidate_id]);
assert.deepEqual(entries("touches_desc").map(row=>row.item.candidate_id),[itemB.candidate_id,itemA.candidate_id]);
await api.openProjectRally(entries()[1].source,itemB);
assert.equal(opened[0].run,"run-b");
assert.equal(opened[0].video,"b.mp4");
assert.equal(opened[0].analysis,"b2");
assert.equal(opened[0].offset+itemB.start_frame/opened[0].fps,81);
assert.equal(ui.list.scrollTop,24);
assert.deepEqual(Array.from(api.projectRallyRuns(chain,s.projectVideos,"run-b-old"),row=>row.run),
  ["run-a","run-b-old"],"An explicitly selected CPU version replaces only its own source");
console.log("PASS: project rally list, per-source versions, filters and cross-video playback timing.");
