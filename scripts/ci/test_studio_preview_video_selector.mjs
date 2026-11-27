import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import {previewWasmWindow} from "./helpers/preview_wasm.mjs";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
assert.ok(!html.includes('id="studio-preview-video"'));
const from=js.indexOf("  const projectRallyRuns=");
const to=js.indexOf("  const refreshProjectCardNames=",from);
assert.ok(from>=0&&to>from);

const itemA={candidate_id:"a".repeat(24),start_frame:60,end_frame:90,motion_event_count:3};
const itemB={candidate_id:"b".repeat(24),start_frame:30,end_frame:60,motion_event_count:4};
const run=(video,id,created_at)=>({video,rally_run_id:id,created_at});
const chain={byAnalysis:new Map([
  ["launch-a",[run("a.mp4","run-a",10),run("b.mp4","run-b-old",15),run("b.mp4","run-b",20)]],
  ["launch-b",[run("a.mp4","run-a-new",30),run("b.mp4","run-b-new",40)]]
])};
const s={project:"project",projectVideos:["a.mp4","b.mp4","unprocessed.mp4"],chain,analysis:"launch-a",
  run:"run-a",projectRallies:new Map(),playbackMode:"sequence",current:itemA.candidate_id,generation:1};
const ui={analysis:{value:"launch-a"},list:{scrollTop:24},video:{currentTime:20,paused:false,pause(){this.paused=true;}}};
const opened=[],requests=[],messages=[],followed=[];
const context={window:previewWasmWindow(),s,ui,cards(){},valid:id=>/^[a-f0-9]{24}$/.test(id),message:value=>messages.push(value),
  rallyClipFrames:row=>({start:row.start_frame,end:row.end_frame}),
  url:(path,query)=>({path,...query}),
  previewCache:{load:async query=>{
    requests.push(query);
    const first=query.rally_run_id==="run-a",item=first?itemA:itemB;
    if(query.path.endsWith("/marks"))return {candidate_ids:first?[item.candidate_id]:[],writable:true,revision:"rev"};
    if(query.path.endsWith("/preview"))return {schema:"volley-native-rally-preview/1",project_id:query.project_id,
      rally_run_id:query.rally_run_id,video:first?"a.mp4":"b.mp4",
      items:[{...item,detected_start_frame:item.start_frame,detected_end_frame:item.end_frame,
        metadata:{team_1:[first?4:42],team_2:[]},estimated_action_counts:{dig:1}}]};
    return {schema:"volley-native-rally-timeline/1",project_id:query.project_id,
      rally_run_id:query.rally_run_id,video:first?"a.mp4":"b.mp4",
      fps:30,analysis_offset_sec:first?17:80,rallies:[item]};
  }},
  view:item=>opened.push(item),refreshPicker(){},followSequenceRally:()=>followed.push(s.current),
  setAnalysis:async (id,exact,_video,rally)=>{
    s.run=id;s.current=rally.item.candidate_id;s.video=rally.video;s.analysisOffset=rally.offset;s.fps=rally.fps;
    opened.push({analysis:ui.analysis.value,exact,...rally});
  }
};
const api=runInNewContext(js.slice(from,to)+
  "\n({projectRallyRuns,refreshProjectRallies,projectRallyEntries,openProjectRally,advanceProjectRally})",context);
await api.refreshProjectRallies();
assert.deepEqual([...s.projectRallies.keys()],["run-a","run-b"]);
assert.equal(requests.length,6);
assert.equal(messages.length,0);
await api.refreshProjectRallies();
assert.equal(requests.length,6,"Repainting does not reload unchanged sources");
const entries=(filter="all")=>Array.from(api.projectRallyEntries(
  s.projectRallies.values(),s.run,[{...itemA,detail:true}],null,filter));
assert.deepEqual(entries().map(row=>row.item.candidate_id),[itemA.candidate_id,itemB.candidate_id]);
assert.equal(entries()[1].item.metadata.team_1[0],42,"Inactive video uses verified detailed card data");
assert.equal(entries()[1].item.estimated_action_counts.dig,1);
assert.ok(entries().every(row=>row.detail));
assert.deepEqual(entries("marked").map(row=>row.item.candidate_id),[itemA.candidate_id]);
assert.deepEqual(entries("unmarked").map(row=>row.item.candidate_id),[itemB.candidate_id]);
assert.deepEqual(entries("touches_desc").map(row=>row.item.candidate_id),[itemB.candidate_id,itemA.candidate_id]);
await api.openProjectRally(entries()[1].source,itemB,false,81.5);
assert.equal(opened[0].run,"run-b");
assert.equal(opened[0].video,"b.mp4");
assert.equal(opened[0].requestedTime,81.5,"Cross-video card seek is passed into playback");
assert.equal(opened[0].analysis,"launch-a","Switching video must not change the selected project analysis");
assert.equal(opened[0].offset+itemB.start_frame/opened[0].fps,81);
assert.equal(ui.list.scrollTop,24);
assert.deepEqual(Array.from(api.projectRallyRuns(chain,s.projectVideos,"run-b-old","launch-a"),row=>row.run),
  ["run-a","run-b-old"],"An explicitly selected CPU version replaces only its own source");
assert.deepEqual(Array.from(api.projectRallyRuns(chain,s.projectVideos,"run-a-new","launch-b"),row=>row.run),
  ["run-a-new","run-b-new"],"A different analysis exposes only its own rallies");
assert.deepEqual(Array.from(api.projectRallyRuns(chain,s.projectVideos,"run-a-new","launch-a"),row=>row.run),
  ["run-a","run-b"],"A stale active source cannot mix analyses");
s.run="run-a";s.video="a.mp4";s.current=itemA.candidate_id;s.analysisOffset=17;s.stopAt=20;
const boundaryFrom=js.indexOf("  const nextRally=()=>{");
const boundaryTo=js.indexOf('  ui.video.addEventListener("timeupdate"',boundaryFrom);
const boundary=runInNewContext(js.slice(boundaryFrom,boundaryTo)+"\n({onRallyBoundary})",{
  window:previewWasmWindow(),s,ui,timeline:{data:{project_id:"project",rally_run_id:"run-a",rallies:[itemA]}},
  fps:()=>30,training:{active:false},advanceProjectRally:api.advanceProjectRally,
  failed:error=>{throw error;},view(){throw Error("Unexpected same-video rally");},
  followSequenceRally(){}
});
s.items=[itemA];
boundary.onRallyBoundary();
await new Promise(resolve=>setImmediate(resolve));
assert.equal(s.run,"run-b");
assert.equal(s.video,"b.mp4");
assert.equal(s.analysis,"launch-a");
assert.equal(s.analysisOffset,80,"Rally timing must use the new source's offset");
assert.equal(followed.at(-1),itemB.candidate_id,"Sequence follows the active cross-video rally");
console.log("PASS: project rally list, per-source versions, filters and cross-video playback timing.");

{
  const group="launch-a",sources=new Map([
    ["run-a",{analysis:group,run:"run-a",video:"a.mp4",rows:[itemA],fps:30,offset:17}],
    ["run-b",{analysis:group,run:"run-b",video:"b.mp4",rows:[itemB],fps:30,offset:80}]
  ]);
  const state={project:"project",projectVideos:["a.mp4","b.mp4"],analysis:group,run:"run-a",
    video:"a.mp4",videoKey:"project:a.mp4",current:itemA.candidate_id,playbackMode:"sequence",
    generation:1,analysisOffset:17,fps:30,selected:new Map(),projectRallies:sources,
    chain:{choices:[{value:group,kind:"launch",video:"a.mp4",members:[{video:"a.mp4"},{video:"b.mp4"}]}],
      byAnalysis:new Map([[group,[run("a.mp4","run-a",10),run("b.mp4","run-b",20)]]])}};
  const rail={scrollTop:24},media={currentTime:20,readyState:2};
  const controls={analysis:{value:group},run:{value:"run-a"},video:media,list:rail,
    job:{},cancel:{classList:{add(){}}},download:{classList:{add(){}}},
    viewTitle:{},viewDescription:{},videoTime:{},emptyVideo:{classList:{add(){}}}};
  const data={epoch:1,data:{video:"a.mp4",rally_run_id:"run-a",video_duration_sec:120},status:{},middle:{},end:{}};
  const requests=[],videoLoads=[],drawn=[],followed=[];
  let cleared=0;
  const block=(first,last)=>js.slice(js.indexOf(first),js.indexOf(last,js.indexOf(first)));
  const runtime={s:state,ui:controls,timeline:data,overlay:{raw:false},training:{active:false},
    window:{...previewWasmWindow(),dispatchEvent(){}},
    CustomEvent:class{},sourceSelection:{save(){}},refreshPicker(){},
    options:(select,rows,key)=>{select.options=rows.map(row=>({value:row[key]}));},
    setOverlaySource(){},updateRawDetectorLabel(){},prefetchOverlayAtFrame(){},rawFusion:()=>null,
    fps:()=>state.fps,clock:frame=>(frame/state.fps).toFixed(3),
    cards(){},rallyClipFrames:row=>({start:row.start_frame,end:row.end_frame}),
    renderRallyTouches(){},rallyCardProgresses:new Map(),previewIsVisible:()=>true,
    loadPreviewVideo:autoplay=>{videoLoads.push({video:state.video,autoplay,seek:state.seekAt});media.currentTime=state.seekAt;},
    drawTimeline:()=>drawn.push({video:state.video,source:data.data?.video||null,clipStart:state.clipStart}),
    ensurePreviewVideo:()=>assert.equal(state.videoKey,"project:"+state.video),
    refreshProjectRallies:async()=>{},loadPage:async()=>{},
    clearVideo:()=>{state.videoKey="";state.current="";state.clipStart=state.clipEnd=null;},
    clearTimeline:()=>{data.epoch++;data.data=null;cleared++;},
    stopPoll(){},wipe(){},renderSelection(){},message(){},failed:error=>{throw error;},
    url:(path,query)=>({path,...query}),previewCache:{load:async query=>{
      requests.push(query);
      return {schema:"volley-native-rally-timeline/1",project_id:"project",rally_run_id:"run-b",video:"b.mp4",
        total_frames:5400,video_duration_sec:180,fps:30,analysis_offset_sec:80,rallies:[itemB],
        touch_frames:[],touch_annotations:[],authoritative_game_events:false};
    }},
    validSurfaceContacts:()=>true,cpuTouchStyle:{},floorContacts:()=>[],
    timelineDuration:()=>data.data?.video_duration_sec||0,timeLabel:String,syncFavoritesSelection(){},
    drawOverlay(){},refreshRallyCardTouchTicks(){},updateRallyCardProgress(){},
    followSequenceRally:()=>followed.push(state.current)
  };
  const entry=runInNewContext(
    block("  const reset=()=>{","  const options=")+
    block("  const setRuns=async(","  const selectReferenceEntry=")+
    block("  const view=(item,requestedTime=null)=>{","  const renderReferenceCards=")+
    block("  const loadTimeline=async()=>{","  const rallyAtTimelineTime=")+
    block("  const openProjectRally=async(","  const refreshProjectCardNames=")+
    "\n({advanceProjectRally})",runtime);
  await entry.advanceProjectRally();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(state.analysis,group);
  assert.equal(controls.analysis.value,group);
  assert.equal(state.run,"run-b");
  assert.equal(state.video,"b.mp4");
  assert.equal(cleared,1,"The old full-video timeline is cleared");
  assert.deepEqual(videoLoads,[{video:"b.mp4",autoplay:true,seek:81}]);
  assert.equal(state.clipStart,81,"The rally timeline uses the second video's offset");
  assert.equal(state.clipEnd,80+61/30);
  assert.equal(data.data.video,"b.mp4");
  assert.equal(data.data.rally_run_id,"run-b");
  assert.equal(data.end.textContent,"180","The full timeline uses the second video's actual duration");
  assert.equal(requests[0].rally_run_id,"run-b");
  assert.equal(drawn.at(-1).source,"b.mp4");
  assert.equal(followed.at(-1),itemB.candidate_id);
}
{
  const state={project:"project",analysis:"previous",run:"old",chain:{},queueRallyUpdates:new Map()};
  const calls=[];
  const from=js.indexOf("  const refreshQueueRallies=status=>{");
  const to=js.indexOf('  window.addEventListener("volley:pipeline-status"',from);
  const scope={s:state,invalidateCatalog(){},loadChain:async(...args)=>calls.push(args),failed:error=>{throw error;}};
  const update=runInNewContext(js.slice(from,to)+"\nrefreshQueueRallies",scope);
  update({project_id:"project",queue_id:"new-queue",project_queue:{active:true,videos:[{id:"new-queue"}]}});
  update({project_id:"project",queue_id:"new-queue",pipeline_id:"child",project_queue:{active:false,videos:[
    {id:"new-queue",postprocessing_id:"run-a"},{id:"child",postprocessing_id:"run-b"}]}});
  assert.equal(calls[0][0],JSON.stringify(["","launch","new-queue"]),"A fast full launch selects its queue ID, not its last video job");
  assert.equal(calls[0][1],"run-a");
}
console.log("PASS: cross-video sequence changes full-video timeline and rally interval while preserving the launch.");
