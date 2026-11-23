import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const slice=(start,end)=> {
  const lo=source.indexOf(start),hi=source.indexOf(end,lo);
  assert.ok(lo>=0&&hi>lo,`Missing source section: ${start}`);
  return source.slice(lo,hi);
};

{
  const calls=[];
  const state={project:"project",video:"match.mp4",replay:"replay",
    rallyRun:"selected-run",tracksLoaded:true,loadingKey:null};
  const context={state,ui:{project:{value:"project"}},
    get:id=>id==="studio-run"?{value:"selected-run"}:null,
    loadLibraryRoster:()=>calls.push("library"),
    loadPage:async arg=>calls.push(["reload",arg]),JSON};
  runInNewContext(slice("  const loadVideos=async()=>{","  const publishTrackCovers=")+
    "\nthis.loadVideos=loadVideos;",context);
  await context.loadVideos();
  assert.equal(calls.length,2);
  assert.equal(calls[0],"library");
  assert.equal(calls[1][0],"reload");
  assert.equal(calls[1][1].preserveCards,true);
}

{
  const events=[];
  const state={
    project:"project",video:"match.mp4",replay:"replay",rallyRun:"run",
    token:0,selected:new Set(),viewPlayer:"",viewConflicts:false,
    offset:0,limit:24,revision:"old-revision",sourceSha:"a".repeat(64),
    groupRevision:null,groups:[],grouped:new Set(),playerMatches:{},tracksLoaded:true,
    assignmentPending:new Set()
  };
  const response={schema:"volley-native-player-track-links/1",
    source_replay_sha256:"a".repeat(64),automatic_reid_binding:true,
    tracks:[],rally_track_ids:[2],total_tracks:1,filtered_total:0,
    unassigned_track_ids:[],excluded_track_ids:[],comparison_track_ids:[],roster:[],
    revision:"new-revision",writable:true,assigned_count:1,excluded_count:0,
    automatic_assignment_threshold_percent:92};
  const calls=[];
  const context={
    state,ui:{list:{replaceChildren(){}},page:{textContent:""},
      assigned:{textContent:""},excluded:{textContent:""}},
    galleryAutoThreshold:{value:"80"},galleryAutoValue:{textContent:"80%"},
    node:()=>({}),path:()=>"/api/players/native/tracks",
    read:async()=>response,
    unlock:()=>{},note:()=>{},updateUnassignedLabel:()=>{},
    publishTrackCovers:()=>{},showPlayerChoices:()=>{},
    renderRosterBar:()=>{},renderRows:()=>{},renderGroupCards:()=>{},
    animateTrackCards:f=>f(),reviewReady:()=>false,reviewLoad:()=>{},
    window:{dispatchEvent:e=>events.push(e)},
    CustomEvent:class {constructor(type,options){this.type=type;this.detail=options.detail;}},
    console
  };
  runInNewContext(slice("  const loadPage=async(","  const select=")+
    "\nthis.loadPage=loadPage;",context);
  await context.loadPage({preserveCards:true});
  assert.equal(state.revision,"new-revision");
  assert.equal(context.galleryAutoThreshold.value,"92");
  assert.equal(context.galleryAutoValue.textContent,"92%");
  assert.equal(events.length,1);
  assert.equal(events[0].type,"volley:track-assignments-updated");
  assert.equal(events[0].detail.replay_id,"replay");
  await context.loadPage({preserveCards:true});
  assert.equal(events.length,1,"unchanged revision must not invalidate preview labels");
}
{
  const events=[],added=[],chips=[];
  const state={project:"match",libraryPlayers:[
    {id:"player-a",name:"Игрок А"},{id:"player-b",name:"Игрок Б"}
  ],projectPlayerIds:new Set(["player-a"]),playerTrackCounts:{"player-a":17},
    rosterBusy:false,rosterLoaded:true};
  const rosterBar={hidden:false},rosterTitle={textContent:""},
    rosterAddAll={hidden:false,disabled:false,textContent:""},
    rosterChips={append:chip=>chips.push(chip)},
    rosterHint={textContent:""};
  const context={state,rosterBar,rosterTitle,rosterAddAll,rosterChips,rosterHint,
    clear:()=>{chips.length=0;},
    node:(_tag,_klass,value)=>({textContent:value,addEventListener(_name,fn){this.click=fn;}}),
    attachPlayers:ids=>added.push(ids),
    window:{dispatchEvent:e=>events.push(e)},
    CustomEvent:class {constructor(type,options){this.type=type;this.detail=options.detail;}}};
  runInNewContext(slice("  const renderRosterBar=()=>{","  const showPlayerChoices=")+
    "\nthis.renderRosterBar=renderRosterBar;",context);
  context.renderRosterBar();
  const linked=chips.find(chip=>chip.textContent.includes("Игрок А"));
  const unlinked=chips.find(chip=>chip.textContent.includes("Игрок Б"));
  assert.match(linked.textContent,/17 ID/);
  assert.equal(linked.disabled,false);
  linked.click();
  assert.equal(events.length,1);
  assert.equal(events[0].type,"volley:player-gallery-select");
  assert.equal(events[0].detail.player_id,"player-a");
  unlinked.click();
  assert.equal(added.length,1);
  assert.equal(added[0][0],"player-b");
}
console.log("Players reloads live assignments; preview labels invalidate on revision change: OK");
