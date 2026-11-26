import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const section=(from,to)=>{
  const start=source.indexOf(from),end=source.indexOf(to,start);
  assert.ok(start>=0&&end>start,from);
  return source.slice(start,end);
};
const project="a".repeat(12),first="b".repeat(32),second="c".repeat(32);
const queue={active:true,completed_videos:0,total_videos:2,progress_percent:10,frames_processed:300,rallies_total:50,
  videos:[{id:first,video:"first.mp4",status:"running",progress_percent:50,
    rallies_total:33,rallies_completed:12,stage:"RF-DETR",analysis_range:{start_sec:38,end_sec:1000}},
  {id:second,video:"second.mp4",status:"queued",progress_percent:0}]};
const run={schema:"volley-studio-pipeline/1",id:first,project_id:project,
  queue_id:first,video:"first.mp4",status:"running",project_queue:queue};
let current=run,fail=false,resetCount=0,historyCount=0,stops=0;
const queries=[],displays=[],errors=[];
const state={pipelineId:"",pipelineProject:"",pipelineQueue:"",analysisGeneration:1,
  polling:false,pollErrors:0,pipelineTimer:null};
const ctx={
  state,ui:{logPanel:{}},projectId:()=>project,
  get:async(_path,query)=>{queries.push({...query});if(fail)throw Error("offline");return current;},
  recordHistoryRun(){},resetLog(){resetCount++;},loadHistory(){historyCount++;},
  displayPipeline:s=>displays.push(s),saveMessage:message=>errors.push(message),
  stopPipelinePolling(){stops++;},setInterval:()=>1,updateProfile(){}
};
const api=runInNewContext(section("  const pollPipeline=async()=>{","  const loadVideos=async()=>{")+
  "\n({pollPipeline,restorePipeline})",ctx);
await api.restorePipeline();
assert.equal(state.pipelineQueue,first);
assert.equal(state.pipelineId,first);
assert.equal(queries.at(-1).queue_id,first);
assert.equal(state.pipelineTimer,1);
current={...run,id:second,video:"second.mp4",
  project_queue:{...queue,completed_videos:1}};
await api.pollPipeline();
assert.equal(state.pipelineId,second);
assert.equal(state.pipelineQueue,first);
assert.equal(resetCount,1);
assert.equal(historyCount,1);
assert.equal(displays.at(-1).video,"second.mp4");
fail=true;
for(let index=0;index<4;index++)await api.pollPipeline();
assert.equal(stops,1,"Transient failures must not stop the poll timer");
assert.equal(state.pipelineId,second);
assert.equal(state.pollErrors,4);
assert.ok(errors.at(-1).includes("повторяем подключение"));
fail=false;
await api.pollPipeline();
assert.equal(state.pollErrors,0);

class Node{
  constructor(tag,css,text){this.tag=tag;this.css=css;this.textContent=text||"";this.children=[];
    this.classList={toggle:(_klass,hidden)=>this.hidden=hidden};}
  append(...children){this.children.push(...children);}
  setAttribute(name,value){this[name]=value;}
}
const target=new Node("div");
const renderer=runInNewContext(section("  const pipelineStages=","  const pipelineEdges=")+
  section("  const graphLabel=","  const graphText=")+
  section("  const pipelineStageActivity=","  const updatePipelineGraph=")+
  section("  const renderProjectQueue=queue=>{","  const showGraphResult=")+
  "\nrenderProjectQueue",{
    $:()=>target,clear:node=>node.children=[],make:(...args)=>new Node(...args),
    ui:{video:{options:[{value:"first.mp4",textContent:"Match A"},{value:"second.mp4",textContent:"Match B"}]}},
    window:{VolleyballUi:{translate:text=>text}},historyDuration:ms=>String(ms/1000)
  });
renderer(queue);
assert.equal(target.hidden,false);
assert.equal(target.children[0].children[0].textContent,"Проект · 0/2 видео");
assert.equal(target.children[1].value,10);
assert.match(target.children[0].children[1].textContent,/300 кадров.*50 розыгр/);
assert.equal(target.children[2].children[0].children[0].textContent,"Match A · 38 – 1000");
assert.match(target.children[2].children[0].children[1].textContent,/12\/33 розыгр/);
assert.match(target.children[2].children[0].children[1].textContent,/RF-DETR/);
assert.match(target.children[3].children[0].children[1].textContent,/В очереди/);
assert.equal(target.children[3].children[1].value,0);
renderer(null);
assert.equal(target.hidden,true);
assert.equal(target.children.length,0);

const courtChecks=[];
const requireCourt=runInNewContext(section("  const requireAnalysisCourts=async","  const launchPipeline=async")+"\nrequireAnalysisCourts",{
  get:async(path,query)=>{courtChecks.push({path,...query});return {court:query.video==="first.mp4"?[[0,0],[1,0],[1,1],[0,1]]:[]};}
});
const videoOptions=[{value:"first.mp4",textContent:"Match A"},{value:"second.mp4",textContent:"Match B"}];
await requireCourt(project,videoOptions.slice(0,1));
await assert.rejects(requireCourt(project,videoOptions),/Match B/);
assert.equal(courtChecks.at(-1).path,"/api/video-edit");
assert.equal(courtChecks.at(-1).video,"second.mp4");
assert.match(source,/await requireAnalysisCourts\(project,/);
const posts=[];
const start=runInNewContext(section("  const startWithRecovery=async","  const announce=")+
  "\nstartWithRecovery",{post:async(path,payload)=>{posts.push({path,payload});return {id:first};}});
await start({project_id:project,video:"first.mp4"},true);
await start({project_id:project,video:"first.mp4"});
assert.equal(posts[0].path,"/api/studio/pipeline/start-project");
assert.equal(posts[1].path,"/api/studio/pipeline/start");
console.log("PASS: project queue restoration, transition, reconnect, per-video progress and launch endpoint.");