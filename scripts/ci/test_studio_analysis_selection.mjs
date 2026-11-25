import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const src=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const chainSource=readFileSync(new URL("../../rust/volley-app/static/studio_chain.js",import.meta.url),"utf8");
const chainRuntime={window:{}};
runInNewContext(chainSource,chainRuntime);
const latestRun=chainRuntime.window.VolleyStudioChain.latestRun;
const slice=(start,end)=>{
  const a=src.indexOf(start),b=src.indexOf(end,a);
  assert.ok(a>=0&&b>a,"Cannot locate "+start);
  return src.slice(a,b);
};
const primaryA="[\"match.mp4\",\"fusion\",\"a\"]",primaryB="[\"match.mp4\",\"fusion\",\"b\"]";
const runA="a".repeat(32),runB="b".repeat(32);
const choice=value=>({value,id:value===primaryA?"a":"b",kind:"fusion",video:"match.mp4",
  label:value===primaryA?"Current GPU":"Other GPU",display_name:value===primaryA?"Current GPU":"Other GPU"});
const run=id=>({rally_run_id:id,display_name:id===runA?"Current CPU":"Other CPU",video:"match.mp4",created_at:0});
const chain=(values,runs)=>({
  choices:values.map(choice),
  byAnalysis:new Map(values.map(value=>[value,runs[value]||[]]))
});

// Test the actual picker click callbacks, not just static markup.
function element(tag,cls="",text=""){
  const classes=new Set(cls.split(" ").filter(Boolean));
  return {
    tag,cls,textContent:text,value:"",disabled:false,children:[],handlers:{},
    classList:{
      add:x=>classes.add(x),contains:x=>classes.has(x),
      toggle:(x,force)=>{if(force)classes.add(x);else classes.delete(x);}
    },
    setAttribute(k,v){this[k]=v;},getAttribute(k){return this[k]??"";},
    append(...nodes){this.children.push(...nodes);},
    addEventListener(k,fn){(this.handlers[k]??=[]).push(fn);},
    click:async function(){for(const fn of this.handlers.click||[])await fn();}
  };
}
const state={project:"abc",analysis:primaryA,run:runA,
  chain:chain([primaryA,primaryB],{[primaryA]:[run(runA),run(runB)],[primaryB]:[run(runB)]})};
const parent={insertBefore(){}};
const select=()=>Object.assign(element("select"),{parentNode:parent,
  options:[{value:"",textContent:"Choose"},{value:primaryA,textContent:"Current GPU"},
    {value:primaryB,textContent:"Other GPU"}].map(row=>({...row,remove(){this.removed=true;}})),
  selectedOptions:[{textContent:"Current GPU"}]});
const ui={analysis:select(),run:select()};
ui.analysis.value=primaryA;
ui.run.options=[{value:"",textContent:"Choose"},{value:runA,textContent:"Current CPU"},
  {value:runB,textContent:"Other CPU"}].map(row=>({...row,remove(){this.removed=true;}}));
ui.run.value=runA;
const calls=[],posts=[];
const pickerCtx={s:state,ui,make:element,wipe:x=>{x.children=[];},
  document:{addEventListener(){}},window:{confirm:()=>{throw Error("No delete confirmation allowed");},prompt:()=> "New name"},
  post:async(path,body)=>posts.push({path,body}),
  invalidateCatalog(){},message(){},failed:e=>{throw e;},
  loadChain:async(...args)=>calls.push(args),pickers:new Map()};
runInNewContext(slice("  const makeAnalysisPicker=","  // Shared by the analysis tab")+
  "\nthis.makeAnalysisPicker=makeAnalysisPicker;",pickerCtx);
pickerCtx.makeAnalysisPicker(ui.analysis,true);
// Capture the picker hosts that are inserted next to their native selects.
let hosts=[];
parent.insertBefore=(node)=>hosts.push(node);
pickerCtx.makeAnalysisPicker(ui.run,false);
const cpuHost=hosts[0];
assert.equal(cpuHost.children[1].children[0].children.length,3,
  "Every row needs choice, delete and rename buttons");
const actionButton=(row,action)=>row.children.find(item=>item.cls?.includes("analysis-picker-"+action));
assert.ok(actionButton(cpuHost.children[1].children[0],"delete").title.startsWith("Удалить:"));
assert.doesNotMatch(slice("  const makeAnalysisPicker=","  // Shared by the analysis tab"),
  /window\.confirm\(/,"Picker deletion must never show a confirmation");
await actionButton(cpuHost.children[1].children[1],"delete").click(); // delete a nonselected CPU run
assert.equal(calls.length,1);
assert.deepEqual(Array.from(calls.pop()),[primaryA,runA,true],
  "Deleting another CPU version must retain the active source");
await actionButton(cpuHost.children[1].children[0],"delete").click(); // delete active CPU run
assert.deepEqual(Array.from(calls.pop()),[primaryA,"",false],
  "Deleting active CPU should reset only dependent selection");
await actionButton(cpuHost.children[1].children[1],"rename").click(); // rename another CPU run
assert.deepEqual(Array.from(calls.pop()),[primaryA,runA,true]);
assert.equal(posts.at(-1).path,"/api/studio/analysis-rename");

pickerCtx.makeAnalysisPicker(ui.analysis,true);
const gpuHost=hosts.at(-1);
await actionButton(gpuHost.children[1].children[1],"delete").click(); // delete other GPU version
assert.deepEqual(Array.from(calls.pop()),[primaryA,runA,true],
  "Deleting another primary analysis must not blank current analysis");
await actionButton(gpuHost.children[1].children[0],"delete").click(); // delete selected GPU
assert.deepEqual(Array.from(calls.pop()),["","",false],
  "Deleting selected GPU correctly clears the affected selections");
assert.equal(src.includes('make("details","analysis-picker-actions")'),false,
  "Nested menu must not be reintroduced");

// Test metadata refresh without destroying the running preview.
const counters={reset:0,page:0,timeline:0,events:0};
const s={project:"abc",projectVideos:["match.mp4"],analysis:primaryA,run:runA,
  chain:chain([primaryA,primaryB],{[primaryA]:[run(runA)],[primaryB]:[run(runB)]}),
  chainEpoch:0};
const analysis={options:[],value:primaryA,disabled:false};
const cpu={options:[],value:runA,disabled:false};
const refreshed=chain([primaryA],{[primaryA]:[run(runA)]});
const overview={schema:"volley-native-analysis-overview/1",project_id:"abc",
  video:"match.mp4",metadata_only:true};
const options=(target,rows,key,label)=>{
  target.options=rows.map(row=>({value:row[key],textContent:label(row)}));
  target.value="";
};
const context={s,ui:{analysis,run:cpu,video:{currentTime:0}},overlay:{raw:false},
  ensurePreviewVideo(){},setOverlaySource(){},updateRawDetectorLabel(){},prefetchOverlayAtFrame(){},fps:()=>30,rawFusion:()=>null,
  window:{VolleyballUi:{ready:Promise.resolve()},
    VolleyStudioChain:{stamp:()=>"",build:()=>refreshed,latestRun},
    dispatchEvent:()=>counters.events++},
  CustomEvent:class{constructor(type){this.type=type;}},
  reset(){counters.reset++;s.run="";},
  options,refreshPicker(){},chainSnapshot:{read:()=>null,save(){}},
  request:async path=>({schema:"volley-studio-pipeline/1",project_id:new URL(path,"http://localhost").searchParams.get("project_id"),status:"idle"}),
  previewCache:{load:async path=>path.includes("/runs")?{runs:[]}:overview},
  url:(path,params)=>path+"?"+new URLSearchParams(params),
  sourceSelection:{save(){}},
  loadPage:()=>{counters.page++;return Promise.resolve();},
  loadTimeline:()=>{counters.timeline++;return Promise.resolve();},
  message(){},failed:e=>{throw e;},console};
runInNewContext(slice("  const setRuns=async(","  const projects=async(")+
  "\nthis.loadChain=loadChain;",context);
await context.loadChain(primaryA,runA,true);
assert.equal(counters.reset,0,"Refreshing other versions must never reset preview");
assert.equal(counters.page,0,"No preview reload for unrelated delete");
assert.equal(counters.events,0,"No GPU/CPU change event for unrelated delete");
assert.equal(s.analysis,primaryA);
assert.equal(s.run,runA);
assert.equal(analysis.value,primaryA);
assert.equal(cpu.value,runA);
assert.equal(cpu.options.length,1,"Deleted result must disappear from picker");

// Removing the active GPU version must select a surviving GPU and its linked CPU.
context.window.VolleyStudioChain.build=()=>chain([primaryB],{[primaryB]:[run(runB)]});
await context.loadChain("", "", false);
assert.ok(counters.reset>0,"Active GPU deletion should reset old preview");
assert.equal(s.analysis,primaryB);
assert.equal(s.run,runB);
// A project may have several GPU versions across multiple videos. The latest
// real source wins when no valid explicit selection exists, regardless of order.
const choices=chain([primaryA,primaryB],{
  [primaryA]:[run(runA)],[primaryB]:[run(runB)]
});
choices.choices[0].created_at=100;
choices.choices[1].created_at=300;
context.window.VolleyStudioChain.build=()=>choices;
await context.loadChain("", "", false);
assert.equal(s.analysis,primaryB,"Most recently created GPU analysis must be selected");
assert.equal(analysis.value,primaryB);
await context.loadChain(primaryA,runA,false);
assert.equal(s.analysis,primaryA,"A valid manually chosen GPU analysis must be retained");
assert.equal(s.run,runA);

// Deleting the selected newest source falls back to the previous creation.
context.window.VolleyStudioChain.build=()=>chain([primaryA],{[primaryA]:[run(runA)]});
await context.loadChain("", "", false);
assert.equal(s.analysis,primaryA);
assert.equal(s.run,runA);
context.window.VolleyStudioChain.build=()=>chain([],{});
await context.loadChain("", "", false);
assert.equal(s.analysis,"","An empty project must not invent a GPU source");
assert.equal(analysis.disabled,true,"No primary analyses must disable the selector");
assert.equal(cpu.disabled,true,"No primary analysis means no CPU snapshots");

// A populated primary selector must not allow a user to clear it to blank.
const optionContext={
  make:(tag,cls,label)=>({tag,value:"",textContent:label}),
  wipe:select=>{select.children=[];}
};
runInNewContext(slice("  const options=(select,rows","  // The linked source chain")+
  "\nthis.options=options;",optionContext);
const picker={children:[],append(child){this.children.push(child);}};
optionContext.options(picker,[choice(primaryA)],"value",x=>x.label,"Выберите",false);
assert.deepEqual(picker.children.map(x=>x.value),[primaryA]);
optionContext.options(picker,[],"value",x=>x.label,"Нет GPU-анализов",false);
assert.equal(picker.children.length,1);
assert.equal(picker.children[0].value,"");

// A completed pipeline selects the exact newly created fusion, not another
// version chosen by a fallback during the metadata refresh.
const completeHandlers=new Map();
const completeState={project:"abc",analysis:primaryA,chain:chain([primaryA,primaryB],{})};
const completeUi={analysis:{value:primaryA}};
let chainLoaded=0;
const completeContext={
  window:{addEventListener:(name,callback)=>completeHandlers.set(name,callback)},
  s:completeState,ui:completeUi,invalidateCatalog(){},
  loadChain:async()=>{chainLoaded++;},setAnalysis:async()=>{completeState.analysis=completeUi.analysis.value;},
  failed:error=>{throw error;}
};
runInNewContext(slice("  // One authoritative place to select the new GPU -> CPU lineage:",
  "  window.addEventListener(\"volley:open-legacy-project\"") ,completeContext);
await completeHandlers.get("volley:pipeline-complete")({
  detail:{project_id:"abc",video:"match.mp4",analysis_id:"b",postprocessing_id:null}
});
assert.equal(chainLoaded,1);
assert.equal(completeState.analysis,primaryB);

// Deleting the last GPU chain must trigger the analysis button state update.
const tabSource=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const start=tabSource.indexOf('  window.addEventListener("volley:chain-change"');
const end=tabSource.indexOf('  window.addEventListener("volley:track-photo-covers"',start);
assert.ok(start>=0&&end>start);
let updates=0,chainHandler;
runInNewContext(tabSource.slice(start,end),{
  window:{addEventListener:(_,handler)=>{chainHandler=handler;}},
  state:{tab:"analysis"},projectId:()=>"abc",
  ui:{video:{options:[],value:""}},updateProfile:()=>{updates++;},renderGraphSummary:()=>{}
});
chainHandler({detail:{project_id:"abc",video:""}});
assert.equal(updates,1,"Removing all analyses must update postprocess availability");

console.log("PASS: primary analysis selection, completion, deletion and empty-state guards.");

const freshCalls=[];
context.previewCache.load=async(path,ttl,persist,fresh)=>{
  freshCalls.push({path,fresh});
  if(path.includes("/runs"))return {runs:[]};
  throw Error("Temporary overview failure");
};
let saves=0,builds=0;
context.chainSnapshot.save=()=>{saves++;};
context.window.VolleyStudioChain.build=()=>{builds++;return chain([primaryA],{});};
await assert.rejects(context.loadChain("","",false),/индекс|стадии|каталог/i);
assert.equal(builds,0,"Incomplete source catalog must not fabricate an orphan analysis");
assert.equal(saves,0,"Incomplete source catalog must not be cached");
assert.ok(freshCalls.every(call=>call.fresh===true),"Catalog revalidation must bypass stale metadata cache");

let release;
context.window.VolleyballUi.ready=new Promise(resolve=>{release=resolve;});
s.project="abc";s.analysis=primaryA;s.run=runA;
analysis.value=primaryA;cpu.value=runA;
let reads=0;
context.chainSnapshot.read=()=>{reads++;return null;};
context.previewCache.load=async path=>path.includes("/runs")?{runs:[]}:overview;
context.window.VolleyStudioChain.build=()=>chain([], {});
const stale=context.loadChain(primaryA,runA,false,true);
assert.equal(s.analysis,"","Old analysis must be cleared before WASM readiness");
assert.equal(s.run,"","Old CPU run must be cleared before WASM readiness");
s.project="other";
s.projectVideos=[];
const current=context.loadChain("","",false,true);
release();
await Promise.all([stale,current]);
assert.equal(reads,0,"Explicit project switching must not restore a persisted snapshot");
assert.equal(s.project,"other");
assert.equal(s.analysis,"");

s.project="abc";s.projectVideos=["match.mp4"];
s.pipelineBlocks=new Map();s.pipelineCompleted=new Map();
context.request=async()=>({schema:"volley-studio-pipeline/1",project_id:"abc",id:"active",status:"running"});
context.window.VolleyStudioChain.build=()=>chain([primaryA],{[primaryA]:[run(runA)]});
context.chainSnapshot.read=()=>chain([primaryA],{[primaryA]:[run(runA)]});
await context.loadChain(primaryA,runA,false);
assert.equal(s.run,"","Running analysis must not restore a historical CPU snapshot");
assert.equal(s.analysis,"","Running analysis must not restore a historical GPU snapshot");
assert.equal(cpu.disabled,true);
assert.equal(s.pipelineBlocks.get("abc"),"active");
context.request=async()=>({schema:"volley-studio-pipeline/1",project_id:"abc",id:"active",status:"cancelled"});
await context.loadChain(primaryA,runA,false);
assert.equal(s.run,"","Cancelled analysis must not automatically resurrect previous results");
assert.equal(s.pipelineBlocks.has("abc"),false);
assert.equal(s.analysis,"","Cancelled analysis must keep the preview empty");
assert.equal(analysis.disabled,false,"Historical analyses remain available for explicit selection");

context.request=async()=>({schema:"volley-studio-pipeline/1",project_id:"abc",id:"new",status:"running"});
await context.loadChain(primaryA,runA,false);
context.request=async()=>({schema:"volley-studio-pipeline/1",project_id:"abc",id:"new",status:"complete",
  video:"match.mp4",analysis_id:"b",postprocessing_id:runB});
context.window.VolleyStudioChain.build=()=>chain([primaryA,primaryB],{[primaryA]:[run(runA)],[primaryB]:[run(runB)]});
await context.loadChain(primaryA,runA,false);
assert.equal(s.analysis,primaryB);
assert.equal(s.run,runB,"Completion must select the actual new CPU run");

context.request=async()=>({schema:"volley-studio-pipeline/1",project_id:"abc",id:"gpu",status:"running"});
await context.loadChain(primaryB,runB,false);
context.request=async()=>({schema:"volley-studio-pipeline/1",project_id:"abc",id:"gpu",status:"complete",
  video:"match.mp4",analysis_id:"b",postprocessing_id:null});
await context.loadChain(primaryB,runB,false);
assert.equal(s.analysis,primaryB);
assert.equal(s.run,"","GPU-only completion must not select a historical CPU run");
await assert.rejects(context.loadChain(primaryA,"c".repeat(32),false,true,true),/Новая версия анализа/);
assert.equal(s.run,"","Missing exact CPU result must not fall back to older results");

let clears=0;
const canvas={get width(){return 640;},set width(value){assert.equal(value,640);clears++;}};
const overlayContext={overlay:{source:{},canvas,controllers:new Map(),fetching:new Set(),
  cache:new Map(),rawFrames:new Map(),failed:new Map(),presentedTime:9},
  overlayMessage(){},invalidateOverlayLabels(){},drawOverlay(){}};
runInNewContext(slice("  const setOverlaySource=",'  window.addEventListener("volley:court-calibration-updated"')+
  "\nthis.setOverlaySource=setOverlaySource;",overlayContext);
overlayContext.setOverlaySource(null);
assert.equal(clears,1,"A hidden canvas must clear even when it has zero client dimensions");
assert.equal(overlayContext.overlay.source,null);
console.log("PASS: active/failed/completed analysis preview isolation and hidden-canvas clearing.");
