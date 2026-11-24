import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const start=source.indexOf("  const loadVideos=async()=>{");
const end=source.indexOf("  const launchPipeline=async(",start);
assert.ok(start>=0&&end>start);

const first="a".repeat(12),second="b".repeat(12);
const catalogs=[
  {id:first,videos:["match-1.mp4","match-2.mp4"]},
  {id:second,videos:["second-1.mp4","second-2.mp4"]},
  {id:"c".repeat(12),videos:[]}
];
const cache=new Map();
let project=first,activeChain="",currentToken=0;
const editorLoads=[];
const video={value:"",options:[],append(node){this.options.push(node);}};
const state={tab:"analysis",analysisGeneration:0,formProject:""};
const context={
  state,ui:{video},projectId:()=>project,
  runtime:{
    beginLoad:()=>++currentToken,
    currentLoad:(_key,token)=>token===currentToken
  },
  loadHistory(){},resetLog(){},updateProfile(){},renderGraphSummary(){},
  clear:node=>{node.options=[];node.value="";},
  make:(_tag,_class,label)=>({value:"",textContent:label}),
  window:{
    VolleyballStudioProjectCatalog:async()=>({projects:catalogs}),
    VolleyballEditor:{load:(...args)=>editorLoads.push(args)}
  },
  api:async()=>{throw Error("Unexpected catalog fetch");},
  restoreAnalysisSettings:key=>cache.get(key)||"",
  $:id=>id==="studio-inference"?{value:activeChain}:null,
  restorePipeline:async()=>{}
};
runInNewContext(source.slice(start,end)+"\nthis.loadVideos=loadVideos;",context);

await context.loadVideos();
assert.equal(video.value,"match-1.mp4","New multi-video project must select its first video");
assert.equal(video.options.length,2,"Populated selector must not contain an empty option");
assert.ok(video.options.every(option=>option.value),"Every option must be a video");
assert.deepEqual(Array.from(editorLoads.at(-1)),[first,"match-1.mp4",true]);

video.value="match-2.mp4";
await context.loadVideos();
assert.equal(video.value,"match-2.mp4","Refresh retains manually chosen video");

state.formProject="";
video.value="";
cache.set(first,"match-2.mp4");
await context.loadVideos();
assert.equal(video.value,"match-2.mp4","Saved project preference survives reload");

state.formProject="";
cache.set(first,"removed.mp4");
await context.loadVideos();
assert.equal(video.value,"match-1.mp4","Removed saved video falls back to first available");

activeChain=JSON.stringify(["match-2.mp4","fusion","abc"]);
await context.loadVideos();
assert.equal(video.value,"match-2.mp4","Selected GPU analysis remains authoritative");

activeChain="";
project=second;
await context.loadVideos();
assert.equal(video.value,"second-1.mp4","Switching projects selects a valid source video");
assert.deepEqual(Array.from(editorLoads.at(-1)),[second,"second-1.mp4",true]);

project="c".repeat(12);
await context.loadVideos();
assert.equal(video.value,"","Empty catalog must not invent a video");
console.log("PASS: analysis video default, persisted selection and project switches.");
