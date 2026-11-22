// Regression: project picker reads live registry, selects most recent, opens duplicates.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {join} from "node:path";
import vm from "node:vm";

const repo=join(fileURLToPath(new URL("../..",import.meta.url)));
const src=readFileSync(join(repo,"rust/volley-app/static/studio.js"),"utf8");
const catalogLine=src.match(/window\.VolleyballStudioProjectCatalog=\(\)=>request\("\/api\/projects"\);/)?.[0];
assert.ok(catalogLine,"project registry must bypass the preview cache");

const entries=[
  {id:"111111111111",name:"previous",videos:[]},
  {id:"222222222222",name:"middle",videos:[]},
  {id:"333333333333",name:"latest",videos:["match.mp4"]},
];
let reads=0;
const window={
  dispatchEvent(){},
};
vm.runInNewContext(catalogLine,{window,request:async path=>{
  assert.equal(path,"/api/projects");
  reads++;
  return {projects:entries};
}});
await window.VolleyballStudioProjectCatalog();
await window.VolleyballStudioProjectCatalog();
assert.equal(reads,2,"catalog should fetch a fresh registry each time");

const start=src.indexOf("  const projects=async(");
const end=src.indexOf("  const loadMarks=",start);
assert.ok(start>0&&end>start);
const state={
  project:"",analysis:"",run:"",catalogEpoch:0,projectCatalog:[],projectVideos:[]
};
const selector={value:""};
let saved={project:"",analysis:"",run:""};
let calls=0;
const pickerContext={
  s:state,reference:{active:false},ui:{project:selector},window,Event:class{constructor(type){this.type=type;}},
  sourceSelection:{read:()=>saved},
  options:(select,rows)=>{select.options=rows.map(row=>({value:row.id}));},
  refreshPicker:()=>{},
  loadChain:async()=>{calls++;},
};
vm.runInNewContext(src.slice(start,end)+"\nthis.readProjects=projects;",pickerContext);
await pickerContext.readProjects();
assert.equal(state.project,"333333333333");
assert.equal(selector.value,"333333333333");
assert.deepEqual([...state.projectVideos],["match.mp4"]);
assert.equal(calls,1);
state.project="";
saved={project:"111111111111",analysis:"",run:""};
await pickerContext.readProjects();
assert.equal(state.project,"111111111111","persisted valid project takes priority");

const formStart=src.indexOf('    ui.createForm.addEventListener("submit"');
const formEnd=src.indexOf('  ui.project.addEventListener("change"',formStart);
assert.ok(formStart>0&&formEnd>formStart);
const completeForm=src.slice(formStart,formEnd);
assert.match(completeForm,/  }\r?\n$/);
const formSnippet=completeForm.replace(/  }\r?\n$/,"");
let onSubmit=null,closed=0,opened="",navigate=0,reported="",errorShown="";
let catalogReads=0;
const existingId="333333333333";
const formContext={
  ui:{
    createForm:{addEventListener:(event,fn)=>{assert.equal(event,"submit");onSubmit=fn;}},
    projectName:{value:"LATEST"},projectCancel:{disabled:false},
    projectClose:{disabled:false},projectCreateError:{textContent:""},projectCreateStatus:{textContent:""},
    projectDialog:{open:true,close(){closed++;this.open=false;}}
  },
  creating:false,uploading:false,dialogEpoch:1,
  selectedVideos:()=>["match.mp4"],selectedPlayers:()=>[],
  updateSubmit:()=>{},post:async()=>{throw Error("Проект с таким названием уже существует");},
  invalidateCatalog:()=>{},
  window:{VolleyballStudioProjectCatalog:async()=>{
    catalogReads++;
    return {projects:entries};
  }},
  projects:async id=>{opened=id;},
  document:{getElementById:id=>{assert.equal(id,"studio-nav-analysis");return {click:()=>{navigate++;}};}},
  message:m=>{reported=m;},
  alertCreate:e=>{errorShown=e.message;},
  console,
};
vm.runInNewContext(formSnippet,formContext);
assert.equal(typeof onSubmit,"function");
await onSubmit({preventDefault(){}});
assert.equal(opened,existingId);
assert.equal(closed,1);
assert.equal(navigate,1);
assert.equal(catalogReads,1);
assert.match(reported,/открыт существующий/);
assert.equal(errorShown,"");
assert.equal(formContext.creating,false);
console.log("PASS: live project catalog, latest fallback, existing-name selection.");
