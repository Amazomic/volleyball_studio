import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const root=new URL("../../rust/volley-app/static/",import.meta.url);
const html=readFileSync(new URL("studio.html",root),"utf8");
const source=readFileSync(new URL("studio_tabs.js",root),"utf8");
for(const id of ["studio-gpu-interpolate-players","studio-gpu-interpolate-pose"]){
  assert.match(html,new RegExp('<input id="'+id+'"[^>]*type="checkbox"[^>]*checked[^>]*data-default="1"'));
}
const slice=(begin,end)=>{
  const start=source.indexOf(begin),stop=source.indexOf(end,start);
  assert.ok(start>=0&&stop>start,begin+" section");
  return source.slice(start,stop);
};
const values={
  "studio-gpu-stride":9,"studio-gpu-memory":0,"studio-gpu-threads":0,
  "studio-gpu-person-stride":10,
};
const boxes={
  "studio-gpu-pose-enabled":{value:"1"},
  "studio-gpu-parallel-rally-poses":{type:"checkbox",checked:true,dataset:{default:"1"},id:"studio-gpu-parallel-rally-poses",checkValidity:()=>true},
  "studio-gpu-threads":{type:"number",value:"0",dataset:{default:"0"},id:"studio-gpu-threads",checkValidity:()=>true},
  "studio-gpu-interpolate-players":{type:"checkbox",checked:true,dataset:{default:"1"},id:"studio-gpu-interpolate-players",checkValidity:()=>true},
  "studio-gpu-interpolate-pose":{type:"checkbox",checked:true,dataset:{default:"1"},id:"studio-gpu-interpolate-pose",checkValidity:()=>true}
};
const state={saved:null};
const context={
  $:id=>boxes[id]??{value:""},inputValue:id=>values[id],
  ui:{analysis:{querySelectorAll:()=>[boxes["studio-gpu-parallel-rally-poses"],boxes["studio-gpu-interpolate-players"],boxes["studio-gpu-interpolate-pose"],boxes["studio-gpu-threads"]]},video:{value:"clip.mp4"}},
  projectId:()=>"abc123def456",
  analysisFormCache:{write:(project,video,fields)=>{state.saved={project,video,fields};},
    read:()=>({video:state.saved?.video,fields:state.saved?.fields})}
};
runInNewContext(slice("  const gpuProfile=()","  const allProfileFields=()")+
  slice("  const allProfileFields=()","  const selectedGpuSource=()")+
  "\nthis.api={gpuProfile,postProfile,saveAnalysisSettings,restoreAnalysisSettings,fieldValue,applyFieldValue};",context);
const api=context.api;
assert.equal(api.postProfile().interpolate_players,true);
assert.equal(api.postProfile().interpolate_pose,true);
assert.equal(api.gpuProfile().pose_stride,10);
assert.ok(html.indexOf('id="studio-gpu-interpolate-players"')>html.indexOf('id="studio-post-heading"'));
boxes["studio-gpu-interpolate-players"].checked=false;
boxes["studio-gpu-parallel-rally-poses"].checked=false;
api.saveAnalysisSettings();
assert.equal(state.saved.fields["studio-gpu-parallel-rally-poses"][1],"0");
assert.equal(state.saved.fields["studio-gpu-interpolate-players"][1],"0");
assert.ok(!("studio-gpu-interpolate-pose" in state.saved.fields));
boxes["studio-gpu-interpolate-players"].checked=true;
api.restoreAnalysisSettings("abc123def456");
assert.equal(api.gpuProfile().parallel_rally_poses,false);
assert.equal(api.postProfile().interpolate_players,false);
assert.equal(api.postProfile().interpolate_pose,true);
state.saved={video:"clip.mp4",fields:{"studio-gpu-threads":["8","12"]}};
api.restoreAnalysisSettings("abc123def456");
assert.equal(boxes["studio-gpu-threads"].value,"12","Preserve saved manual CPU limits from older defaults");
boxes["studio-gpu-interpolate-players"].checked=true;
boxes["studio-gpu-interpolate-pose"].checked=false;
const post=api.postProfile();
assert.equal(post.interpolate_players,true);
assert.equal(post.interpolate_pose,false);
assert.ok(!("interpolate_players" in api.gpuProfile()));
assert.ok(!("interpolate_pose" in api.gpuProfile()));
console.log("CPU player interpolation and pose-preview interpolation: defaults and saved settings OK");
