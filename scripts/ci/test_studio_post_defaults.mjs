import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
assert.match(html,/<input id="studio-post-ball-stabilizer-score"[^>]*value="0\.70" data-default="0\.70"/);
const extract=(begin,end)=>{
  const start=source.indexOf(begin),stop=source.indexOf(end,start);
  assert.ok(start>=0&&stop>start,"Cannot locate "+begin);
  return source.slice(start,stop);
};
const storage=new Map();
const localStorage={
  getItem:key=>storage.get(key)??null,
  setItem:(key,value)=>storage.set(key,value)
};
const project="f39dff74992f";
const key="volley:studio-analysis-form:v1:"+project;
const inputs=[
  {id:"studio-post-ball-score",value:"0.35",dataset:{default:"0.35"},checkValidity:()=>true},
  {id:"studio-post-ball-stabilizer-score",value:"0.70",dataset:{default:"0.70"},checkValidity:()=>true}
];
const ui={analysis:{querySelectorAll:()=>inputs},video:{value:"match.mp4"}};
const env={ui,localStorage,$:id=>inputs.find(x=>x.id===id),projectId:()=>project};
runInNewContext(extract("  const analysisFormCache=(()=>{","  const make=(tag,klass,text)=>")+ "\nthis.cache=analysisFormCache;",env);
runInNewContext(extract("  const allProfileFields=()=>","  const selectedGpuSource=()=>")+
  "\nthis.restore=restoreAnalysisSettings;this.save=saveAnalysisSettings;",env);

for(const version of [1,2]){
  storage.set(key,JSON.stringify({version,video:"match.mp4",fields:{
    "studio-post-require-pose":["0","0"],
    "studio-post-ball-score":["0.35","0.44"],
    "studio-post-ball-stabilizer-score":["0.55","0.65"],
    "studio-post-touch-algorithm":["pose_motion","legacy"]
  }}));
  assert.equal(env.restore(project),"match.mp4");
  assert.equal(inputs[0].value,"0.44","Other saved settings must survive migration");
  assert.equal(inputs[1].value,"0.65","Previous custom stabilizer score must survive new default");
  assert.equal(JSON.parse(storage.get(key)).version,3);
  assert.equal(JSON.parse(storage.get(key)).fields["studio-post-require-pose"],undefined);
}
env.save();
assert.equal(JSON.parse(storage.get(key)).fields["studio-post-require-pose"],undefined,
  "Removed pose policy must never return to storage");
assert.equal(JSON.parse(storage.get(key)).fields["studio-post-touch-algorithm"],undefined,
  "Removed algorithm selector must not return to storage");
assert.equal(env.restore(project),"match.mp4");
assert.equal(inputs[0].value,"0.44");
assert.equal(inputs[1].value,"0.65");
assert.deepEqual(JSON.parse(storage.get(key)).fields["studio-post-ball-stabilizer-score"],
  ["0.70","0.65"],"Saved manual threshold must migrate to the new baseline");
storage.delete(key);
assert.equal(env.restore(project),"");
assert.equal(inputs[1].value,"0.70","New analysis must default to 70% confident anchors");
console.log("PASS: 70% ball anchors, preserved prior custom threshold and v1/v2 form migrations.");