import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const markup=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const tabs=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
const beginning=markup.indexOf('<section id="studio-analysis-view"');
const finish=markup.indexOf('<section id="studio-players-view"',beginning);
assert.ok(beginning>=0&&finish>beginning,"Real analysis view must exist");
const view=markup.slice(beginning,finish);
assert.doesNotMatch(view,/02 \/ ПОЛНЫЙ ЦИКЛ/);
assert.doesNotMatch(view,/studio-post-ball-stabilizer-score/);
assert.match(css,/@property --stage-progress\{syntax:"<percentage>"/);
assert.match(css,/analysis-stage-pulse 2\.2s ease-in-out infinite/);
assert.match(css,/analysis-flow-node\{.*padding:7px 12px/);

// All parameter names have visible hover/focus explanations and a11y descriptions.
const labels=[...view.matchAll(/<label\b([^>]*data-analysis-hint="([^"]+)"[^>]*)>([\s\S]*?)<\/label>/g)];
assert.equal(labels.length,42,"Tooltip inventory changed: audit new setting labels");
const ids=new Set();
for(const [,attrs,hint,body] of labels){
  assert.ok(hint.length>=50,"Explanations must state the effect, not only repeat the title");
  assert.match(attrs,/tabindex="0"/);
  assert.match(attrs,/aria-description="[^"]+"/);
  assert.match(attrs,/data-analysis-hint-en="[^"]{50,}"/,
    "English-mode tooltips must not disappear");
  const target=attrs.match(/\bfor="([^"]+)"/)?.[1]||
    body.match(/\bid="(studio-[^"]+)"/)?.[1];
  assert.ok(target,"Tooltip must belong to a concrete setting: "+body);
  assert.ok(!ids.has(target),"Duplicate tooltip for "+target);
  ids.add(target);
}
assert.match(css,/#studio-analysis-view \.analysis-setting-help:is\(:hover,:focus-visible\)::after/);
assert.match(css,/html:lang\(en\) #studio-analysis-view/);
const uiSource=readFileSync(new URL("../../rust/volley-app/static/studio_ui.js",import.meta.url),"utf8");
assert.match(uiSource,/label\.setAttribute\("aria-description",description\)/);
assert.ok(!view.includes('id="studio-post-tcn-experimental"'),
  "Removed ineffective pose switch must not be offered");

// Run actual profile constructors, and prove that changing EVERY displayed
// analysis profile control modifies the outgoing native Rust request profile.
const start=tabs.indexOf("  const gpuProfile=()=>({");
const stop=tabs.indexOf("  const allProfileFields=()=>",start);
assert.ok(start>=0&&stop>start);
const controls=[...view.matchAll(/<(input|select)\b([^>]*\bdata-default="([^"]+)"[^>]*)>/g)]
  .map(([_,kind,attributes,baseline])=>{
    const id=attributes.match(/\bid="([^"]+)"/)?.[1];
    const min=Number(attributes.match(/\bmin="([^"]+)"/)?.[1]);
    const max=Number(attributes.match(/\bmax="([^"]+)"/)?.[1]);
    assert.ok(id);
    return {id,kind,baseline,min,max,check:attributes.includes('type="checkbox"')};
  });
assert.equal(controls.length,39,"Unexpected primary and CPU profile controls");
const state=new Map(controls.map(c=>[c.id,c.baseline]));
const node=id=>({value:state.get(id),checked:state.get(id)==="1"});
const scope={
  $:node,
  inputValue:id=>Number(node(id).value)
};
runInNewContext(tabs.slice(start,stop)+"\nthis.profiles={gpuProfile,postProfile};",scope);
const snapshot=()=>JSON.stringify({gpu:scope.profiles.gpuProfile(),post:scope.profiles.postProfile()});
const original=snapshot();
assert.equal(scope.profiles.postProfile().touches.algorithm,"ball_trajectory");
assert.equal(scope.profiles.postProfile().pose_classification,true);
for(const c of controls){
  assert.ok(ids.has(c.id),"Missing help for real profile input "+c.id);
  if(c.id==="studio-gpu-stride"){
    assert.equal(c.baseline,"9");
    assert.match(view,/id="studio-gpu-stride"[^>]*readonly/);
    continue;
  }
  const next=c.check?(c.baseline==="1"?"0":"1"):
    c.kind==="select"?(c.baseline==="1"?"0":"1"):
    Number.isFinite(c.max)&&c.max!==Number(c.baseline)?String(c.max):
    String(c.min);
  assert.notEqual(next,c.baseline,"Test must really change "+c.id);
  state.set(c.id,next);
  assert.notEqual(snapshot(),original,"Setting "+c.id+" does not enter the native request");
  state.set(c.id,c.baseline);
}
assert.equal(snapshot(),original,"Settings fixture must restore exactly");
for(const id of ["studio-analysis-trim-start","studio-analysis-trim-end"]){
  assert.ok(ids.has(id),"Editor trim labels need accurately scoped tooltips");
}
console.log("Analysis settings tooltips: "+labels.length+" labels, "+controls.length+
  " profile controls correctly map to native request, obsolete switch absent: OK");
