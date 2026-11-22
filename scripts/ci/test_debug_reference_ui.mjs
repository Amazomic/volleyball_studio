import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read = file => readFileSync(new URL("../../" + file, import.meta.url), "utf8");
const html = read("rust/volley-app/static/studio.html");
const css = read("rust/volley-app/static/studio.css");
const js = read("rust/volley-app/static/studio.js");
const rust = read("rust/volley-app/src/touch_annotation.rs");
const main = read("rust/volley-app/src/main.rs");

for (const id of [
  "studio-debug-mode-analysis", "studio-debug-mode-reference",
  "studio-debug-reference-picker", "studio-debug-reference-video",
  "studio-debug-reference-counts",
]) {
  assert.ok(html.includes('id="' + id + '"'), "Missing reference-mode control " + id);
}
assert.match(html, /class="studio-debug-action-group"/);
assert.match(html, /class="studio-debug-settings-group"/);
assert.match(css, /\.studio-debug-action-group\{/);
assert.match(css, /\.studio-reference-card\{/);
assert.match(js, /const refreshReferenceCatalog=async/);
assert.match(js, /const renderReferenceCards=\(\)=>/);
assert.match(js, /reference\.active\|\|timelineRallyLayers\.touches\.checked/);
assert.match(js, /url\("\/api\/training\/annotations\/catalog"/);
assert.match(main, /"\/api\/training\/annotations\/catalog"/);
assert.match(rust, /pub async fn catalog\(/);
assert.match(rust, /\.filter\(\|v\|\s*v\.project_id == project_id && v\.reviewed\)/);

const start = js.indexOf("  const renderReferenceCards=()=>");
const end = js.indexOf("  const cards=()=>",start);
assert.ok(start>0&&end>start);
const list = {children:[],append(item){this.children.push(item);}};
const make = (tag,className="",textContent="")=>({
  tag,className,textContent,children:[],handlers:{},
  append(...items){this.children.push(...items);},
  setAttribute(){},
  addEventListener(event,callback){this.handlers[event]=callback;}
});
const s={video:"sample.mp4",run:"run_1",analysisOffset:10};
const reference={
  entries:[{video:"sample.mp4",rally_run_id:"run_1"}],
  counts:{textContent:""}
};
const annotation={
  data:{reviewed:true,
    rallies:[{id:"r1",start:30,end:90}],
    touches:[{id:"t1",frame:45},{id:"t2",frame:130}]},
  rallyId:null,touchId:null
};
const seeks=[];
const {renderReferenceCards}=runInNewContext(js.slice(start,end)+
  "\n({renderReferenceCards})",{
    reference,s,annotation,annotationSourceValid:()=>true,
    annotationReady:()=>true,
    annotationSorted:type=>[...annotation.data[type]],
    timeLabel:value=>String(value),
    fps:()=>30,ui:{list},make,
    seekTimeline:time=>seeks.push(time),cards:()=>{}
  });
renderReferenceCards();
assert.equal(list.children.length,2,"Rally plus an out-of-rally touch must be visible");
assert.match(reference.counts.textContent,/Розыгрыши: 1 · касания: 2/);
const card=list.children[0];
const touch=card.children[2].children[0];
assert.equal(touch.textContent,"● 11.5");
touch.handlers.click({stopPropagation(){}});
assert.equal(seeks[0],11.5,"Reference touch seeks the original video with offset");
assert.equal(annotation.data.touches[0].frame,45,"Viewing must not mutate a touch");
annotation.data.reviewed=false;
list.children.length=0;
renderReferenceCards();
assert.equal(list.children.length,1);
assert.equal(list.children[0].className,"empty-state",
  "An unreviewed annotation must not be exposed as a verified reference");

console.log("Debug reference UI: grouped controls, verified videos, read-only cards and touch navigation: OK");
