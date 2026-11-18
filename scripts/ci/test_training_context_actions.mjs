import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const js=read("rust/volley-app/static/studio.js");
const html=read("rust/volley-app/static/studio.html");
const css=read("rust/volley-app/static/studio.css");
assert.match(html,/id="studio-comparison-recalculate">Смоделировать розыгрыши и касания/);
assert.ok(html.includes('id="studio-annotation-context-actions"'));
assert.ok(css.includes(".video-wrap.training-annotation #studio-video-overlay"));
assert.ok(js.includes("showAnnotationContext(item.frame,e,item.id)"));
assert.ok(js.includes("comparison.data.simulated_touch_candidates"));

class Node {
  constructor() {
    this.children=[];
    this.listeners={};
    this.textContent="";
    this.style={};
    this.className="";
    const classes=new Set(["hidden"]);
    this.classList={
      add:value=>classes.add(value),
      remove:value=>classes.delete(value),
      contains:value=>classes.has(value)
    };
  }
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=[...items];}
  addEventListener(name,callback){this.listeners[name]=callback;}
  click(){this.listeners.click?.({stopPropagation:()=>{}});}
}
const elements=new Map();
const $=id=>{
  if(!elements.has(id))elements.set(id,new Node());
  return elements.get(id);
};
const canvas=new Node();
const wrap={getBoundingClientRect:()=>({left:0,top:0,width:800,height:460})};
canvas.closest=()=>wrap;
const input=[];
let paused=false,keys;
const annotation={
  busy:false,touchId:"manual-1",rallyId:null,pendingStart:null,
  data:{total_frames:1500,touches:[{id:"manual-1",frame:30}],
    rallies:[{id:"r1",start:10,end:80}]}
};
const comparison={simulated:true,data:{
  schema:"volley-rally-training-comparison/1",
  simulated_touch_candidates:[120]
}};
const context={
  $,document:{createElement:()=>new Node(),
    addEventListener:(name,callback)=>{keys=callback;}},
  comparison,comparisonMatches:()=>true,comparisonTime:frame=>(frame/30).toFixed(2)+" с",
  annotation,annotationReady:()=>Boolean(annotation.data),
  training:{active:true,poseAdvanced:{open:false}},
  s:{project:"p",video:"v",run:"r"},
  ui:{video:{pause:()=>{paused=true;}}},
  overlay:{canvas},
  timeline:{data:{total_frames:1500}},
  editAnnotation:(action,fields)=>{input.push({action,fields});return Promise.resolve();},
  annotationFrame:()=>120,drawAnnotation:()=>{}
};
const start=js.indexOf("  const annotationContext={");
const end=js.indexOf("  const pickTrainingPlayer=event=>",start);
assert.ok(start>0&&end>start);
const {showAnnotationContext}=runInNewContext(js.slice(start,end)+
  "\n({showAnnotationContext})",context);
const actions=()=>elements.get("studio-annotation-context-actions").children;
const click=label=>{
  const item=actions().find(item=>item.textContent===label);
  assert.ok(item,"Action should be available: "+label);
  item.click();
};
showAnnotationContext(30,{clientX:340,clientY:220},"manual-1");
assert.equal(paused,true);
assert.equal(elements.get("studio-annotation-context").classList.contains("hidden"),false);
click("Удалить касание");
assert.equal(input.at(-1).action,"delete_touch");
assert.equal(input.at(-1).fields.id,"manual-1");
assert.equal(elements.get("studio-annotation-context").classList.contains("hidden"),true);
showAnnotationContext(120,{clientX:540,clientY:220});
assert.match(elements.get("studio-annotation-context-label").textContent,/Кандидат касания/);
click("✓ Добавить кандидат в эталон");
assert.equal(input.at(-1).action,"add_touch");
assert.equal(input.at(-1).fields.frame,120);
showAnnotationContext(40,{clientX:300,clientY:160});
click("Начало розыгрыша здесь");
assert.equal(input.at(-1).action,"move_rally");
assert.deepEqual(JSON.parse(JSON.stringify(input.at(-1).fields)),
  {id:"r1",start:40,end:80});
showAnnotationContext(200,{clientX:160,clientY:350});
click("Начало нового розыгрыша");
assert.equal(annotation.pendingStart,200);
showAnnotationContext(260,{clientX:160,clientY:350});
click("Закончить новый розыгрыш");
await Promise.resolve();
assert.equal(input.at(-1).action,"add_rally");
assert.deepEqual(JSON.parse(JSON.stringify(input.at(-1).fields)),{start:200,end:260});
showAnnotationContext(300,{});
keys({key:"Escape"});
assert.equal(elements.get("studio-annotation-context").classList.contains("hidden"),true);
console.log("Training context actions: video click, simulated touch confirmation, manual touch, rally bounds, keyboard dismissal: OK");
