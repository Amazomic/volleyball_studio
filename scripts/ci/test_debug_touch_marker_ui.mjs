import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const read=file=>readFileSync(new URL("../../"+file,import.meta.url),"utf8");
const js=read("rust/volley-app/static/studio.js");
const html=read("rust/volley-app/static/studio.html");
for(const id of ["studio-debug-touch-mark","studio-debug-touch-save","studio-debug-manual-point"]){
  assert.match(html,new RegExp('id="'+id+'"'));
}
const start=js.indexOf("  const annotationContext={");
const end=js.indexOf("  const pickTrainingPlayer=",start);
assert.ok(start>0&&end>start);
class Element {
  constructor(){
    this.children=[];this.listeners={};this.style={};
    this.classList={add(){},remove(){}};
  }
  append(...children){this.children.push(...children);}
  replaceChildren(){this.children.length=0;}
  addEventListener(type,listener){this.listeners[type]=listener;}
  setAttribute(name,value){this[name]=value;}
  getBoundingClientRect(){return {left:100,top:200,width:1280,height:720};}
}
const elements=new Map();
const $=id=>{
  if(!elements.has(id))elements.set(id,new Element());
  return elements.get(id);
};
const wrapper=new Element();
const canvas=new Element();
canvas.closest=()=>wrapper;
const edits=[];
const annotation={data:{touches:[],rallies:[],total_frames:200},
  busy:false,markMode:true,touchId:null,rallyId:null,pendingStart:null,dirty:false};
const reference={active:false};
const training={active:true,poseAdvanced:{open:false}};
const ui={video:{paused:true,pause(){this.paused=true;},videoWidth:1920,videoHeight:1080}};
const {showAnnotationContext}=runInNewContext(js.slice(start,end)+
  "\n({showAnnotationContext})",{
    $,document:{createElement:()=>new Element(),addEventListener(){}},
    overlay:{canvas},annotation,reference,training,ui,
    s:{run:"run",video:"match.mp4"},
    timeline:{data:{total_frames:200}},
    annotationFrame:()=>45,annotationReady:()=>true,annotationSourceValid:()=>true,
    manualTouchKinds:{serve:"Подача",attack:"Атака",dig:"Приём снизу"},
    comparison:{simulated:false,data:null},comparisonMatches:()=>false,
    comparisonTime:frame=>String(frame),editAnnotation:(action,fields)=>edits.push({action,fields}),
    seekTimeline:()=>{},drawAnnotation:()=>{},
    window:{},Number
  });
const click=canvas.listeners.click;
assert.equal(typeof click,"function");
click({clientX:100+1280*.3,clientY:200+720*.6});
const actions=elements.get("studio-annotation-context-actions");
const select=actions.children[0].children[0];
assert.equal(select["aria-label"],"Тип ручного касания");
select.value="serve";
const add=actions.children.find(item=>item.textContent==="+ Касание здесь");
assert.ok(add);
add.listeners.click();
assert.deepEqual(JSON.parse(JSON.stringify(edits[0])),{
  action:"add_touch",fields:{frame:45,kind:"serve",position:[0.3,0.6]}
});
reference.active=true;
click({clientX:300,clientY:400});
assert.equal(edits.length,1,"Reference mode must stay read-only");
console.log("Debug touch marker: point -> kind -> draft, verified reference stays read-only: OK");
