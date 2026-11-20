import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {join} from "node:path";

const root=fileURLToPath(new URL("../..",import.meta.url));
const read=p=>readFileSync(join(root,p),"utf8");
const html=read("rust/volley-app/static/studio.html");
const js=read("rust/volley-app/static/studio.js");
const rust=read("rust/volley-app/src/training_samples.rs");
const css=read("rust/volley-app/static/studio.css");
assert.ok(!html.includes('id="studio-annotation-touch-lane"'),"Training has no duplicate time ruler");
assert.equal([...html.matchAll(/id="studio-timeline-track"/g)].length,1,"One shared seek ruler");
assert.ok(html.indexOf('id="studio-timeline"')>html.indexOf('id="studio-video"'));
assert.ok(js.includes('lane:$("studio-timeline-track")'),"Annotation ticks use the shared timeline");
assert.ok(!js.includes('training.panel.before(element)'),
  "Training must not relocate the full-width ruler into the player column");
assert.ok(js.includes('if(element.parentElement!==layout)layout.append(element)'),
  "Training and preview must keep the shared ruler in the two-column workspace");
assert.ok(css.includes('grid-area:timeline;grid-column:1 / -1;min-width:0;width:100%'),
  "Full-video ruler must span the entire width of the viewer workspace");
assert.ok(css.includes('grid-template-areas:"player list" "timeline timeline" "training training"'),
  "The training editor must follow the full-width ruler instead of preceding it");
assert.ok(css.includes('grid-template-areas:"player" "timeline" "list" "training"'),
  "Narrow screens must retain the timeline immediately under the video");
assert.match(js,/training\.panel\.classList\.add\("hidden"\);\s*if\(annotation\.priorFlags\)/,
  "Leaving training clears its interactive ticks");

assert.ok(js.includes('querySelectorAll(".studio-annotation-tick,.studio-trajectory-proposal")'),
  "Only temporary overlay markers should be cleared; canvas and playhead remain");
assert.ok(!css.includes('.studio-annotation-touch-lane{'),"Second ruler CSS is removed");
assert.ok(js.includes('action:"relabel"'),"Training examples support correction");
assert.ok(js.includes('action:editing?"replace_touch":"touch"'),"Captured examples support recapture at a new frame");
assert.ok(js.includes('setAnalysis(item.rally_run_id)'),"Example reopening preserves CPU source");
assert.ok(rust.includes('"relabel"')&&rust.includes('"replace_touch"'),"Rust validates training revisions");
import {runInNewContext} from "node:vm";

class Node {
  constructor(name="div"){
    this.tagName=name.toUpperCase();
    this.children=[];this.dataset={};this.style={};this.listeners=new Map();
    this.textContent="";this.disabled=false;this.className="";
    this.classList={add:()=>{},remove:()=>{}};
  }
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  remove(){this.removed=true;}
  querySelectorAll(selector){
    return selector.includes(".studio-annotation-tick")?
      this.children.filter(row=>row.className.startsWith("studio-annotation-tick")&&!row.removed):[];
  }
  getBoundingClientRect(){return {left:10,width:300};}
  addEventListener(event,callback){this.listeners.set(event,callback);}
  setAttribute(){}
  setPointerCapture(){}
}
const canvas=new Node("canvas"),playhead=new Node("div"),lane=new Node();
lane.append(canvas,playhead);
const source={touches:[{id:"touch-1",frame:60}],rallies:[],total_frames:300,reviewed:false};
const edits=[],seeks=[];
const annotation={
  data:source,busy:false,dirty:false,source:["p","cpu","v.mp4"].join("\u0000"),
  touchId:null,rallyId:null,pendingStart:null,prediction:null,
  lane,touchList:new Node(),rallies:new Node(),metrics:new Node(),saveState:new Node(),pending:new Node(),
  seed:new Node(),calculate:new Node(),adopt:new Node(),save:new Node(),
  add:new Node(),move:new Node(),remove:new Node(),next:new Node(),
  begin:new Node(),finish:new Node(),review:new Node(),current:new Node()
};
const context={
  annotation,training:{active:true},ui:{video:{currentTime:2}},
  s:{seekAt:0,project:"p",run:"cpu",video:"v.mp4"},timeline:{data:{rallies:[]}},
  document:{createElement:name=>new Node(name)},fps:()=>30,timelineDuration:()=>10,
  annotationFrame:()=>60,
  editAnnotation:(action,fields)=>edits.push({action,fields}),
  seekTimeline:seconds=>seeks.push(seconds),
  annotationClock:frame=>(frame/30).toFixed(2)+" с"
};
const slice=js.slice(js.indexOf("  const annotationReady="),js.indexOf("  const refreshAnnotation="));
assert.ok(slice.includes('querySelectorAll(".studio-annotation-tick,.studio-trajectory-proposal")'));
const draw=runInNewContext(slice+"\n({drawAnnotation})",context).drawAnnotation;
annotation.touchLayerVisible=false;
draw();
assert.equal(lane.querySelectorAll(".studio-annotation-tick")[0].hidden,true,
  "Redrawing annotations must keep touch markers hidden when the checkbox is off");
annotation.touchLayerVisible=true;
draw();
assert.equal(lane.querySelectorAll(".studio-annotation-tick")[0].hidden,false,
  "Re-enabling the checkbox restores touch markers after redraw");
assert.equal(lane.children.includes(canvas),true,"Timeline canvas is never removed");
assert.equal(lane.children.includes(playhead),true,"Single playhead is preserved");
assert.equal(lane.querySelectorAll(".studio-annotation-tick").length,1);
const tick=lane.querySelectorAll(".studio-annotation-tick")[0];
assert.equal(tick.dataset.trainingTouchId,"touch-1");
const event={button:0,pointerId:4,clientX:160,preventDefault(){},stopPropagation(){}};
tick.listeners.get("pointerdown")(event);
tick.listeners.get("pointermove")(event);
tick.listeners.get("pointerup")(event);
assert.deepEqual(JSON.parse(JSON.stringify(edits)),[{action:"move_touch",fields:{id:"touch-1",frame:150}}]);
assert.deepEqual(seeks,[]);
draw();
assert.equal(lane.children.filter(row=>!row.removed&&row===canvas).length,1);

const layout={children:[],append(element){
  if(element.parentElement&&element.parentElement!==this)
    element.parentElement.children=element.parentElement.children.filter(node=>node!==element);
  element.parentElement=this;
  if(!this.children.includes(element))this.children.push(element);
},querySelector:selector=>selector===".viewer-panel"?player:null};
const player={children:[],append:layout.append};
const ruler={parentElement:player};
const trainingPanel={parentElement:player};
player.children.push(ruler,trainingPanel);
let redraws=0;
const timelineStart=js.indexOf("  const placeTrainingTimeline=");
const timelineEnd=js.indexOf("  const timeLabel=",timelineStart);
const {placeTrainingTimeline}=runInNewContext(
  js.slice(timelineStart,timelineEnd)+"\n({placeTrainingTimeline})",{
    $:id=>id==="studio-timeline"?ruler:null,
    training:{panel:trainingPanel},
    document:{querySelector:selector=>
      selector==="#studio-preview-view .viewer-layout"?layout:null},
    requestAnimationFrame:fn=>fn(),
    drawTimeline:()=>{redraws++;}
  }
);
placeTrainingTimeline(true);
assert.equal(ruler.parentElement,layout,"Training ruler must span the full viewer width");
assert.equal(player.children.includes(ruler),false);
assert.equal(trainingPanel.parentElement,layout,"Editor goes below the full-width ruler");
assert.deepEqual(layout.children,[ruler,trainingPanel],"Training order: player/list, timeline, editor");
placeTrainingTimeline(false);
assert.equal(layout.children.length,1,"Preview/training must not create duplicate rulers");
assert.equal(trainingPanel.parentElement,player,"Leaving training restores original editor parent");
assert.equal(redraws,2,"Switching both modes must recalculate timeline canvas dimensions");
console.log("Training workspace: one full-width ruler, no duplicate seek track, browse and update examples: OK");
