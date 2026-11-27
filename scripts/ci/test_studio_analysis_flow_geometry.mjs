import assert from "node:assert/strict";
import vm from "node:vm";
import {readFileSync} from "node:fs";

const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
const flow=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const rule=selector=>{
  const escaped=selector.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");
  const match=css.match(new RegExp(escaped+"\\{([^}]*)\\}"));
  assert.ok(match,"Missing CSS rule "+selector);
  return match[1];
};
assert.match(rule(".analysis-flow-diagram"),/container-type:inline-size/);
assert.match(rule(".analysis-flow-node"),/min-height:76px/);
for(const selector of [".analysis-flow-node-title",".analysis-flow-node-stats"]){
  const style=rule(selector);
  assert.doesNotMatch(style,/white-space:nowrap|text-overflow:ellipsis|overflow:hidden/);
  assert.match(style,/overflow-wrap:anywhere/);
}
assert.match(css,/@container \(max-width:330px\)\{\.analysis-flow-phase\[data-phase\] \.analysis-flow-cards\{grid-template-columns:minmax\(0,1fr\)/);
assert.doesNotMatch(css,/\.analysis-flow-diagram\{[^}]*min-width:950px/);
assert.match(flow,/phases\.get\(row\)\.append\(cell\)/);
assert.match(rule('.analysis-flow-phase[data-phase="cpu"] .analysis-flow-cards'),/direction:rtl/);
assert.match(rule('.analysis-flow-phase[data-phase="cpu"] .analysis-flow-cell'),/direction:ltr/);

const start=flow.indexOf("const graphRoutes=");
const end=flow.indexOf("const drawGraphEdges=",start);
assert.ok(start>=0&&end>start,"The graph must route arrows between actual stage positions");
const route=vm.runInNewContext(flow.slice(start,end)+"graphRoutes;");
const definition=(begin,end)=>flow.slice(flow.indexOf(begin),flow.indexOf(end,flow.indexOf(begin)));
const stages=vm.runInNewContext(definition("const pipelineStages=","const pipelineNames=")+"pipelineStages;");
const links=vm.runInNewContext(definition("const pipelineEdges=","const graphLinkState=")+"pipelineEdges;");
assert.ok(links.some(([from,to,condition])=>from==="touch_detector"&&to==="touches"&&!condition),"Touch Attribution must directly consume Touch Detector candidates");
assert.ok(links.some(([from,to,condition])=>from==="ball_stabilizer"&&to==="touch_detector"&&!condition),"Touch Detector must consume stabilized trajectory");
const uneven=new Map([
  ["a",{l:20,r:120,t:30,b:106}],
  ["b",{l:140,r:240,t:30,b:130}],
  ["pose",{l:260,r:360,t:30,b:106}],
  ["tracking",{l:220,r:360,t:220,b:312}],
  ["touches",{l:60,r:200,t:220,b:312}]
]);
const checkRoute=(from,to,expected)=>{
  const points=route(uneven,[[from,to]],380)[0];
  assert.equal(JSON.stringify(points),JSON.stringify(expected));
};
checkRoute("a","b",[[120,68],[130,68],[130,80],[140,80]]);
checkRoute("tracking","touches",[[220,266],[200,266]]);
checkRoute("pose","tracking",[[360,68],[370,68],[370,266],[360,266]]);
for(const columns of [8,7,4,3,2,1]){
  const width=columns*140+64;
  const boxes=new Map();
  let top=66;
  for(const phase of [1,2]){
    const cards=stages.filter(stage=>stage[3]===phase);
    const count=Math.min(columns,cards.length);
    const pitch=(width-64)/count;
    for(const [index,[id]] of cards.entries()){
      const column=phase===2?count-1-index%count:index%count;
      const l=32+column*pitch,t=top+Math.floor(index/count)*150;
      boxes.set(id,{l,r:l+pitch-19,t,b:t+78});
    }
    top+=Math.ceil(cards.length/count)*150+55;
  }
  for(const [index,points] of route(boxes,links,width).entries()){
    const [from,to]=links[index].map(id=>boxes.get(id));
    const first=points[0],last=points.at(-1);
    const onBorder=([x,y],box)=>x>=box.l&&x<=box.r&&y>=box.t&&y<=box.b&&
      (x===box.l||x===box.r||y===box.t||y===box.b);
    assert.ok(onBorder(first,from)&&onBorder(last,to));
    for(let i=1;i<points.length;i++){
      const [x,y]=points[i-1],[nx,ny]=points[i];
      assert.ok(x===nx||y===ny,"orthogonal arrows");
      assert.ok(x>0&&x<width&&nx>0&&nx<width,"arrows stay in the diagram");
      for(const box of boxes.values()){
        const crossing=x===nx?x>box.l&&x<box.r&&Math.min(y,ny)<box.b&&Math.max(y,ny)>box.t:
          y>box.t&&y<box.b&&Math.min(x,nx)<box.r&&Math.max(x,nx)>box.l;
        assert.ok(!crossing,"arrows must stay outside cards");
      }
    }
  }
}

console.log("PASS: arrows stay outside cards at wide, tablet and narrow layouts");
const visible=vm.runInNewContext(definition("const graphEdgeVisible=","const drawGraphEdges=")+"graphEdgeVisible;");
assert.equal(visible({local:true,state:"available",focus:"players",focused:false,input:true}),true,
  "The live VBallNet input remains visible while another stage is focused");
assert.equal(visible({local:true,state:"disabled",focus:"players",focused:false,input:true}),false);
assert.equal(visible({local:true,state:"available"}),true);
assert.equal(visible({local:false,state:"available"}),false);
assert.equal(visible({local:false,state:"optional",focus:"pose",focused:true}),true);
assert.equal(visible({local:true,state:"available",focus:"ball",focused:false,active:true}),false);
assert.equal(visible({local:true,state:"disabled",focused:true,active:true}),false);

const draw=vm.runInNewContext(
  definition("const pipelineStages=","const pipelineNames=")+
  definition("const graphRoutes=","const graphSvg=")+"drawGraphEdges;");
const mockNodes=stages.map(([id,,,phase],index)=>{
  const col=phase===1?index:11-index;
  const left=20+col*120,top=phase===1?20:220;
  return {dataset:{stageId:id},getBoundingClientRect:()=>({left,right:left+100,top,bottom:top+76})};
});
const mockEdges=links.map(([edgeSource,edgeTarget,condition])=>({
  dataset:{edgeSource,edgeTarget,state:condition?"optional":"available"},
  classList:{contains:()=>true,toggle:()=>{}},
  setAttribute(name,value){this[name]=value;},
  removeAttribute(name){delete this[name];}
}));
const mockSvg={querySelectorAll:()=>mockEdges,setAttribute:()=>{}};
const mockTarget={dataset:{},clientWidth:980,clientHeight:340,
  getBoundingClientRect:()=>({left:0,top:0}),
  querySelector:()=>mockSvg,querySelectorAll:()=>mockNodes};
const phaseOf=id=>stages.find(stage=>stage[0]===id)[3];
const handoffs=()=>mockEdges.filter(edge=>edge.d&&phaseOf(edge.dataset.edgeSource)!==phaseOf(edge.dataset.edgeTarget));
draw(mockTarget);
assert.equal(handoffs().length,1);
assert.equal(handoffs()[0].dataset.edgeSource,"pose");
mockEdges.find(edge=>edge.dataset.edgeSource==="pose"&&edge.dataset.edgeTarget==="tracking").dataset.state="disabled";
draw(mockTarget);
assert.equal(handoffs().length,1);
assert.equal(handoffs()[0].dataset.edgeSource,"boxes");
for(const edge of mockEdges)edge.classList={contains:()=>false,toggle(name,value){this[name]=value;}};
for(const [from,to] of [["ball_stabilizer","touch_detector"],["touch_detector","touches"]]){
  mockTarget.dataset.focusStage=to;
  draw(mockTarget);
  const edge=mockEdges.find(edge=>edge.dataset.edgeSource===from&&edge.dataset.edgeTarget===to);
  assert.ok(edge.d,from+" -> "+to+" must be visible on focus");
  assert.equal(edge.classList["is-highlighted"],true);
  assert.equal(edge.classList["is-outgoing"],false);
}
