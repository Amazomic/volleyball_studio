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
for(const selector of [".analysis-flow-node-model",".analysis-flow-node-stats"]){
  const style=rule(selector);
  assert.doesNotMatch(style,/white-space:nowrap|text-overflow:ellipsis|overflow:hidden/);
  assert.match(style,/overflow-wrap:anywhere/);
}
assert.match(css,/@container \(max-width:330px\)\{\.analysis-flow-phase\[data-phase\] \.analysis-flow-cards\{grid-template-columns:minmax\(0,1fr\)/);
assert.doesNotMatch(css,/\.analysis-flow-diagram\{[^}]*min-width:950px/);
assert.match(flow,/phases\.get\(row\)\.append\(cell\)/);

const start=flow.indexOf("const graphRoutes=");
const end=flow.indexOf("const drawGraphEdges=",start);
assert.ok(start>=0&&end>start,"The graph must route arrows between actual stage positions");
const route=vm.runInNewContext(flow.slice(start,end)+"graphRoutes;");
const definition=(begin,end)=>flow.slice(flow.indexOf(begin),flow.indexOf(end,flow.indexOf(begin)));
const stages=vm.runInNewContext(definition("const pipelineStages=","const pipelineNames=")+"pipelineStages;");
const links=vm.runInNewContext(definition("const pipelineEdges=","const graphLinkState=")+"pipelineEdges;");
for(const columns of [7,3,2,1]){
  const width=columns*140+64;
  const boxes=new Map();
  let top=66;
  for(const phase of [1,2]){
    const cards=stages.filter(stage=>stage[3]===phase);
    const count=Math.min(columns,cards.length);
    const pitch=(width-64)/count;
    for(const [index,[id]] of cards.entries()){
      const l=32+index%count*pitch,t=top+Math.floor(index/count)*150;
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
