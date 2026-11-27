import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
const document={createElementNS(namespace,tag){
  assert.equal(namespace,"http://www.w3.org/2000/svg");
  return {tag,attrs:{},children:[],setAttribute(key,value){this.attrs[key]=String(value);},append(child){this.children.push(child);}};
}};
const context=vm.createContext({document});
const source=js.slice(js.indexOf("const pipelineGlyphs="),js.indexOf("const pipelineId="));
vm.runInContext(source+"this.icon=pipelineIcon;",context);
const ball=context.icon("ball");
assert.equal(ball.attrs.viewBox,"0 0 48 24");
assert.equal(ball.attrs.stroke,"currentColor");
assert.equal(ball.attrs["aria-hidden"],"true");
const trails=ball.children.filter(part=>part.tag==="path"&&part.attrs.opacity);
assert.equal(trails.length,3);
assert.ok(trails.every(part=>part.attrs.d.includes("C")));
assert.ok(Number(trails[0].attrs.opacity)>Number(trails[1].attrs.opacity));
assert.ok(Number(trails[1].attrs.opacity)>Number(trails[2].attrs.opacity));
const body=ball.children.find(part=>part.tag==="circle");
assert.equal(body.attrs.cx,"36");
assert.equal(body.attrs.r,"10");
assert.equal(context.icon("ball_stabilizer").attrs.viewBox,"0 0 24 24");
assert.equal(context.icon("video").attrs.viewBox,"0 0 24 24");
assert.match(css,/\.analysis-flow-node\[data-stage-id="ball"\] \.analysis-flow-node-icon\{width:32px\}/);
console.log("VballNetV4c trajectory icon: PASS");
