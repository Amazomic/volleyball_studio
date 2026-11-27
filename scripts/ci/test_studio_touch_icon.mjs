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
vm.runInContext(js.slice(js.indexOf("const pipelineGlyphs="),js.indexOf("const pipelineId="))+"this.icon=pipelineIcon;",context);
const touch=context.icon("touch_detector");
assert.equal(touch.attrs.viewBox,"0 0 48 24");
assert.equal(touch.attrs.stroke,"currentColor");
assert.equal(touch.attrs["aria-hidden"],"true");
const palm=touch.children.find(part=>part.attrs["data-part"]==="palm");
assert.equal(palm.tag,"path");
assert.equal((palm.attrs.d.match(/a1\.5 1\.5 0 0 0-3 0/g)||[]).length,4);
const ball=touch.children.find(part=>part.tag==="circle");
assert.equal(ball.attrs.r,"6");
assert.ok(touch.children.some(part=>part.attrs["data-part"]==="ball-seams"));
const trail=touch.children.find(part=>part.attrs["data-part"]==="trajectory");
assert.match(trail.attrs.d,/C/);
assert.ok(trail.attrs.opacity);
assert.equal(context.icon("touches").attrs.viewBox,"0 0 24 24");
assert.ok(!context.icon("touches").children.some(part=>part.tag==="circle"));
assert.match(css,/\.analysis-flow-node\[data-stage-id="touch_detector"\] \.analysis-flow-node-icon\{width:32px\}/);
console.log("Touch Detector open palm and ball trajectory icon: PASS");
