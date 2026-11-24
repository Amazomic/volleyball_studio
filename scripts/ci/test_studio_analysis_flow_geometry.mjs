import assert from "node:assert/strict";
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
assert.match(css,/@container \(max-width:330px\)\{\.analysis-flow-cards\{grid-template-columns:minmax\(0,1fr\)/);
assert.doesNotMatch(css,/\.analysis-flow-diagram\{[^}]*min-width:950px/);
assert.match(flow,/phases\.get\(row\)\.append\(cell\)/);
console.log("PASS: the analysis graph wraps its cards and metrics within the panel");
