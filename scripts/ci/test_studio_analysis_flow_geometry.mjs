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
const px=(style,key)=>{
  const match=style.match(new RegExp("(?:^|;)\\s*"+key+"\\s*:\\s*(-?\\d+)px(?:;|$)"));
  assert.ok(match,"Expected px value for "+key);
  return Number(match[1]);
};
const phase=rule(".analysis-flow-phase");
const cell=rule(".analysis-flow-cell");
const label=rule(".analysis-flow-phase-label");
const grid=rule(".analysis-flow-grid");
const margins=phase.match(/margin:\s*(-?\d+)px\s+(-?\d+)px\s+(-?\d+)px/);
assert.ok(margins,"Phase must have explicit symmetric inset and vertical clearance");
const [top,right,bottom]=margins.slice(1).map(Number);
assert.ok(top<0&&right<0&&bottom<0);
const labelTop=px(label,"top");
const labelHeight=px(label,"line-height");
const cardTop=px(cell,"padding-top");
assert.ok(labelTop>=10,"Phase heading needs room above it");
assert.ok(cardTop-(top+labelTop+labelHeight)>=10,"Phase heading must not touch stage buttons");
assert.ok(px(grid,"row-gap")-(-top-bottom)>=20,"Phase panels need a visible separation");
assert.match(flow,/panel\.style\.gridColumn="1 \/ -1"/);
assert.match(flow,/\["gpu",.*?"1"\]/);
assert.match(flow,/\["cpu",.*?"2"\]/);
console.log("PASS: GPU and CPU phase frames have balanced borders and isolated headings");