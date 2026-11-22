import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const setMode=mode=>{");
const end=source.indexOf('  singlesMode.addEventListener',start);
const button=()=>({attributes:{},setAttribute(name,value){this.attributes[name]=value;}});
const viewLabel={};
const area=()=>({children:[],append(child){
  if(child.parent)child.parent.children=child.parent.children.filter(item=>item!==child);
  this.children.push(child);child.parent=this;
}});
const viewBar=area(),modeInfo=area();modeInfo.append(viewLabel);
const context={tracksPanel:{dataset:{}},singlesMode:button(),groupsMode:button(),playerMode:button(),
  playerGallery:{},groupButton:{},groupingControls:{},autoAssignControls:{},
  filterBar:{},viewBar,modeInfo,viewLabel};
runInNewContext(source.slice(start,end)+"\nthis.setMode=setMode;",context);
context.setMode("singles");assert.equal(context.groupingControls.hidden,true);
assert.equal(viewLabel.parent,modeInfo);assert.equal(context.filterBar.hidden,false);
assert.equal(context.singlesMode.attributes["aria-pressed"],"true");
context.setMode("groups");assert.equal(context.groupingControls.hidden,false);
assert.equal(context.groupsMode.attributes["aria-pressed"],"true");
assert.equal(context.singlesMode.attributes["aria-pressed"],"false");
context.setMode("player");assert.equal(context.groupingControls.hidden,true);
assert.equal(viewLabel.parent,viewBar);assert.equal(context.filterBar.hidden,true);
assert.equal(context.autoAssignControls.hidden,true);
context.setMode("singles");assert.equal(context.autoAssignControls.hidden,false);
assert.equal(viewLabel.parent,modeInfo);assert.equal(context.filterBar.hidden,false);

console.log("Grouping controls are shown only in the grouped IDs view: OK");
