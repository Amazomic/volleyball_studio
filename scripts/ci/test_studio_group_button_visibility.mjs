import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const setMode=mode=>{");
const end=source.indexOf('  singlesMode.addEventListener',start);
const button=()=>({attributes:{},setAttribute(name,value){this.attributes[name]=value;}});
const area=()=>({
  children:[],
  append(child){
    if(child.parent)child.parent.children=child.parent.children.filter(item=>item!==child);
    this.children.push(child);child.parent=this;
  },
  prepend(child){
    if(child.parent)child.parent.children=child.parent.children.filter(item=>item!==child);
    this.children.unshift(child);child.parent=this;
  }
});
const viewLabel={},modeTabs=button(),viewBar=area(),modeInfo=area(),modeSwitch=area();
const autoAssignControls=area();
modeInfo.append(viewLabel);autoAssignControls.append(modeInfo);
const context={state:{viewPlayer:""},tracksPanel:{dataset:{}},singlesMode:button(),
  groupsMode:button(),playerMode:button(),playerGallery:{},groupButton:{},
  groupingControls:{},autoAssignControls,viewBar,modeInfo,modeTabs,modeSwitch,viewLabel};
runInNewContext(source.slice(start,end)+"\nthis.setMode=setMode;",context);
context.setMode("singles");assert.equal(context.groupingControls.hidden,true);
assert.equal(viewLabel.parent,modeInfo);assert.equal(context.modeSwitch.hidden,true);
assert.equal(context.singlesMode.attributes["aria-pressed"],"true");
context.setMode("groups");assert.equal(context.groupingControls.hidden,false);assert.equal(context.modeSwitch.hidden,false);
assert.equal(context.groupsMode.attributes["aria-pressed"],"true");
assert.equal(context.singlesMode.attributes["aria-pressed"],"false");
context.state.viewPlayer="player-a";
context.setMode("player");assert.equal(context.groupingControls.hidden,true);
assert.equal(viewLabel.parent,modeInfo);assert.equal(modeInfo.parent,modeSwitch);
assert.equal(context.modeSwitch.hidden,false);assert.equal(autoAssignControls.hidden,true);
assert.equal(context.playerGallery.hidden,false);
context.setMode("singles");assert.equal(context.playerGallery.hidden,true);assert.equal(context.modeSwitch.hidden,false);
assert.equal(autoAssignControls.hidden,true);
assert.equal(context.singlesMode.textContent,"Привязанные ID");
context.state.viewPlayer="";
context.setMode("singles");assert.equal(autoAssignControls.hidden,false);
assert.equal(modeInfo.parent,autoAssignControls);assert.equal(viewLabel.parent,modeInfo);
assert.equal(context.modeSwitch.hidden,true);

console.log("Grouping controls and both player tabs switch without hiding navigation: OK");
