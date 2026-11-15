import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_ui.js",import.meta.url),"utf8");
const animationSource=source.slice(0,source.indexOf("// Minimal browser adapters"));
assert.ok(animationSource.includes("iconFrames"),"favicon cache must exist");
let currentTime=0,encoded=0,updates=0,painted=0;
let nextId=0;
const callbacks=[];
const favicon={_href:"/studio/volleyball-photo.png",
  get href(){return this._href;},
  set href(value){this._href=value;updates++;}};
const ball={src:"/studio/volleyball-photo.png",currentSrc:"/studio/volleyball-photo.png",
  style:{transform:""},closest:()=>null};
const ctx={clearRect(){},save(){},translate(){},rotate(){},drawImage(){painted++;},restore(){},
  imageSmoothingQuality:""};
const canvas={width:0,height:0,getContext:()=>ctx,
  toDataURL:()=>("data:image/png;base64,frame"+(++encoded))};
class FakeImage{
  set src(value){this.currentSrc=value;this.onload();}
}
const doc={hidden:false,body:{dataset:{}},
  querySelector:selector=>selector===".brand .volleyball-ball"?ball:
    selector==='link[rel="icon"]'?favicon:null,
  createElement:tag=>{assert.equal(tag,"canvas");return canvas;},
  addEventListener(){}};
const timers=new Map();
let timerId=0;
const testWindow={matchMedia:()=>({matches:false,addEventListener(){}}),
  addEventListener(){},fetch:()=>Promise.resolve()};
const context={
  document:doc,window:testWindow,Image:FakeImage,performance:{now:()=>currentTime},
  requestAnimationFrame:cb=>{callbacks.push(cb);return ++nextId;},
  cancelAnimationFrame(){},
  setInterval:cb=>{timers.set(++timerId,cb);return timerId;},
  clearInterval:id=>timers.delete(id),
  setTimeout:()=>-1,clearTimeout(){}
};
runInNewContext(animationSource,context);
assert.equal(canvas.width,48);
assert.equal(canvas.height,48);
assert.ok(favicon.href.startsWith("data:image/png;"),"initial icon is painted");
testWindow.VolleyballStudioActivity.setTaskActive("gpu",true);
for(let i=0;i<1800;i++){
  currentTime+=1000/60;
  const callback=callbacks.shift();
  assert.ok(callback,"running animation should keep producing frames");
  callback(currentTime);
}
assert.ok(ball.style.transform.startsWith("rotate("),"sidebar ball remains animated");
assert.ok(updates>150,"tab icon still updates while running");
assert.ok(encoded<=180,"only 180 unique PNGs should be encoded, got "+encoded);
assert.ok(updates>encoded,"cached URLs should be reused");
assert.equal(painted,encoded,"PNG renders only happen on a cache miss");
testWindow.VolleyballStudioActivity.setTaskActive("gpu",false);
console.log("PASS: favicon rotation reuses encoded frames, sidebar motion unchanged");
