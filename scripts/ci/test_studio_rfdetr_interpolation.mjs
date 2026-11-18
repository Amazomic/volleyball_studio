import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const source=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
assert.match(html, /id="studio-raw-player-interpolated"[^>]*type="checkbox" checked/);
assert.match(source, /variant!=="rfdetr-medium"/,
  "Only verified RF-DETR GPU sources expose the interpolation checkbox");
const from=source.indexOf("        if(overlay.rawPlayerInterpolated?.checked &&");
const until=source.indexOf("        const poseBoxes=",from);
assert.ok(from>=0&&until>from,"Synthetic raw player rendering must exist");
const draw=source.slice(from,until);
const strokes=[];
const ctx={
  save(){},restore(){},setLineDash(dashes){this.dashes=dashes;},
  strokeRect(x,y,w,h){strokes.push({x,y,w,h,color:this.strokeStyle,dashes:this.dashes});},
  fillText(){}
};
const context={
  overlay:{rawPlayerInterpolated:{checked:true},rawDetectors:new Map([["p:f:v","rfdetr-medium"]])},
  rawDetectorKey:(p,r,v)=>[p,r,v].join(":"),project:"p",run:"f",video:"v",
  raw:{players_interpolated:[{box:[0.2,0.3,0.4,0.8]}]},
  ctx,x:10,y:20,width:100,height:200,Number,Array
};
runInNewContext(draw,context);
assert.equal(strokes.length,1);
assert.deepEqual([strokes[0].x,strokes[0].y,strokes[0].w,strokes[0].h],[30,80,20,100]);
assert.equal(strokes[0].color,"#ffbc69","Synthetic proposals use a separate amber color");
assert.equal(strokes[0].dashes.join(","),"6,4","Synthetic proposals use dashed boxes");
context.overlay.rawDetectors.set("p:f:v","legacy");
runInNewContext(draw,context);
assert.equal(strokes.length,1,"Legacy detectors must never display RF-DETR interpolation");
context.overlay.rawDetectors.set("p:f:v","rfdetr-medium");
context.overlay.rawPlayerInterpolated.checked=false;
runInNewContext(draw,context);
assert.equal(strokes.length,1,"Hiding interpolated boxes does not touch model detections");
console.log("RF-DETR raw interpolation toggle and amber geometry: OK");
