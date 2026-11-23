import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const start=js.indexOf("  const annotationSourceTime=frame=>");
const end=js.indexOf("  const annotationNotice=(",start);
const positionStart=js.indexOf("  const annotationPosition=frame=>");
const positionEnd=js.indexOf("  const annotationMarkCurrent=()=>",positionStart);
assert.ok(start>=0&&end>start&&positionStart>=0&&positionEnd>positionStart);

const analysisOffset=1572.17;
const duration=7065.669;
const startFrame=2083;
const firstTouchFrame=2091;
const expectedFirstRally=analysisOffset+startFrame/30;
const expectedFirstTouch=analysisOffset+firstTouchFrame/30;
const s={analysisOffset,seekAt:0};
const ui={video:{currentTime:expectedFirstTouch}};
const annotation={data:{total_frames:164799}};
const context={s,ui,annotation,fps:()=>30,timelineDuration:()=>duration};
const helpers=runInNewContext(js.slice(start,end)+js.slice(positionStart,positionEnd)+
  "\n({annotationFrame,annotationClock,annotationPosition,annotationSourceTime,annotationFrameAtPosition})",
  context);
const near=(a,b,label)=>assert.ok(Math.abs(a-b)<0.00001,label+": "+a+" vs "+b);
assert.equal(helpers.annotationFrame(),firstTouchFrame,
  "Current original-video time must map back to the local analysis frame");
near(Number(helpers.annotationPosition(startFrame)),100*expectedFirstRally/duration,
  "First rally must appear at 27:21 on the original-video timeline");
near(Number(helpers.annotationPosition(firstTouchFrame)),100*expectedFirstTouch/duration,
  "First touch must share the original-video coordinate system with its rally");
assert.ok(helpers.annotationPosition(firstTouchFrame)>23,
  "Touch from the 27th minute cannot appear near the first minute");
assert.ok(helpers.annotationPosition(firstTouchFrame)<24);
near(helpers.annotationSourceTime(firstTouchFrame),expectedFirstTouch,
  "Touch navigation must seek the source video time");
assert.equal(helpers.annotationFrameAtPosition(expectedFirstTouch/duration,164799),
  firstTouchFrame,"Dragging a marker must invert the timeline mapping");
assert.equal(helpers.annotationFrameAtPosition(0,164799),0,
  "Dragging before analysis start must clamp to its first frame");
assert.match(helpers.annotationClock(firstTouchFrame),/1641,87/,
  "Displayed timestamp must be source-video time, not relative analysis time");
ui.video.currentTime=0;
assert.equal(helpers.annotationFrame(),0,"Outside the analysis range cannot yield a negative frame");
assert.match(js,/seekTimeline\(annotationSourceTime\(item\.frame\)\)/);
assert.match(js,/seekTimeline\(annotationSourceTime\(ground\.frame\)\)/);
assert.match(js,/seekTimeline\(annotationSourceTime\(item\.start\)\)/);
console.log("Debug timeline source offset: markers, navigation, dragging and timestamps stay in the 27th minute");