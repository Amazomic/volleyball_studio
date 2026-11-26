import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const start=js.indexOf("  const timeLabel=");
const end=js.indexOf("  const drawTimeline=",start);
assert.ok(start>=0&&end>start);
const video={duration:4000,currentTime:0};
const s={project:"p",video:"v",videoKey:"p:v",seekAt:0};
const timeline={data:null,playhead:{style:{}},middle:{},track:{setAttribute(){}}};
const context={ui:{video},s,timeline};
runInNewContext(js.slice(start,end)+"\nthis.update=updateTimelineProgress;",context);
for(const [position,label] of [[48.719,"00:48.7"],[0,"00:00.0"],[59.999,"00:59.9"],[60,"01:00.0"],[3600.25,"1:00:00.2"]]){
  video.currentTime=position;
  context.update();
  assert.equal(timeline.middle.textContent,label);
  assert.equal(timeline.playhead.style.left,(100*position/4000)+"%");
}
assert.doesNotMatch(js,/timeline\.middle\.textContent="00:00\.000"/);
console.log("Studio preview timeline tenths: OK");
