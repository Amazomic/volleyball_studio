import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const js=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const css=readFileSync(new URL("../../rust/volley-app/static/studio.css",import.meta.url),"utf8");
assert.ok(!html.includes('id="studio-rally-video-markers"'),"No separate marker rail");
assert.ok(!html.includes('class="rally-scroll-region"'),"Keep the original scroll container");
assert.match(css,/\.rally-grid::-webkit-scrollbar-track\{[^}]*background-image:var\(--rally-video-track,none\)/);
const start=js.indexOf("  const renderRallyVideoMarkers=");
const end=js.indexOf("  const refreshRallyVideoMarkers=",start);
assert.ok(start>=0&&end>start);
const render=runInNewContext(js.slice(start,end)+"\nrenderRallyVideoMarkers");
const card=(video,top)=>({getAttribute:()=>video,getBoundingClientRect:()=>({top:top-list.scrollTop})});
let cards=[card("a.mp4",0),card("a.mp4",100),card("b.mp4",250),card("c.mp4",700),card("b.mp4",900)];
const props=new Map();
const list={scrollHeight:1000,clientHeight:250,scrollTop:0,clientTop:0,
  getBoundingClientRect:()=>({top:0}),querySelectorAll:()=>cards,
  style:{setProperty(key,value){props.set(key,value);},removeProperty(key){props.delete(key);}}};
const track=()=>props.get("--rally-video-track");
const positions=()=>Array.from(track().matchAll(/transparent calc\(([\d.]+)% - 1px\)/g),m=>Number(m[1]));
render(list);
assert.deepEqual(positions(),[0,25,70]);
assert.equal((track().match(/linear-gradient/g)||[]).length,3);
list.scrollTop=185;
render(list);
assert.deepEqual(positions(),[0,25,70],"Markers stay fixed as the thumb moves");
cards=[card("b.mp4",0),card("a.mp4",300)];
render(list);
assert.deepEqual(positions(),[0,30]);
cards=[card("a.mp4",0)];
render(list);
assert.equal(track(),undefined);
cards=[card("a.mp4",0),card("b.mp4",100)];list.scrollHeight=250;
render(list);
assert.equal(track(),undefined);
console.log("PASS: native scrollbar track markers, fixed positions, filters and single-video reset.");
