import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {join} from "node:path";
import vm from "node:vm";

const root=fileURLToPath(new URL("../..",import.meta.url));
const js=readFileSync(join(root,"rust/volley-app/static/studio_fragments.js"),"utf8");
const chainJs=readFileSync(join(root,"rust/volley-app/static/studio_chain.js"),"utf8");

const pending=[];
const direct=[];
const window={
  MediaSource:class {static isTypeSupported(){return true;}},
  VolleyballPlayback:{
    primary:()=>new Promise(resolve=>pending.push(resolve)),
    direct:(video,url)=>{direct.push(url);video.src=url;}
  },
  addEventListener(){}
};
const document={
  documentElement:{},
  addEventListener(){},
  querySelectorAll(){return [];}
};
const globals={window,document,console,URL,URLSearchParams,MediaSource:window.MediaSource,
  MutationObserver:class{observe(){}},location:{href:"http://localhost/studio",origin:"http://localhost"}};
vm.runInNewContext(js,globals);
const video={
  src:"",paused:true,currentTime:0,
  pause(){this.paused=true;},
  load(){},
  removeAttribute(key){if(key==="src")this.src="";},
  getAttribute(key){return key==="src"?(this.src||null):null;}
};
const playback=window.VolleyballBrowserVideo;
assert.equal(typeof playback?.cancel,"function");
const old=playback.load(video,"p1","first.mp4");
assert.equal(pending.length,1);
playback.cancel(video);
pending.shift()("/api/video-primary-preview?project_id=p1&video=first.mp4");
await old;
assert.equal(video.src,"");
assert.equal(direct.length,0);

const abandoned=playback.load(video,"p1","first.mp4");
const current=playback.load(video,"p2","second.mp4");
assert.equal(pending.length,2);
pending.pop()("/api/video-primary-preview?project_id=p2&video=second.mp4");
await current;
pending.shift()("/api/video-primary-preview?project_id=p1&video=first.mp4");
await abandoned;
assert.equal(video.src,"/api/video-primary-preview?project_id=p2&video=second.mp4");
assert.equal(direct.length,1);

const chainContext={module:{exports:{}}};
vm.runInNewContext(chainJs,chainContext);
const latest=chainContext.module.exports.latestRun;
assert.equal(typeof latest,"function");
const rows=[
  {rally_run_id:"older",created_at:1700},
  {rally_run_id:"newer",created_at:1800}
];
assert.equal(latest(rows,"older"),"older");
assert.equal(latest(rows,"deleted"),"newer");
assert.equal(latest(rows), "newer");
assert.equal(latest([]),"");
console.log("PASS: old media requests cannot resurrect cleared sources; saved CPU choice or latest available run is restored");
