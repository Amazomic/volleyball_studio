import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_video_library.js",import.meta.url),"utf8");
const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const project="1234abcdef01",job="montage_"+"1".repeat(32);
const url="/api/montage/native/media?project_id="+project+"&job_id="+job;
class Element{
  constructor(){this.children=[];this.handlers=new Map();this.value="";this.paused=true;this.plays=0;this.open=false;this.classList={add(){},remove(){}};}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  setAttribute(){}
  removeAttribute(name){if(name==="src")this.src="";}
  focus(){}
  load(){}
  addEventListener(name,handler){this.handlers.set(name,handler);}
  fire(name,event={}){return this.handlers.get(name)?.(event);}
  showModal(){this.open=true;}
  close(){this.open=false;this.fire("close");}
  pause(){this.paused=true;}
  async play(){assert.equal(dialog.open,true,"Open dialog before requesting playback");this.plays++;this.paused=false;}
}
const nodes=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element()]));
const get=id=>nodes.get(id),dialog=get("studio-video-library-player"),media=get("studio-video-player-media");
get("studio-project").value=project;
const windowEvents=new Map();
runInNewContext(source,{
  document:{getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element(),addEventListener(){}},
  window:{addEventListener:(name,fn)=>windowEvents.set(name,fn),VolleyballStudioActivity:{fetch:async()=>({ok:true,json:async()=>({videos:[{project_id:project,id:job,url,label:"Test"}]})})}}
});
get("studio-video-library-toggle").fire("click");
await new Promise(resolve=>setImmediate(resolve));
const entry=get("studio-video-library-list").children[0].children[1];
entry.children[1].children[0].fire("click");
assert.equal(media.src,url);
assert.equal(media.paused,false,"One library Play click must start playback");
const key=(options={})=>{
  const event={code:"Space",key:" ",preventDefault(){this.prevented=true;},stopPropagation(){this.stopped=true;},...options};
  dialog.fire("keydown",event);
  return event;
};
let e=key();
assert.equal(media.paused,true,"Space pauses");
assert.ok(e.prevented&&e.stopped,"Prevent default native action and bubbling");
e=key({repeat:true});
assert.equal(media.paused,true,"Held Space must not toggle again");
assert.ok(e.prevented);
key();
assert.equal(media.paused,false,"Space resumes");
key({ctrlKey:true});
assert.equal(media.paused,false,"Modified shortcuts are ignored");
get("studio-video-player-close").fire("click");
assert.equal(media.paused,true);
assert.equal(media.src,"");
e=key();
assert.ok(!e.prevented,"Closed player must not intercept Space");
windowEvents.get("volley:montage-complete")({detail:{project_id:project,id:job,media_url:url}});
assert.equal(dialog.open,true);
assert.equal(media.paused,true,"Background montage completion must not autoplay");
key();
assert.equal(media.paused,false,"Space starts playback in completion dialog");
console.log("Library playback: single-click start, Space toggle, repeat, close and completion: PASS");
