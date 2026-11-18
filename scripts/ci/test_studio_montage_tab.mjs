import {readFileSync,existsSync} from "node:fs";
import {join} from "node:path";
import {fileURLToPath} from "node:url";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const root=join(fileURLToPath(new URL("../..",import.meta.url)));
const html=readFileSync(join(root,"rust/volley-app/static/studio.html"),"utf8");
assert.equal(existsSync(join(root,"rust/volley-app/static/studio_highlights.js")),false);
assert.equal(existsSync(join(root,"scripts/ci/test_native_highlights_ui.mjs")),false);
const script=readFileSync(join(root,"rust/volley-app/static/studio_tabs.js"),"utf8");

function element(id){
  const match=new RegExp('<([a-z]+)\\b[^>]*\\bid="'+id+'"[^>]*>',"i").exec(html);
  assert.ok(match,"Missing element "+id);
  const tag=match[1].toLowerCase();
  const start=match.index+match[0].length;
  let depth=1;
  const token=new RegExp('<(/?)'+tag+'\\b[^>]*>',"gi");
  token.lastIndex=start;
  let m;
  while((m=token.exec(html))){
    depth+=m[1]?-1:1;
    if(depth===0)return html.slice(start,m.index);
  }
  assert.fail("Unclosed "+id);
}
const preview=element("studio-preview-view");
const montage=element("studio-montage-view");
assert.ok(preview.includes('id="studio-video"'),"Preview retains playback");
assert.ok(preview.includes('id="studio-rallies"'),"Preview retains rally selection");
for(const id of ["studio-montage","studio-montage-presets","studio-montage-mode",
    "studio-montage-action","studio-montage-duration","studio-montage-start",
    "studio-montage-player","studio-montage-audio-fit","studio-montage-camera"]){
  assert.ok(!preview.includes('id="'+id+'"'),id+" must not be in preview");
  assert.ok(montage.includes('id="'+id+'"'),id+" must belong to montage");
  assert.equal(html.split('id="'+id+'"').length,2,"No duplicate "+id);
}
assert.ok(!montage.includes('id="studio-highlights-panel"'),
    "Highlights must not be a second independent montage block");
assert.ok(!montage.includes('data-profile="report"'));
assert.ok(montage.includes('id="studio-montage-statistics"'));
assert.ok(montage.includes('data-profile="player"'));
assert.ok(montage.includes('data-profile="attacks"'));
assert.ok(montage.includes('data-profile="favorites"'));
assert.match(html, /id="studio-montage-view" class="hidden tab-workspace"/);
assert.match(html, /id="studio-nav-montage"[^>]+data-studio-tab="montage"/);

const start=script.indexOf('  const views={');
const end=script.indexOf('  const projectId=()=>',start);
assert.ok(start>=0&&end>start);
const nodes=Object.fromEntries(["preview","montage","quick","analysis","players","report"].map(k=>{
  const state={hidden:k!=="preview"};
  return [k,{classList:{toggle:(name,on)=>{if(name==="hidden")state.hidden=on;}},
    get hidden(){return state.hidden;}}];
}));
const ui={...nodes};
const events=[],state={tab:"preview"};
const runtime={navigate:requested=>requested,beginLoad:()=>{}};
const window={dispatchEvent:event=>events.push(event),addEventListener(){}};
const document={querySelectorAll:()=>[],addEventListener(){}};
const CustomEvent=class{constructor(type,options){this.type=type;this.detail=options.detail;}};
const loadVideos=()=>Promise.resolve(),loadPlayers=()=>Promise.resolve(),failure=()=>{};
const context={ui,runtime,state,window,document,CustomEvent,loadVideos,loadPlayers,failure};
const fn=script.slice(start,end)+"\nthis.switchTab=switchTab;";
runInNewContext(fn,context);
context.switchTab("montage");
assert.equal(nodes.preview.hidden,true,"Preview must be hidden for montage");
assert.equal(nodes.montage.hidden,false,"Montage must be visible");
for(const key of ["analysis","players","quick","report"])assert.equal(nodes[key].hidden,true);
context.switchTab("preview");
assert.equal(nodes.preview.hidden,false);
assert.equal(nodes.montage.hidden,true);
assert.deepEqual(events.map(e=>e.detail.tab),["montage","preview"]);
assert.ok(!script.includes('scrollIntoView({behavior:"smooth"})'),"Montage must not scroll the preview");
console.log("Montage tab ownership and navigation: OK");
