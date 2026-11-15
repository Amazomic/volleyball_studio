import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(
  new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8"
);
// Use the actual in-canvas label formatter, not a hand-written duplicate.
const first=source.indexOf("        const name=owner||String(player.id);");
const last=source.indexOf("        ctx.fillStyle=\"#f1f8ff\";ctx.fillText(name,px+5,py+rh+1,tw-10);",first);
assert.ok(first>0&&last>first,"Video labels must display the linked name or a plain numeric ID");
const snippet=source.slice(first,last+
  '        ctx.fillStyle="#f1f8ff";ctx.fillText(name,px+5,py+rh+1,tw-10);'.length);
const render=(id,owner)=>{
  const drawn=[];
  const ctx={
    measureText:text=>({width:text.length*8}),
    fillRect(){},
    fillText:(text)=>drawn.push(text),
  };
  runInNewContext(snippet,{

    player:{id},owner,ctx,px:10,py:5,rh:80,
  });
  return drawn;
};
assert.deepEqual(render(195,null),["195"]);
assert.deepEqual(render(203,"Тимур"),["Тимур"]);
assert.deepEqual(render(195,"Влад"),["Влад"]);

// A server-side auto-assignment may happen while the Players tab is open.
// Returning to Preview must fetch a new name snapshot without recomputing
// immutable CPU frame windows.
const eventStart=source.indexOf('  window.addEventListener("volley:tab-change",event=>{');
assert.ok(eventStart>=0,"Preview must respond to workspace activation");
const eventEnd=source.indexOf("  });",eventStart)+5;
const callbacks=new Map();
const calls=[];
runInNewContext(source.slice(eventStart,eventEnd),{
  window:{addEventListener:(kind,callback)=>callbacks.set(kind,callback)},
  s:{project:"p",run:"cpu-run"},
  invalidateOverlayLabels:()=>calls.push("invalidate"),
  drawOverlay:()=>calls.push("redraw"),
});
const handler=callbacks.get("volley:tab-change");
assert.equal(typeof handler,"function");
handler({detail:{tab:"players"}});
assert.deepEqual(calls,[],"Don't refetch while the preview is hidden");
handler({detail:{tab:"preview"}});
assert.deepEqual(calls,["invalidate","redraw"]);
console.log("Video overlay shows names or plain numeric IDs and refreshes on return: OK");
