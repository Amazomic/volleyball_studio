import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const js=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
for(const side of ["before","after"]){
  const id="studio-gpu-rally-padding-"+side+"-ms";
  assert.match(html,new RegExp('id="'+id+'"[^>]*min="0"[^>]*value="500"[^>]*data-default="500"'));
  assert.ok(js.includes('padding_'+side+'_ms:inputValue("'+id+'")'));
  assert.ok(js.includes('"#'+id+'"'));
}
