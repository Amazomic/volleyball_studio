import assert from "node:assert/strict";
import {readFileSync,readdirSync} from "node:fs";
const text=path=>readFileSync(new URL("../../"+path,import.meta.url),"utf8");
const renderer=text("rust/volley-app/src/montage.rs");
assert.ok(!/\bnative_rally_preview\b/.test(renderer),"Renderer must not depend on HTTP rally preview");
const material=text("rust/volley-app/src/montage_material.rs");
const evidence=text("rust/volley-app/src/native_rally_evidence.rs")+text("rust/volley-app/src/native_rally_evidence/camera.rs");
assert.ok(!/\bnative_rally_preview\b|\bcrate::montage\b|(?<![\w:])montage::/.test(material),"Shared material cannot call preview or renderer");
assert.ok(!/\bnative_rally_preview\b|\bmontage_material\b|\bcrate::montage\b|(?<![\w:])montage::/.test(evidence),"Evidence must not call material, renderer or HTTP preview");
assert.ok(!/\b(?:State|Query|HeaderMap|ApiResult)\b/.test(evidence),"Evidence must not own HTTP handlers");
const preview=text("rust/volley-app/src/native_rally_preview.rs");
assert.ok(!/fn (?:inspect_project|generate|publish_material|material_bindings|validate_material_binding)\b/.test(preview),"HTTP adapter must not duplicate source or material services");
const owner=JSON.parse(text("docs/RUST_CANONICAL_OWNERS.json"));
for(const file of ["rust/volley-app/src/montage_material.rs","rust/volley-app/src/native_rally_evidence.rs"]){
  assert.ok(owner.domains.some(domain=>domain.adapters?.includes(file)),"Missing owner registration: "+file);
}
const sourceRoot=new URL("../../rust/volley-app/src/",import.meta.url);
function inspect(directory){
  for(const entry of readdirSync(directory,{withFileTypes:true})){
    const path=new URL(entry.name+(entry.isDirectory()?"/":""),directory);
    if(entry.isDirectory()){inspect(path);continue;}
    if(!entry.name.endsWith(".rs")||["main.rs","native_rally_preview.rs","tests.rs","api_http_tests.rs"].includes(entry.name))continue;
    const runtime=readFileSync(path,"utf8").split(/#\[cfg\(test\)\]\s*mod \w+\s*\{/)[0];
    assert.ok(!/\bnative_rally_preview\s*(?:::|as\b|[,}])/.test(runtime),"Runtime source must not consume HTTP preview: "+entry.name);
  }
}
inspect(sourceRoot);
console.log("Rally material dependency boundaries: renderer -> material -> evidence; HTTP preview remains an adapter");
