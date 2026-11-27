import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

export function previewWasmWindow(){
  const bytes=readFileSync(new URL("../../../rust/volley-app/static/volley_web.wasm",import.meta.url));
  const engine=new WebAssembly.Instance(new WebAssembly.Module(bytes),{}).exports;
  const window={VolleyballUi:{engine,isReady:()=>true}};
  const bridge=readFileSync(new URL("../../../rust/volley-app/static/studio_chain.js",import.meta.url),"utf8");
  runInNewContext(bridge,{window,TextEncoder,TextDecoder});
  return window;
}
