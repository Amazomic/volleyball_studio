import assert from "node:assert/strict";
import {readFileSync,readdirSync} from "node:fs";
const text=p=>readFileSync(new URL("../../"+p,import.meta.url),"utf8");
const preparation=text("rust/volley-app/src/preparation.rs");
const preparationFiles=[preparation,...readdirSync(new URL("../../rust/volley-app/src/preparation/",import.meta.url)).filter(p=>p.endsWith(".rs")).map(p=>text("rust/volley-app/src/preparation/"+p))];
for(const file of preparationFiles)assert.ok(!/\bquick_reel\b/.test(file),"Preparation must not call the Quick Reel orchestrator");
const source=text("rust/volley-app/src/quick_reel_source.rs");
assert.ok(!/\b(?:quick_reel|preparation|QuickReelManager|PreparationManager)\b/.test(source.replaceAll("volley_media::preparation::","")),"Source owner must not call orchestrators or managers");
const reel=text("rust/volley-app/src/quick_reel.rs");
for(const declaration of ["struct Source", "fn safe_source(","fn verify_source(","fn source_for(","fn prepared_plan_for_quick_reel("]){assert.ok(!reel.includes(declaration),"Duplicate source implementation: "+declaration);assert.ok(source.includes(declaration),"Missing moved implementation: "+declaration);}
assert.ok(preparation.includes("crate::quick_reel_source::prepared_plan_for_quick_reel"));
const ledger=JSON.parse(text("docs/RUST_CANONICAL_OWNERS.json"));
assert.ok(ledger.domains.find(d=>d.id==="quick_reel").adapters.includes("rust/volley-app/src/quick_reel_source.rs"));
console.log("Preparation and Quick Reel share source identity; source owner does not call either runner");
