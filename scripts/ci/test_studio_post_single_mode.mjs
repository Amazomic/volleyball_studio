import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const html=readFileSync(new URL("../../rust/volley-app/static/studio.html",import.meta.url),"utf8");
const tabs=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const start=html.indexOf('<section class="analysis-profile-card" aria-labelledby="studio-post-heading">');
assert.ok(start>=0,"Postprocessing stage is missing");
const end=html.indexOf("</section>",start);
assert.ok(end>start,"Postprocessing stage is not closed");
const panel=html.slice(start,end);
assert.doesNotMatch(panel,/studio-post-touch-algorithm|<select[^>]*id="studio-post-touch-algorithm"/,
  "Obsolete algorithm selection must not appear");
const details=panel.indexOf('<details class="analysis-profile-advanced">');
const close=panel.indexOf("</details>",details);
assert.ok(details>0 && close>details,"Exactly one collapsible parameter section is required");
assert.equal((panel.match(/<details\b/g)||[]).length,1);
assert.doesNotMatch(panel.slice(details,close),/<details[^>]*\bopen(?:\s|=|>)/,
  "Postprocessing parameters must be collapsed by default");
assert.match(panel.slice(details,close),/<summary>Параметры постобработки<\/summary>/);
for(const id of [
  "studio-post-touch-threshold",
  "studio-post-player-score",
  "studio-post-auto-assign-threshold",
  "studio-post-max-gap",
  "studio-post-lost-low-gap",
  "studio-post-lost-low-iou",
  "studio-post-rally-min",
  "studio-analysis-reset"
]){
  const token='id="'+id+'"';
  assert.ok(panel.indexOf(token)>details&&panel.indexOf(token)<close,
    "Parameter "+id+" must be accessible inside the single collapsed section");
}
assert.doesNotMatch(panel,/id="studio-post-require-pose"/,
  "Pose confirmation must be mandatory, not configurable");
assert.doesNotMatch(panel.slice(0,details),/id="studio-post-(?:touch|player|require|rally)[^"]*"/,
  "No individual postprocessing parameters may remain visible when collapsed");

// Evaluate the actual production profile factory without a synthetic algorithm selector.
const a=tabs.indexOf("  const postProfile=()=>({");
const b=tabs.indexOf("  const allProfileFields=()=>",a);
assert.ok(a>=0&&b>a,"Postprocessing request factory is missing");
const env={
  $:id=>({checked:true,value:"1",id}),
  inputValue:id=>({"studio-post-auto-assign-threshold":75,
    "studio-post-lost-low-gap":30,"studio-post-lost-low-iou":0.30}[id]??0.48)
};
runInNewContext(tabs.slice(a,b)+"\nthis.profile=postProfile;",env);
const profile=env.profile();
assert.equal(profile.touches.algorithm,"ball_trajectory");
assert.doesNotMatch(panel,/id="studio-post-tcn-experimental"/,
  "Alternate touch detectors must be absent from the active UI");
env.$=id=>({checked:true,value:"1",id});
assert.equal(env.profile().touches.algorithm,"ball_trajectory",
  "Unrelated toggles cannot replace Touch Detector");
assert.equal(profile.touches.threshold,0.48);
assert.equal(profile.auto_assign_threshold_percent,75);
assert.equal(profile.touches.require_pose,false);
assert.equal(profile.interpolate_players,true);
assert.equal(profile.tracking.lost_low_confidence_max_gap_frames,30);
assert.equal(profile.tracking.lost_low_confidence_min_iou,0.30);
assert.doesNotMatch(tabs,/\$\("studio-post-touch-algorithm"\)/,
  "No hidden DOM dependency on the removed algorithm selector");
console.log("Postprocessing: one ball_trajectory detector, parameters only in closed details: OK");
