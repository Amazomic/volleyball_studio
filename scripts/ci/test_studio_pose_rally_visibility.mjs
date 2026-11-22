import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio.js",import.meta.url),"utf8");
const start=source.indexOf("  const poseVisibleInRally=frame=>");
const end=source.indexOf("  const drawOverlay=()=>",start);
assert.ok(start>0&&end>start,"Processed preview must define a separate pose visibility guard");
const ctx={};
runInNewContext(source.slice(start,end)+"\nthis.visible=poseVisibleInRally;",ctx);
const pose=[{keypoints:Array.from({length:17},()=>[0.5,0.5,0.9])}];
assert.equal(ctx.visible({in_rally:true,pose_points:pose}),true,"Show poses within a rally");
assert.equal(ctx.visible({in_rally:false,pose_points:pose}),false,"Hide poses between rallies");
assert.equal(ctx.visible({pose_points:pose}),false,"Fail closed when scope was not verified");
assert.equal(ctx.visible({in_rally:true,pose_points:null}),false);
assert.equal(ctx.visible(null),false);
assert.match(source,/if\(overlay\.flags\.pose && poseVisibleInRally\(current\)\)/);
assert.match(source,/if\(!overlay\.stabilizerView&&overlay\.rawPoseVisible\.checked\)/,
  "Raw GPU pose observations must remain independent of rally scope outside trajectory-only comparison");
console.log("Studio processed RTMPose respects rally-only visibility; raw GPU diagnostic stays available.");
