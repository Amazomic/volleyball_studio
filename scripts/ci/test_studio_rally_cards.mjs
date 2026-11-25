import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {join} from "node:path";
import {fileURLToPath} from "node:url";
import {runInNewContext} from "node:vm";

const root=join(fileURLToPath(new URL("../..",import.meta.url)));
const js=readFileSync(join(root,"rust/volley-app/static/studio.js"),"utf8");
const css=readFileSync(join(root,"rust/volley-app/static/studio.css"),"utf8");
const segment=(from,to)=>{
  const a=js.indexOf(from),b=js.indexOf(to,a);
  assert.ok(a>=0&&b>a,"Missing preview segment "+from);
  return js.slice(a,b);
};
const helpers=segment("  const length=item=>","  const valid=id=>");
const estimates=segment("  const estimatedTypes=item=>","  const view=(item,requestedTime=null)=>");
const renderer=segment("  const cards=()=>{","  const uploadAudio=async()=>{");
const element=(tag,klass,text)=>({
  tag,klass,textContent:text===undefined?"":String(text),children:[],listeners:{},
  append(...parts){this.children.push(...parts);},
  replaceChildren(...parts){this.children=parts;},
  addEventListener(name,fn){this.listeners[name]=fn;},
  getBoundingClientRect(){return {left:100,width:200};},
  setAttribute(){},
});
const list=element("div","rally-grid");
const ui={list,video:{currentTime:0},caption:{textContent:""},previous:{disabled:false},next:{disabled:false}};
const s={items:[{
  candidate_id:"a".repeat(24),detected_start_frame:1263,detected_end_frame:1592,
  manual_clip:{start_frame:1218,end_frame:1637},
  motion_event_count:7,motion_supported_touch_proposals:7,
  provisional_track_ids:[4,4,9],stable_track_count:6,ball_coverage:0.8,
  estimated_action_counts:{attack:3,dig:2},unclassified_action_count:2,
}],offset:0,limit:24,filtered:1,gold:new Set(),current:"",
selected:{has:()=>false}};
const playbackCalls=[];
const training={active:false};
const timelineRallyLayers={touches:{checked:true},ball:{checked:true}};
const comparison={data:null};
const trainingRallyGroups=()=>({
  touches:s.items.map(item=>({start:item.detected_start_frame,end:item.detected_end_frame,
    members:[item]})),
  ball:(comparison.data?.trajectory_rallies||[]).map(item=>({
    start:item.start,end:item.end,members:[item]}))
});
const context={fps:()=>30,ui,s,wasmReady:true,
  training,reference:{active:false},timelineRallyLayers,comparison,comparisonMatches:data=>Boolean(data),
  trainingRallyGroups,updateTrainingSummaries:()=>{},
  second:frame=>(frame/30).toFixed(1).replace(".",","),
  rallyCardProgresses:new Map(),timeline:{data:null},
  make:element, wipe:el=>el.replaceChildren(),
  valid:id=>/^[a-f0-9]{24}$/.test(id),clock:n=>String(n),
  document:{createTextNode:s=>({textContent:s})},
  view:(...args)=>playbackCalls.push(args),toggleGold(){},toggleSelected(){}};
runInNewContext(helpers+"\n"+estimates+"\n"+renderer+"\nthis.render=cards;",context);
s.items[0].metadata={team_1:[4,9],team_2:[10,11],winning_team:"team_2"};
context.render();
const card=list.children[0];
const teamBlock=card.children.find(node=>node.klass==="rally-card-teams");
assert.ok(teamBlock,"Rally card displays team rosters");
assert.equal(teamBlock.children[0].children[1].textContent,"ID 4, ID 9");
assert.equal(teamBlock.children[1].children[1].textContent,"ID 10, ID 11");
assert.equal(teamBlock.children[0].children.length,2);
assert.equal(teamBlock.children[1].children[2].textContent,"+1");
assert.equal(teamBlock.children.length,2,"Only team rows, no duplicate score caption");
assert.equal(card.klass,"rally-card");
const stats=card.children.find(node=>node.klass==="rally-card-foot").children[0];
assert.deepEqual(stats.children.map(node=>node.klass),["rally-card-duration","rally-card-stat","rally-card-stat"]);
assert.match(css,/\.rally-card-duration\{[^}]*flex-basis:auto;/,"Duration must not reserve a full row");
assert.match(css,/\.rally-card \.card-stats\{[^}]*align-items:baseline;justify-content:space-between;/,"Metrics share a baseline and use the available width");
const progress=card.children.find(node=>node.klass==="rally-card-progress");
assert.equal(card.children.at(-1),progress,"Timeline is the last card child");
assert.match(css,/grid-template-areas:"head" "foot" "types" "teams" "progress"/);
assert.match(css,/\.rally-card-teams\{grid-area:teams;/);
assert.ok(progress,
  "Each rally reserves a timeline slot without moving the list during selection");
let bubbling=false;
progress.listeners.click({
  clientX:200,stopPropagation(){bubbling=true;}
});
assert.equal(bubbling,true,"Card time clicks must not bubble into the card-start handler");
assert.equal(playbackCalls.length,1);
assert.equal(playbackCalls[0][0],s.items[0]);
assert.equal(playbackCalls[0][1],(1263+330/2)/30,
  "Card progress uses detected range, not the 45-frame montage padding");
assert.ok(Math.abs(playbackCalls[0][1]-(1263+330/2)/30)<0.00001,
  "Clicking the center of a card must seek to the center of its detected clip");
playbackCalls.length=0;
progress.listeners.keydown({
  key:"End",preventDefault(){},stopPropagation(){},shiftKey:false
});
assert.equal(playbackCalls.length,1);
assert.ok(playbackCalls[0][1]>=(1592/30),
  "End key should seek toward the final frame");
const values=(node)=>[node.textContent,...(node.children||[]).flatMap(values)];
const text=values(card).join(" ");
assert.match(text,/11\.0 с/,"duration should use every frame including final");
assert.match(text,/42,1–53,1 с/,
  "Start and end must show absolute seconds as a readable interval");
assert.match(text,/Касаний 7/);
assert.match(text,/Игроков 6/,"All observed players, including players without touches");
assert.match(text,/Атака ×3/,"Show all provisional attacks in the card");
assert.match(text,/Приём снизу ×2/,"Show provisional digs in the card");
assert.match(css,/grid-template-areas:"head" "foot"/);
assert.match(css,/#studio-preview-view \.rally-card\{[\s\S]*?min-height:0;/,
  "Rally card must use the compact preview grid sizing");
assert.match(css,/#studio-preview-view \.rally-card-time-value\{/,
  "Absolute start and end seconds should share one readable line");
assert.match(css,/#studio-preview-view \.rally-card \.pick\{[\s\S]*?font-size:10px/,
  "Montage label must stay readable");
assert.match(text,/☆/,"Favorite toggle marks rallies for montage");
assert.match(css,/#studio-preview-view \.rally-card\.active\{[\s\S]*?inset 3px 0 #70aaff/,
  "Selected rally uses a restrained left rail, not a neon outline");
assert.match(css,/#studio-preview-view \.rally-card \.card-bottom\{[\s\S]*?display:flex/);
assert.match(css,/#studio-preview-view \.rally-card-time-range/);
s.items[0].stable_track_count=undefined;
context.render();
assert.match(values(list.children[0]).join(" "),/Игроков —/,
  "No Track ID evidence must not be represented as zero players");
// A fast bookmarks fetch / WASM initialization must never replace a pending
// CPU page load with the misleading "No rallies found" placeholder.
ui.filter={value:"all"};
s.run="r";s.total=76;s.filtered=76;s.items=[];
s.pageLoading=true;s.pageError="";
context.render();
assert.match(list.children[0].textContent,/Загрузка розыгрышей/i,
  "An in-flight CPU request is not an empty CPU result");
s.pageLoading=false;s.pageError="Source checksum mismatch";
context.render();
assert.match(list.children[0].textContent,/Source checksum mismatch/,
  "The real backend/render error remains visible");
s.pageError="";s.filtered=0;ui.filter.value="marked";
context.render();
assert.match(list.children[0].textContent,/нет избранных розыгрышей/i,
  "Empty favorites must not say the CPU analysis has no rallies");
ui.filter.value="all";s.total=0;
context.render();
assert.match(list.children[0].textContent,/нет обнаруженных розыгрышей/i,
  "Only the genuinely empty run may show the no-rallies message");
training.active=true;
ui.filter.value="all";
s.run="r";s.pageLoading=false;s.pageError="";s.total=1;s.filtered=1;
s.items=[{
  candidate_id:"b".repeat(24),detected_start_frame:120,detected_end_frame:230,
  motion_event_count:2,provisional_track_ids:[7],estimated_action_counts:{},unclassified_action_count:2
}];
comparison.data={trajectory_status:"ready",trajectory_rallies:[
  {start:60,end:180},{start:300,end:420}
]};
context.render();
assert.equal(list.children.length,2,"Only ball-trajectory rallies appear in training");
assert.ok(list.children.every(item=>item.klass.includes("training-ball")),
  "Old touch-based rally cards must never be shown");
assert.match(values(list.children[0]).join(" "),/По мячу/);
assert.match(values(list.children[0]).join(" "),/не подтверждены/,
  "Ball-only rallies must not claim verified player or touch metrics");
assert.ok(!values(list.children[0]).join(" ").includes("☆"),
  "Ball-only proposals must not be offered as CPU favorites");
const selectedBall=list.children[0];
playbackCalls.length=0;
selectedBall.listeners.click({target:{closest:()=>null}});
assert.equal(playbackCalls.length,1);
assert.equal(playbackCalls[0][0].ballRally,true);
assert.equal(playbackCalls[0][0].detected_start_frame,60);
timelineRallyLayers.touches.checked=false;
context.render();
assert.equal(list.children.length,2);
timelineRallyLayers.touches.checked=true;
context.render();
assert.equal(list.children.length,2,"Legacy touch toggle cannot enable other rally cards");
timelineRallyLayers.ball.checked=false;
context.render();
assert.match(list.children[0].textContent,/Включите розыгрыши по траектории/);
assert.equal(list.children.length,1);
timelineRallyLayers.ball.checked=true;
comparison.data=null;
context.render();
assert.match(list.children[0].textContent,/ещё не рассчитаны/);
training.active=false;
context.render();
assert.equal(list.children.length,1,
  "Ordinary preview keeps CPU rally cards even when training filters are off");
assert.equal(list.children[0].klass,"rally-card");
console.log("Studio rally cards: trajectory-only training, seek, and existing standard preview: OK");
