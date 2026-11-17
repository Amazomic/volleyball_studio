import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {dirname, join} from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const js = readFileSync(join(root, "rust", "volley-app", "static", "studio.js"), "utf8");
const css = readFileSync(join(root, "rust", "volley-app", "static", "studio.css"), "utf8");
const rust = readFileSync(join(root, "rust", "volley-app", "src", "projects.rs"), "utf8");

const begin = js.indexOf("    const formatVideoDuration=seconds=>{");
const end = js.indexOf("    const renderPlayers=(players,selected)=>{", begin);
assert(begin >= 0 && end > begin, "Linked video renderer must exist");
const source = js.slice(begin, end);
assert(!source.includes("/api/videos"), "Global video catalog must never appear in the picker");
assert(js.includes("const selectedVideos=()=>[...linkedFiles.keys()]"),
  "Only explicitly linked videos may be selected");
assert(!js.includes("refreshVideos(new Set(),token)"),
  "Opening a new project must not fetch existing videos");
assert(rust.includes('"duration_sec":duration_sec'),
  "Native source picker must include probed duration");
assert(css.includes("display:flex;flex-direction:column;height:214px;max-height:214px;"),
  "The list must keep a five-row scroll viewport");
assert(css.includes(".studio-project-linked-video{flex:0 0 36px;"),
  "Linked video rows must be compact and full width");

const linkedFiles = new Map([
  ["one", {display_name:"game_1.mp4",duration_sec:59.5}],
  ["two", {display_name:"game_2.mov",duration_sec:125}],
  ["three", {display_name:"game_3.mkv",duration_sec:3601}],
  ["four", {display_name:"game_4.mp4",duration_sec:0}],
  ["five", {display_name:"game_5.mp4",duration_sec:90}],
  ["six", {display_name:"game_6.mp4",duration_sec:120}],
]);
const nodes = [];
const ui = {projectVideos: {
  replaceChildren(...children){nodes.splice(0,nodes.length,...children);},
  append(node){nodes.push(node);},
}};
let updated = 0;
const updateSubmit = ()=>updated++;
const make = (tag, classes="", text="")=>({
  tag,classes,text,children:[],attrs:{},listeners:{},
  append(...children){this.children.push(...children);},
  addEventListener(type, handler){this.listeners[type]=handler;},
  setAttribute(name,value){this.attrs[name]=value;},
});
const test = new Function("ui", "linkedFiles", "make", "updateSubmit",
  source+"\nreturn {formatVideoDuration, renderLinkedFiles};")(
  ui,linkedFiles,make,updateSubmit);

assert.equal(test.formatVideoDuration(5), "00:05");
assert.equal(test.formatVideoDuration(60), "01:00");
assert.equal(test.formatVideoDuration(3601), "60:01");
assert.equal(test.formatVideoDuration(NaN), "--:--");
test.renderLinkedFiles();
assert.equal(nodes.length, 6);
for(const node of nodes){
  assert.equal(node.children.length, 3);
  assert.equal(node.children[0].tag, "span");
  assert.equal(node.children[1].tag, "time");
  assert.equal(node.children[2].tag, "button");
  assert.equal(node.children[2].type, "button");
  assert(node.children[2].attrs["aria-label"].startsWith("Отвязать "));
}
assert.equal(nodes[0].children[1].text, "01:00");
assert.equal(nodes[2].children[1].text, "60:01");
ui.projectVideos.scrollTop=42;
nodes[0].children[2].listeners.click();
assert.equal(ui.projectVideos.scrollTop, 42, "Unlink must preserve scroll position");
assert.equal(linkedFiles.size, 5, "Unlink must remove exactly one item");
assert.equal(nodes.length, 5);
for(const node of [...nodes])node.children[2].listeners.click();
assert.equal(linkedFiles.size, 0);
assert.equal(nodes.length, 1, "The empty state should replace removed videos");
assert(nodes[0].text.includes("Видео пока не выбраны"));
assert(updated >= 7, "Submit state must track each list update");
console.log("Studio linked videos: PASS (only selected, mm:ss, unlink, five-row layout)");
