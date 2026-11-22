import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const start=source.indexOf("  const playerCandidates=ids=>{");
const end=source.indexOf("  const renderRows=()=>{",start);
const node=(tag,klass,text)=>({tag,klass,text,children:[],dataset:{},events:{},value:"",
  append(...items){this.children.push(...items);},setAttribute(){},addEventListener(name,fn){this.events[name]=fn;}});
const state={roster:[{id:"a",name:"Анна"},{id:"b",name:"Борис"},{id:"c",name:"Вера"}],
  playerMatches:{"7":[{player_id:"a",cosine_similarity:.37},{player_id:"b",cosine_similarity:.79}]}};
const binds=[];
const context={state,node,updateCardLink:()=>{},bindSelected:(ids,player)=>binds.push({ids,player})};
runInNewContext(source.slice(start,end)+"\nthis.attachSuggestions=attachSuggestions;",context);
const card=node("article");
context.attachSuggestions(card,[7]);
const panel=card.children[0],similarity=panel.children[0],controls=panel.children[1];
assert.equal(panel.children.length,2,"show a visible similarity label next to the selector");
assert.match(similarity.text,/Борис.*79\.0%/);
const select=controls.children[0],button=controls.children[1];
assert.equal(select.value,"b","highest similarity is selected by default");
assert.deepEqual(select.children.map(option=>option.text),["Борис · 79.0%","Анна · 37.0%","Вера · нет данных ReID"]);
assert.equal(binds.length,0,"selecting a suggestion never assigns automatically");
button.events.click();
assert.equal(binds[0].player,"b");
state.playerMatches={};
const unmatched=node("article");context.attachSuggestions(unmatched,[7]);
assert.equal(unmatched.children[0].children[1].children[0].children[0].value,"", "no guess when similarity is unavailable");
state.playerMatches={"7":[{player_id:"a",cosine_similarity:.865,reference_kind:"reviewed_track"}]};
const reviewed=node("article");context.attachSuggestions(reviewed,[7]);
assert.match(reviewed.children[0].children[0].text,/86\.5% · с подтверждённым ID/);
state.playerMatches={
  "7":[{player_id:"a",cosine_similarity:.98},{player_id:"b",cosine_similarity:.61}],
  "8":[{player_id:"b",cosine_similarity:.82}],
  "9":[{player_id:"b",cosine_similarity:.84}]
};
const group=node("article");context.attachSuggestions(group,[7,8,9]);
const groupControls=group.children[0].children[1];
assert.equal(groupControls.children[0].value,"b",
  "An isolated 98% crop must not outrank three consistent group observations");
assert.match(group.children[0].children[0].text,/по 3 из 3 ID/);
state.playerMatches={"7":[{player_id:"a",cosine_similarity:.99}]};
const uncertain=node("article");context.attachSuggestions(uncertain,[7,8,9]);
assert.equal(uncertain.children[0].children[1].children[0].value,"",
  "A group cannot preselect its owner based on a single fragment");
console.log("Player dropdown ranks grouped Track IDs by consistent multi-ID evidence: OK");
