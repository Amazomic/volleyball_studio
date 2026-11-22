import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const start=source.indexOf("  const renderSidebarPlayers=()=>{");
const end=source.indexOf("  const renderPlayers=()=>{",start);
const make=(tag,klass,text)=>({tag,klass,text,children:[],events:{},attrs:{},
  append(...items){this.children.push(...items);},
  setAttribute(name,value){this.attrs[name]=value;},
  addEventListener(type,fn){this.events[type]=fn;}});
const sidebar=make("div");
const events=[],confirmations=[];
const context={state:{players:[{id:"a",name:"Анна",saved_photo_count:3}],projectMemberIds:new Set(["a"]),trackCovers:new Map(),playerTrackCounts:{a:2}},
  $:()=>sidebar,make,clear:element=>{element.children=[];},URLSearchParams,projectId:()=>"project",
  window:{dispatchEvent:event=>events.push(event),confirm:message=>{
    confirmations.push(message);return true;
  }},CustomEvent:class{constructor(type,opts){this.type=type;this.detail=opts.detail;}}};
runInNewContext(source.slice(start,end)+"\nthis.render=renderSidebarPlayers;",context);
context.render();
const card=sidebar.children[0];
assert.equal(card.children[1].children[1].text,"3 сохранённых фото · 2 привязанных ID");
const button=card.children.find(item=>item.tag==="button"&&item.klass==="players-side-select");
assert.ok(button,"player card provides selection control");
button.events.click();
assert.equal(events[0].type,"volley:player-gallery-select");
assert.equal(events[0].detail.player_id,"a");
assert.equal(events[0].detail.project_id,"project");
const tracks=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8");
const viewStart=tracks.indexOf('  const viewBar=node("div","track-view-bar");');
const viewEnd=tracks.indexOf('  ui.list.before(playerGallery);',viewStart);
const handlers={},inserted=[],loads=[];
const galleryState={project:"project",projectPlayerIds:new Set(["a"]),viewPlayer:"",viewConflicts:false,
  identityConflictIds:new Set(),offset:48,tracksLoaded:true,unassignedIds:new Set([1,2,3])};
const modeSwitch=make("div"),modeTabs=make("div");modeSwitch.append(modeTabs);
const galleryContext={state:galleryState,node:make,ui:{list:{before:node=>inserted.push(node)}},
  window:{addEventListener:(name,fn)=>{handlers[name]=fn;}},groupsMode:make("button"),
  playerMode:make("button"),modeSwitch,groupingControls:make("div"),autoAssignControls:make("div"),setMode:mode=>loads.push(mode),loadAlbum:()=>loads.push("album"),loadPage:()=>loads.push(galleryState.viewPlayer)};
runInNewContext(tracks.slice(viewStart,viewEnd)+"\nthis.updateUnassignedLabel=updateUnassignedLabel;",galleryContext);
galleryContext.updateUnassignedLabel();
assert.equal(modeSwitch.children[0],modeTabs);
assert.equal(modeSwitch.children[1].klass,"track-view-bar","conflict filter shares the same row as ID tabs");
assert.equal(modeSwitch.children[2],galleryContext.groupingControls);
assert.equal(modeSwitch.children[3].textContent,"Непривязанные Track ID · 3");
assert.equal(inserted[0],modeSwitch,"one toolbar is inserted instead of two stacked rows");
assert.equal(inserted[1],galleryContext.autoAssignControls,"auto-assign threshold is mounted beside the gallery toolbar");
galleryState.unassignedIds.delete(1);galleryContext.updateUnassignedLabel();
assert.equal(modeSwitch.children[3].textContent,"Непривязанные Track ID · 2");
handlers["volley:player-gallery-select"]({detail:{project_id:"project",player_id:"a",name:"Анна"}});
assert.equal(galleryState.viewPlayer,"a");
assert.equal(galleryState.offset,0);
assert.deepEqual(loads,["player","album","a"]);
const back=modeSwitch.children[1].children[0];
back.events.click();
assert.equal(galleryState.viewPlayer,"");
assert.equal(modeSwitch.children[3].textContent,"Непривязанные Track ID · 2");
assert.equal(loads.at(-1),"");
console.log("Player card opens its photo album: OK");
const remove=card.children.find(item=>item.tag==="button"&&item.klass==="players-side-remove");
assert.ok(remove,"player card has a compact remove button instead of the action menu");
assert.equal(remove.text,"−");
assert.equal(remove.attrs["aria-label"],"Удалить Анна из проекта");
assert.ok(!card.children.some(item=>item.tag==="details"),"three-dot menu has been removed");
remove.events.click();
assert.match(confirmations[0],/Фото останутся в библиотеке/);
assert.equal(events.at(-1).type,"volley:remove-project-player");
assert.equal(events.at(-1).detail.player_id,"a");
assert.equal(events.at(-1).detail.project_id,"project");
console.log("Player removal control uses project-only event and keeps photo library: OK");
