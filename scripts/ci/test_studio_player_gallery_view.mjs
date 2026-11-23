import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import assert from "node:assert/strict";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tabs.js",import.meta.url),"utf8");
const start=source.indexOf("  const renderSidebarPlayers=()=>{");
const end=source.indexOf("  const renderPlayers=()=>{",start);
const make=(tag,klass,text)=>({tag,klass,text,children:[],events:{},append(...items){this.children.push(...items);},prepend(...items){this.children.unshift(...items);},
  addEventListener(type,fn){this.events[type]=fn;},setAttribute(key,value){
    this.attributes??={};this.attributes[key]=value;
  }});
const sidebar=make("div");
const events=[];
const context={state:{players:[{id:"a",name:"Анна",saved_photo_count:3}],projectMemberIds:new Set(["a"]),trackCovers:new Map(),playerTrackCounts:{a:2}},
  $:()=>sidebar,make,clear:element=>{element.children=[];},URLSearchParams,projectId:()=>"project",
  window:{dispatchEvent:event=>events.push(event),confirm:()=>true},
  CustomEvent:class{constructor(type,opts){this.type=type;this.detail=opts.detail;}}};
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
  identityConflictIds:new Set(),selected:new Set(),offset:48,tracksLoaded:true,unassignedIds:new Set([1,2,3]),excludedIds:new Set([9]),viewExcluded:false};
const modeSwitch=make("div"),modeTabs=make("div"),autoAssignControls=make("div");
modeTabs.append(make("button","","ID"),make("button","","Группы ID"));autoAssignControls.append(make("label"),make("button"));
const galleryContext={state:galleryState,node:make,ui:{list:{before:node=>inserted.push(node)}},
  window:{addEventListener:(name,fn)=>{handlers[name]=fn;}},groupsMode:make("button"),
  playerMode:make("button"),modeSwitch,modeTabs,groupingControls:make("div"),autoAssignControls,setMode:mode=>loads.push(mode),loadAlbum:()=>loads.push("album"),loadPage:()=>loads.push(galleryState.viewPlayer)};
runInNewContext(tracks.slice(viewStart,viewEnd)+"\nthis.updateUnassignedLabel=updateUnassignedLabel;",galleryContext);
galleryContext.updateUnassignedLabel();
assert.equal(modeSwitch.children[0].klass,"track-filter-buttons","conflict and excluded filters occupy the upper row");
assert.equal(modeSwitch.children[0].children.length,2);
const modeInfo=autoAssignControls.children[0],infoLabel=modeInfo.children[1];
assert.equal(modeInfo.klass,"track-mode-info","ID selector and track count share lower toolbar");
assert.equal(modeInfo.children[0],modeTabs,"ID and groups occupy the lower toolbar");
assert.equal(modeTabs.children[0].text,"ID");
assert.equal(modeTabs.children[1].text,"Группы ID");
assert.equal(modeSwitch.children[1].klass,"track-view-bar","player back button stays beside ID tabs");
assert.equal(modeSwitch.children[1].children.length,1,"only the back button stays in the upper toolbar");
assert.equal(autoAssignControls.children[0],modeInfo,"lower toolbar retains grouped mode and info controls");
assert.equal(modeInfo.children.length,2,"mode and Track ID info are both inside lower toolbar");
assert.equal(autoAssignControls.children[1].tag,"label","threshold stays after filters");
assert.equal(autoAssignControls.children[2].tag,"button","auto-assign action stays after threshold");
assert.equal(modeSwitch.children[2],galleryContext.groupingControls);
assert.equal(infoLabel.textContent,"Непривязанные Track ID · 3");
assert.equal(inserted[0],modeSwitch,"mode toolbar is inserted above auto-assign controls");
assert.equal(inserted[1],galleryContext.autoAssignControls,"auto-assign threshold is mounted beside the gallery toolbar");
galleryState.unassignedIds.delete(1);galleryContext.updateUnassignedLabel();
assert.equal(infoLabel.textContent,"Непривязанные Track ID · 2");
handlers["volley:player-gallery-select"]({detail:{project_id:"project",player_id:"a",name:"Анна"}});
assert.equal(galleryState.viewPlayer,"a");
assert.equal(galleryState.offset,0);
assert.deepEqual(loads,["player","album","a"]);
const back=modeSwitch.children[1].children[0];
back.events.click();
assert.equal(galleryState.viewPlayer,"");
assert.equal(infoLabel.textContent,"Непривязанные Track ID · 2");
assert.equal(loads.at(-1),"");
const excludedToggle=modeSwitch.children[0].children[1];
assert.match(excludedToggle.textContent,/Исключённые · 1/);
excludedToggle.events.click();
assert.equal(galleryState.viewExcluded,true);
assert.equal(infoLabel.textContent,"Исключённые Track ID · 1");
excludedToggle.events.click();
assert.equal(galleryState.viewExcluded,false);
assert.equal(infoLabel.textContent,"Непривязанные Track ID · 2");
console.log("Player card opens its photo album: OK");
const remove=card.children.find(item=>item.tag==="button"&&item.klass==="players-side-remove");
assert.ok(remove,"Player sidebar must expose the project-unlink control");
assert.match(remove.title,/фото останутся в библиотеке/);
assert.equal(remove.attributes["aria-label"],"Удалить Анна из проекта");
remove.events.click();
assert.equal(events.at(-1).type,"volley:remove-project-player");
assert.equal(events.at(-1).detail.player_id,"a");
assert.equal(events.at(-1).detail.project_id,"project");
