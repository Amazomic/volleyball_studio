import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";

const source=readFileSync(new URL("../../rust/volley-app/static/studio_tracks.js",import.meta.url),"utf8").replace(/\r\n/g,"\n");
const startMarker='refreshReferences.addEventListener("click",async()=>{';
const endMarker='\n    });\n    const resetLinks=';
const start=source.indexOf(startMarker);
const end=source.indexOf(endMarker,start);
assert.ok(start>=0&&end>start,"refresh click handler must exist");
const body=source.slice(start+startMarker.length,end);

async function runScenario({fail=false,waitForPhotos=false}={}){
  const updates=[],active=new Map();
  let releaseApi;
  const apiGate=new Promise((resolve,reject)=>{releaseApi={resolve,reject};});
  let releasePhotos;
  const photoGate=new Promise(resolve=>{releasePhotos=resolve;});
  const photoEnrollmentQueues=new Map(waitForPhotos?[["photo",photoGate]]:[]);
  const refreshReferences={disabled:false,setAttribute(){},removeAttribute(){}};
  const refreshStatus={hidden:true,textContent:""};
  const state={busy:false,rosterBusy:false,project:"project",video:"match.mp4",
    replay:"cpu",rallyRun:"rally",sourceSha:"sha",token:1,writable:true};
  const calls=[];
  let timerCallback;
  const window={crypto:{getRandomValues:a=>a.fill(1)},setTimeout(fn){timerCallback=fn;return 1;},clearTimeout(){timerCallback=null;},VolleyballTasks:{
    set(key,value){active.set(key,value);updates.push({key,...value});},
    clear(key){active.delete(key);calls.push("clear:"+key);}
  },dispatchEvent(){}};
  const context={refreshReferences,refreshStatus,state,photoEnrollmentQueues,window,
    Event:class{constructor(name){this.type=name;}},
    post:async (url,payload,progressId)=>{calls.push(url);assert.match(progressId,/^[a-f0-9]{32}$/);return apiGate;},
    read:async()=>({active:true,stage:"track_features",completed:24,total:48}),
    loadLibraryRoster:async()=>{calls.push("roster");},
    refreshPhotoMatches:async(scope)=>{assert.equal(scope.reid_device,"gpu");calls.push("matches");return {trackCount:7};},
    autoAssignSimilar:async()=>{calls.push("auto");}};
  const handler=vm.runInNewContext("(async()=>{"+body+"\n})",context);
  const pending=handler();
  const key="player-references-refresh";
  assert.equal(refreshReferences.disabled,true);
  assert.ok(active.has(key),"refresh must appear in bottom sidebar before awaiting the API");
  assert.equal(active.get(key).label,"Обновление признаков");
  assert.equal(active.get(key).progress,null,"no invented percentage without server progress");
  assert.equal(refreshStatus.hidden,true,"processing text must not shift the players card layout");
  if(waitForPhotos){
    assert.match(active.get(key).stage,/сохранения фотографий/i);
    releasePhotos();
    await new Promise(resolve=>setImmediate(resolve));
  }
  assert.match(active.get(key).stage,/эталонов/i);
  assert.equal(typeof timerCallback,"function","poll real server progress during pending request");
  await timerCallback();
  assert.equal(active.get(key).progress,50);
  assert.match(active.get(key).stage,/24\s*\/\s*48/);
  if(fail)releaseApi.reject(Error("network error"));
  else releaseApi.resolve({added:2,deactivated:1,confirmed_tracks:3});
  await pending;
  assert.equal(refreshReferences.disabled,false);
  assert.equal(timerCallback,null,"polling stops on completion or error");
  assert.equal(active.has(key),false,"task must be removed after success or failure");
  assert.ok(calls.includes("clear:"+key));
  if(fail){
    assert.match(refreshStatus.textContent,/Ошибка обновления признаков: network error/);
    assert.equal(calls.includes("matches"),false);
  }else{
    assert.deepEqual(calls.slice(-3),["roster","matches","clear:"+key]);
    assert.match(refreshStatus.textContent,/Сопоставлено ID: 7/);
    assert.match(updates.map(u=>u.stage).join("|"),/Пересчёт сходства/);
    assert.equal(calls.includes("auto"),false,"Reference refresh must not run automatic assignment");
  }
}
await runScenario();
await runScenario({fail:true});
await runScenario({waitForPhotos:true});
console.log("PASS: reference refresh uses bottom sidebar tasks and cleans up on success/failure");
