"use strict";
// Transitional same-origin browser bridge. Rust validates all source IDs,
// contact evidence, gold revisions, original clip ranges and montage jobs.
(() => {
  const $ = id => document.getElementById(id);
  const ui = {
    project:$("studio-project"),analysis:$("studio-inference"),run:$("studio-run"),
    projectNew:$("studio-project-new"),projectDialog:$("studio-project-dialog"),
    projectDrop:$("studio-project-drop"),projectFiles:$("studio-project-file-input"),
    projectUploadStatus:$("studio-project-upload-progress"),
    projectPlayers:$("studio-project-players"),projectPlayerName:$("studio-project-player-name"),
    projectAddPlayer:$("studio-project-add-player"),
    createForm:$("studio-project-create-form"),projectName:$("studio-project-name"),
    projectVideos:$("studio-project-videos"),projectSubmit:$("studio-project-submit"),
    projectCreateError:$("studio-project-create-error"),
    projectCreateStatus:$("studio-project-create-status"),
    projectCancel:$("studio-project-cancel"),projectClose:$("studio-project-close"),

    list:$("studio-rallies"),filter:$("studio-filter"),
    selectedCount:$("studio-selected-count"),banner:$("studio-banner"),
    video:$("studio-video"),
    emptyVideo:$("studio-video-empty"),videoTime:$("studio-video-time"),
    viewTitle:$("studio-view-title"),viewDescription:$("studio-view-description"),
    events:$("studio-event-summary"),selectionText:$("studio-selection-text"),
    picked:$("studio-picked"),encoder:$("studio-encoder"),music:$("studio-music"),
    layout:$("studio-layout"),volume:$("studio-volume"),
    render:$("studio-render"),job:$("studio-job"),cancel:$("studio-cancel"),
    download:$("studio-download")
  };
  const previewIsVisible=()=>!document.hidden&&
    !$("studio-preview-view").classList.contains("hidden");
  const s = {
    project:"",analysis:"",run:"",video:"",items:[],total:0,offset:0,limit:24,
    projectCatalog:[],projectVideos:[],chain:null,chainEpoch:0,catalogEpoch:0,
    filtered:0,gold:new Set(),revision:"absent",writable:false,marksLoaded:false,selected:new Map(),
    current:"",videoKey:"",seekAt:0,stopAt:null,clipStart:null,clipEnd:null,analysisOffset:0,
    playbackMode:"sequence",generation:0,busy:false,jobId:"",timer:null,
    loadingMore:false,canLoadMore:true,pageLoading:false,pageError:""
  };
  const previewLoad={key:"",epoch:0,pending:false,retries:0,timer:null};
  // Persist selected source IDs only; restored IDs are validated against each catalog.
  const sourceSelection=(()=>{
    const key="volley:studio-source-selection:v1";
    const read=()=>{
      try{
        const raw=localStorage.getItem(key);
        if(!raw||raw.length>2048)return {};
        const value=JSON.parse(raw);
        return value&&typeof value==="object"&&!Array.isArray(value)?value:{};
      }catch{return {};}
    };
    const save=(project,analysis="",run="")=>{
      try{localStorage.setItem(key,JSON.stringify({project,analysis,run}));}catch{}
    };
    return {read,save};
  })();
  // Persist the small, source-scoped GPU -> CPU index. The UI can restore
  // selectors synchronously on refresh while metadata is checked in background.
  // Never cache video bytes or claim this snapshot verifies source evidence.
  const chainSnapshot=(()=>{
    const key="volley:studio-chain:v2";
    const read=(project,videos)=>{
      try{
        const raw=localStorage.getItem(key);
        if(!raw||raw.length>350000)return null;
        const row=JSON.parse(raw);
        if(row.project!==project||row.expires<=Date.now()||
           JSON.stringify(row.videos)!==JSON.stringify(videos)||
           !Array.isArray(row.overviews)||!Array.isArray(row.runs))return null;
        const overviews=new Map();
        for(const entry of row.overviews){
          if(!Array.isArray(entry)||entry.length!==2||!videos.includes(entry[0])||
             entry[1]?.schema!=="volley-native-analysis-overview/1"||
             entry[1].project_id!==project||entry[1].video!==entry[0])return null;
          overviews.set(entry[0],entry[1]);
        }
        if(overviews.size!==videos.length)return null;
        return window.VolleyStudioChain?.build(videos,overviews,row.runs)||null;
      }catch{return null;}
    };
    const save=(project,videos,overviews,runs)=>{
      try{
        const packed=JSON.stringify({
          project,videos,overviews:[...overviews],runs,
          expires:Date.now()+86400000
        });
        if(packed.length<=350000)localStorage.setItem(key,packed);
      }catch{} // Storage may be disabled or full.
    };
    const drop=()=>{try{localStorage.removeItem(key);}catch{}};
    return {read,save,drop};
  })();
  const url=(path,params)=>path+(params?"?"+new URLSearchParams(params).toString():"");
  const request=async(path,opts={})=>{
    const res=await window.VolleyballStudioActivity.fetch(path,{cache:"no-store",credentials:"same-origin",...opts});
    let data={};try{data=await res.json();}catch{}
    if(!res.ok)throw Error(typeof data.message==="string"?data.message:
      typeof data.error==="string"?data.error:"Ошибка HTTP "+res.status);
    return data;
  };
  const post=(path,payload)=>request(path,{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  // Cache only small derived metadata and immutable CPU preview pages, never video bytes.
  // A versioned session cache survives refresh; all writes / new pipeline results invalidate it.
  const previewCache=(()=>{
    const prefix="volley:studio-preview:v5:";
    const memory=new Map(),inflight=new Map(),limit=24;
    let epoch=0;
    const storage=()=>{
      try{return typeof sessionStorage==="undefined"?null:sessionStorage;}catch{return null;}
    };
    const read=path=>{
      let row=memory.get(path);
      if(!row){
        const store=storage();
        if(store){
          try{row=JSON.parse(store.getItem(prefix+path)||"null");}catch{}
        }
      }
      if(!row)return null;
      if(row.expires<=Date.now()){
        memory.delete(path);
        try{storage()?.removeItem(prefix+path);}catch{}
        return null;
      }
      memory.delete(path);memory.set(path,row);
      return row.value;
    };
    const save=(path,value,ttl,persist)=>{
      const row={expires:Date.now()+ttl,value};
      memory.delete(path);memory.set(path,row);
      while(memory.size>limit)memory.delete(memory.keys().next().value);
      if(!persist)return;
      try{
        const store=storage();
        if(!store)return;
        const packed=JSON.stringify(row);
        if(packed.length>240000)return; // No multi-MB evidence arrays in sessionStorage.
        const entries=[];
        for(let i=0;i<store.length;i++){
          const key=store.key(i);
          if(key?.startsWith(prefix))entries.push(key);
        }
        if(entries.length>=12){
          // Keep this bounded even across dozens of analysis branches.
          for(const key of entries.slice(0,entries.length-9))store.removeItem(key);
        }
        store.setItem(prefix+path,packed);
      }catch{} // Storage quotas / private mode must not break preview.
    };
    const drop=predicate=>{
      epoch++;
      for(const key of memory.keys())if(predicate(key))memory.delete(key);
      for(const key of inflight.keys())if(predicate(key))inflight.delete(key);
      const store=storage();
      if(!store)return;
      try{
        const keys=[];
        for(let i=0;i<store.length;i++){
          const key=store.key(i);
          if(key?.startsWith(prefix)&&predicate(key.slice(prefix.length)))keys.push(key);
        }
        for(const key of keys)store.removeItem(key);
      }catch{}
    };
    const load=(path,ttl,persist=false,fresh=false)=>{
      const value=fresh?null:read(path);
      if(value!==null)return Promise.resolve(value);
      if(!fresh&&inflight.has(path))return inflight.get(path);
      const started=epoch;
      const task=request(path).then(value=>{
        if(epoch===started&&inflight.get(path)===task)save(path,value,ttl,persist);
        return value;
      }).finally(()=>{if(inflight.get(path)===task)inflight.delete(path);});
      inflight.set(path,task);
      return task;
    };
    return {load,drop,peek:read};
  })();
  const invalidatePreviewRun=(project,run)=>{
    previewCache.drop(path=>path.startsWith("/api/cpu/native/rallies/preview?")&&
      path.includes("project_id="+encodeURIComponent(project))&&
      path.includes("rally_run_id="+encodeURIComponent(run)));
  };
  const invalidateCatalog=()=>{
    chainSnapshot.drop();
    previewCache.drop(path=>
      path==="/api/projects"||path.startsWith("/api/cpu/native/rallies/runs?")||
      path.startsWith("/api/studio/analysis-overview?")||
      path.startsWith("/api/cpu/native/rallies/marks?")||
      path.startsWith("/api/cpu/native/rallies/preview?")||
      path.startsWith("/api/cpu/native/rallies/timeline?"));
  };
  const make=(tag,klass,text)=>{
    const node=document.createElement(tag);
    if(klass)node.className=klass;
    if(text!==undefined)node.textContent=String(text);
    return node;
  };
  const pickers=new Map();
  const refreshPicker=select=>pickers.get(select)?.refresh();
  const makeAnalysisPicker=(select,primary)=>{
    const isProject=primary==="project";
    const host=make("div","analysis-picker");
    const trigger=make("button","analysis-picker-trigger");
    const list=make("div","analysis-picker-list hidden");
    trigger.type="button";trigger.setAttribute("aria-expanded","false");
    trigger.setAttribute("aria-label",select.getAttribute("aria-label"));
    select.classList.add("analysis-picker-native");
    select.parentNode.insertBefore(host,select.nextSibling);
    host.append(trigger,list);
    let editing=false;
    const close=()=>{list.classList.add("hidden");trigger.setAttribute("aria-expanded","false");};
    const animateRemoval=async row=>{
      if(!row.animate||window.matchMedia?.("(prefers-reduced-motion: reduce)").matches)return;
      const height=row.getBoundingClientRect().height;
      const animation=row.animate([
        {height:height+"px",opacity:1,overflow:"hidden",boxSizing:"border-box"},
        {height:"0px",opacity:0,paddingTop:"0px",paddingBottom:"0px",borderTopWidth:"0px",overflow:"hidden",boxSizing:"border-box"}
      ],{duration:220,easing:"ease-in-out",fill:"forwards"});
      await animation.finished.catch(()=>{});
    };
    const refresh=()=>{
      trigger.disabled=select.disabled;
      trigger.textContent=select.selectedOptions[0]?.textContent||"Выберите результат";
      wipe(list);
      for(const option of select.options){
        if(!option.value)continue;
        const row=make("div","analysis-picker-row");
        const choose=make("button","analysis-picker-choice",option.textContent);
        choose.type="button";choose.setAttribute("aria-pressed",String(option.value===select.value));
        choose.addEventListener("click",()=>{select.value=option.value;select.dispatchEvent(new Event("change",{bubbles:true}));close();refresh();});
        row.append(choose);
        for(const [action,label,symbol] of [["rename","Переименовать","✎"],["delete","Удалить","×"]]){
          const button=make("button","analysis-picker-action analysis-picker-"+action,symbol);
          button.type="button";
          button.title=label+": "+option.textContent;
          button.setAttribute("aria-label",button.title);
          button.addEventListener("click",async()=>{
            if(editing)return;
            if(isProject){
              const name=action==="rename"?window.prompt("Новое название проекта",option.textContent):null;
              if(action==="rename"&&name===null)return;
              editing=true;button.disabled=true;
              const deleting=action==="delete",deletedId=option.value;
              const wasActive=deleting&&s.project===deletedId;
              const removal=deleting?animateRemoval(row):Promise.resolve();
              if(deleting){
                // Optimistic catalog update. A failed server commit restores
                // the authoritative list, without blocking the animation.
                option.remove();
                s.projectCatalog=s.projectCatalog.filter(project=>project.id!==deletedId);
                if(wasActive){
                  ++s.chainEpoch;
                  s.project="";s.analysis="";s.chain=null;s.projectVideos=[];
                  ui.project.value="";reset();
                  options(ui.analysis,[],"value",x=>x,"Сначала выберите проект");
                  options(ui.run,[],"rally_run_id",x=>x,"Сначала выберите анализ");
                  ui.analysis.disabled=true;ui.run.disabled=true;
                  refreshPicker(ui.analysis);refreshPicker(ui.run);
                  sourceSelection.save("","","");
                  window.dispatchEvent(new Event("volley:project-ready"));
                }
              }
              try{
                if(deleting){
                  await post("/api/project-delete",{id:deletedId});
                  invalidateCatalog();
                  await removal;
                  if(wasActive)await projects();
                  else refresh();
                  message("Проект и его связанные данные удалены из каталога.","ok");
                }else{
                  const result=await post("/api/project-update",{id:deletedId,name});
                  const updated=result.project?.name||name.trim();
                  option.textContent=updated;
                  const item=s.projectCatalog.find(project=>project.id===deletedId);
                  if(item)item.name=updated;
                  invalidateCatalog();refresh();
                  message("Название сохранено.","ok");
                }
              }catch(error){
                if(deleting){
                  invalidateCatalog();
                  await removal;
                  try{await projects(wasActive?deletedId:s.project);}catch(refreshError){console.warn(refreshError);}
                }
                failed(error);
              }finally{editing=false;button.disabled=false;}
              return;
            }
            const source=primary?s.chain?.choices.find(item=>item.value===option.value):s.chain?.choices.find(item=>item.value===s.analysis);
            if(!source)return;
            const id=primary?source.id:option.value,kind=primary?source.kind:"rallies";
            const project=s.project;
            let name=null;
            if(action==="rename"){
              name=window.prompt("Новое название результата",primary?source.display_name:(s.chain?.byAnalysis.get(s.analysis)||[]).find(item=>item.rally_run_id===id)?.display_name||"");
              if(name===null)return;
            }
            const deleting=action==="delete";
            editing=true;button.disabled=true;
            // Remove from the visible selector immediately; server atomically
            // commits the linked tree before its background disk cleanup.
            const removal=deleting?animateRemoval(row):Promise.resolve();
            if(deleting)option.remove();
            try{
              await post(deleting?"/api/studio/analysis-edit":"/api/studio/analysis-rename",
                {project_id:project,video:source.video,kind,id,action,name});
              invalidateCatalog();
              if(project!==s.project)return;
              await removal;
              const deletesActiveAnalysis=deleting&&primary&&s.analysis===source.value;
              const deletesActiveRun=deleting&&!primary&&s.run===id;
              const analysis=deletesActiveAnalysis?"":s.analysis;
              const run=deletesActiveAnalysis||deletesActiveRun?"":s.run;
              if(!deleting)close();
              await loadChain(analysis,run,!deletesActiveAnalysis&&!deletesActiveRun);
              message(deleting?"Результат и связанные данные удалены из каталога.":"Название сохранено.","ok");
            }catch(error){
              if(deleting){
                invalidateCatalog();
                await removal;
                if(project===s.project){
                  try{await loadChain(s.analysis,s.run,true);}catch(refreshError){console.warn(refreshError);}
                }
              }
              failed(error);
            }finally{editing=false;button.disabled=false;}
          });
          row.append(button);
        }
        list.append(row);
      }
    };
    trigger.addEventListener("click",()=>{const opening=list.classList.contains("hidden");refresh();list.classList.toggle("hidden",!opening);trigger.setAttribute("aria-expanded",String(opening));});
    select.addEventListener("change",refresh);
    document.addEventListener("click",event=>{if(!host.contains(event.target))close();});
    document.addEventListener("keydown",event=>{if(event.key==="Escape")close();});
    pickers.set(select,{refresh});refresh();
  };
  // Shared by the analysis tab and header: one fetch, cached across page refreshes.
  // The project registry is mutable outside this tab: always read its live version.
  window.VolleyballStudioProjectCatalog=()=>request("/api/projects");
  // Canvas uses saved CPU frame evidence; toggles are local viewing preferences.
  const overlay = {
    canvas:$("studio-video-overlay"),status:$("studio-overlay-status"),
    rallies:$("studio-overlay-rallies"),touches:$("studio-overlay-touches"),
    trajectory:$("studio-overlay-trajectory"),players:$("studio-overlay-players"),
    pose:$("studio-overlay-pose"),court:$("studio-overlay-court"),teams:$("studio-overlay-teams"),
    mode:$("studio-overlay-processed"),rawMode:$("studio-overlay-raw"),
    stabilizerMode:$("studio-debug-stabilizer-compare"),
    rawControls:$("studio-overlay-raw-controls"),actionControls:$("studio-overlay-actions"),
    rawBall:$("studio-raw-ball"),rawBallTrail:$("studio-raw-ball-trail"),rawVballVisible:$("studio-raw-vball-visible"),rawPeopleVisible:$("studio-raw-people-visible"),rawPlayer:$("studio-raw-player"),rawPlayerInterpolated:$("studio-raw-player-interpolated"),
    rawPose:$("studio-raw-pose"),rawPoseVisible:$("studio-raw-pose-visible"),
    raw:false,stabilizerView:false,trajectoryVariant:"after",
    debugTrajectorySwitch:$("studio-debug-trajectory-switch"),
    debugBefore:$("studio-debug-trajectory-before"),debugAfter:$("studio-debug-trajectory-after"),
    debugTrajectoryLegend:$("studio-debug-trajectory-legend"),
    actionFlags:{unknown_touch:true,attack:true,block:true,serve:true,overhead_pass:true,dig:true},
    source:null,cache:new Map(),rawFrames:new Map(),rawDetectors:new Map(),rawOffsets:new Map(),presentedTime:null,
    fetching:new Set(),controllers:new Map(),failed:new Map(),loop:0,
    flags:{rallies:true,touches:true,trajectory:true,players:true,pose:true,court:true,teams:false}
  };
  try {
    const current=localStorage.getItem("volley:preview-overlays:v2");
    const saved=JSON.parse(current||localStorage.getItem("volley:preview-overlays:v1")||"null");
    if(saved&&typeof saved==="object")
      for(const k of Object.keys(overlay.flags))
        if(typeof saved[k]==="boolean"&&(current!==null||k!=="pose"))
          overlay.flags[k]=saved[k];
  }catch{}
  const playerTeamColor=(side,enabled)=>!enabled||side==="near"?"#71acff":side==="far"?"#ff626e":"#b8c2ce";
  const updateOverlayToggles=()=>{
    for(const key of ["rallies","touches","trajectory","players","pose","court","teams"]){
      const button=overlay[key];
      button.setAttribute("aria-pressed",String(overlay.flags[key]));
    }
    overlay.actionControls.classList.toggle("hidden",overlay.raw||!overlay.flags.touches);
  };
  updateOverlayToggles();
  const overlayMessage=text=>{
    overlay.status.textContent=text;
    overlay.status.classList.toggle("hidden",!text);
  };
  const overlayContext=()=>{
    const canvas=overlay.canvas;
    const cw=canvas.clientWidth,ch=canvas.clientHeight;
    if(!cw||!ch)return null;
    const dpr=Math.min(window.devicePixelRatio||1,2);
    const w=Math.round(cw*dpr),h=Math.round(ch*dpr);
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
    const ctx=canvas.getContext("2d");
    if(!ctx)return null;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,cw,ch);
    const vw=ui.video.videoWidth,vh=ui.video.videoHeight;
    if(!vw||!vh)return null;
    const scale=Math.min(cw/vw,ch/vh);
    const width=vw*scale,height=vh*scale;
    return {ctx,x:(cw-width)/2,y:(ch-height)/2,width,height};
  };
  const overlayLabels={
    key:"",names:new Map(),manualIds:new Set(),loaded:false,loading:false,failedUntil:0,
    epoch:0,controller:null
  };
  const playerLabelKey=(project,run,video,sha)=>[project,run,video,sha].join(":");
  const invalidateOverlayLabels=()=>{
    overlayLabels.epoch++;
    overlayLabels.controller?.abort();
    overlayLabels.key="";overlayLabels.names.clear();overlayLabels.manualIds.clear();
    overlayLabels.loaded=false;overlayLabels.loading=false;
    overlayLabels.failedUntil=0;overlayLabels.controller=null;
  };
  const requestOverlayLabels=(project,run,video,sha)=>{
    if(typeof sha!=="string"||!/^[a-f0-9]{64}$/.test(sha))return;
    const key=playerLabelKey(project,run,video,sha);
    if(overlayLabels.key!==key){
      invalidateOverlayLabels();
      overlayLabels.key=key;
    }
    if(overlayLabels.loaded||overlayLabels.loading||overlayLabels.failedUntil>Date.now())return;
    const epoch=overlayLabels.epoch;
    const controller=typeof AbortController==="function"?new AbortController():null;
    overlayLabels.loading=true;overlayLabels.controller=controller;
    request(url("/api/cpu/native/rallies/overlay-player-labels",{
      project_id:project,rally_run_id:run
    }),controller?{signal:controller.signal}:{}).then(value=>{
      if(epoch!==overlayLabels.epoch||key!==overlayLabels.key)return;
      if(value.schema!=="volley-native-preview-player-labels/1"||
         value.project_id!==project||value.rally_run_id!==run||
         value.video!==video||value.source_replay_sha256!==sha||
         !["manual","manual_and_automatic"].includes(value.identity_source)||
         !value.names||typeof value.names!=="object"||Array.isArray(value.names)||
         Object.keys(value.names).length>20000||
         !Array.isArray(value.manual_track_ids)||value.manual_track_ids.length>20000||
         value.manual_track_ids.some(id=>typeof id!=="string"||!/^[1-9]\d*$/.test(id)||
           !Object.hasOwn(value.names,id)))
        throw Error("Имена игроков не соответствуют выбранному анализу");
      const names=new Map();
      for(const [track,name] of Object.entries(value.names)){
        if(/^[1-9]\d*$/.test(track)&&
           typeof name==="string"&&name.trim()&&name.length<=150)
          names.set(track,name.trim());
      }
      overlayLabels.names=names;
      overlayLabels.manualIds=new Set(value.manual_track_ids);
      overlayLabels.loaded=true;
      if(s.project===project&&s.run===run&&s.video===video){drawOverlay();if(typeof cards==="function")cards();}
    }).catch(error=>{
      if(epoch!==overlayLabels.epoch||controller?.signal.aborted||
         error?.name==="AbortError")return;
      overlayLabels.failedUntil=Date.now()+5000;
      if(s.project===project&&s.run===run&&s.video===video)
        overlayMessage("Имена игроков недоступны: "+error.message);
    }).finally(()=>{
      if(epoch===overlayLabels.epoch){
        overlayLabels.loading=false;overlayLabels.controller=null;
      }
    });
  };
  const rawFusion=()=>s.chain?.choices.find(row=>row.value===s.analysis&&row.kind==="fusion")||null;
  const overlaySourceId=()=>overlay.raw?rawFusion()?.id||"":s.run;
  // Identify the detector by the saved GPU fusion, not current runtime settings.
  const rawDetectorKey=(project,run,video)=>[project,run,video].join(":");
  const updateRawDetectorLabel=()=>{
    const input=overlay.rawPeopleVisible;
    const label=input?.closest("label");
    if(!label)return;
    let name=label.querySelector(".raw-source-detector-label");
    if(!name){
      name=label.querySelector("#studio-raw-person-model");
      if(name)name.removeAttribute("id"); // Independent of the global model setting.
      else{
        for(const child of [...label.childNodes])
          if(child.nodeType===3)child.remove();
        name=document.createElement("span");
        label.append(" ",name);
      }
      name.classList.add("raw-source-detector-label");
    }
    const fusion=rawFusion();
    const variant=fusion?overlay.rawDetectors.get(
      rawDetectorKey(s.project,fusion.id,s.video)):null;
    name.textContent=variant==="rfdetr-medium"?"RF-DETR Medium":
      variant==="rfdetr-large"?"RF-DETR Large":
      variant==="legacy"?"Архивный анализ":
      "\u0414\u0435\u0442\u0435\u043a\u0442\u043e\u0440 \u0438\u0433\u0440\u043e\u043a\u043e\u0432";
    input.disabled=!fusion;
    $("studio-raw-player-interpolated-option")?.classList.toggle("hidden",!["rfdetr-medium","rfdetr-large"].includes(variant));
  };
  const overlayWindowStart=frame=>Math.max(0,Math.floor(frame/90)*90-60);
  const overlayWindowKey=(project,run,video,start)=>
    [project,overlay.raw?(overlay.stabilizerView?"gpu-stabilizer":"gpu"):"cpu",run,video,start,...(!overlay.raw&&overlay.flags.pose?["pose"]:[]),...(!overlay.raw&&overlay.flags.teams?["teams"]:[])].join(":");
  const rawFrameLimit=(project,run,video)=>overlay.rawFrames.get([project,run,video].join(":"));
  // The API returns overlapping 180-frame windows every 90 frames.
  // Keep drawing from a previous window while the next one loads.
  const cachedOverlayForFrame=(project,run,video,frame)=>{
    for(const [key,record] of [...overlay.cache].reverse()){
      if(key===overlayWindowKey(project,run,video,record.start_frame)&&
         record.start_frame<=frame&&frame<record.end_frame)return record;
    }
    return null;
  };
  // Fail closed: only provenance-labelled VballNet detections or their
  // bounded interpolation may enter the processed overlay canvas.
  const processedBallOverlayIsValid=value=>
    ["vballnet_only_short_gap_interpolation_v1",
      "vballnet_physical_teleport_repair_player_kink_v3",
      "vballnet_leading_ball_rally_v1",
      "vballnet_local_artifacts_v1"].includes(value.ball_policy)&&
    Array.isArray(value.frames)&&value.frames.every((row,index)=>{
      if(row.frame!==value.start_frame+index)return false;
      if(row.ball===null)
        return row.ball_origin===null&&row.ball_interpolated===false;
      if(!Array.isArray(row.ball)||row.ball.length!==2||
         row.ball.some(coordinate=>!Number.isFinite(coordinate)||coordinate<0||coordinate>1))
        return false;
      return (row.ball_origin==="vballnet"&&row.ball_interpolated===false)||
        (row.ball_origin==="vballnet_interpolated"&&row.ball_interpolated===true);
    });
  const requestOverlayWindow=(project,run,video,start)=>{
    if(overlay.raw&&Number.isSafeInteger(rawFrameLimit(project,run,video))&&
       start>=rawFrameLimit(project,run,video))return;
    const key=overlayWindowKey(project,run,video,start);
    if(overlay.cache.has(key)||overlay.fetching.has(key)||
       (overlay.failed.get(key)||0)>Date.now())return;
    // Obsolete seeks must not queue unbounded expensive verified replay reads.
    if(overlay.fetching.size>=3){
      const stale=overlay.controllers?.keys().next().value;
      if(!stale)return;
      overlay.controllers.get(stale)?.abort();
      overlay.controllers.delete(stale);
      overlay.fetching.delete(stale);
    }
    overlay.failed.delete(key);
    const controller=typeof AbortController==="function"?new AbortController():null;
    overlay.controllers?.set(key,controller);
    overlay.fetching.add(key);
    const isRaw=overlay.raw;
    const path=isRaw?url("/api/gpu/native/overlays",{
      project_id:project,video,fusion_id:run,start_frame:start,count:180,
      ...(overlay.stabilizerView?{stabilizer:true}:{})
    }):url("/api/cpu/native/rallies/overlays",{
      project_id:project,rally_run_id:run,start_frame:start,count:180,
      ...(overlay.flags.pose?{pose:true}:{}),...(overlay.flags.teams?{team_sides:true}:{})
    });
    request(path,controller?{signal:controller.signal}:{}).then(value=>{
      if(controller?.signal.aborted)return;
      const selected=timeline.data?.project_id===project&&
        timeline.data?.rally_run_id===run&&timeline.data?.video===video?
        timeline.data:null;
      if(value.schema!==(isRaw?"volley-native-gpu-overlay/1":"volley-native-preview-overlay/1")||
         (!isRaw&&selected?.source_replay_sha256&&
          value.replay_sha256!==selected.source_replay_sha256)||
         value.fps!==fps()||!Number.isSafeInteger(value.width)||value.width<=0||
         !Number.isSafeInteger(value.height)||value.height<=0||
         value.project_id!==project||
         (isRaw?value.fusion_id!==run:value.rally_run_id!==run)||
         (isRaw&&!["rfdetr-medium","rfdetr-large","legacy","unknown"].includes(value.person_detector))||
         (isRaw?!/^[a-f0-9]{64}$/.test(value.source_fusion_sha256):false)||
         value.video!==video||value.start_frame!==start||
         !Number.isSafeInteger(value.end_frame)||value.end_frame<=start||
         (isRaw&&(!Number.isSafeInteger(value.total_frames)||
           value.total_frames<value.end_frame))||
         !Array.isArray(value.frames)||value.frames.length!==value.end_frame-start)
        throw Error("Несовместимые кадры предпросмотра");
      if(!isRaw&&!processedBallOverlayIsValid(value))
        throw Error("Processed ball source is not VballNet-only");
      if(!isRaw&&value.ball_trajectory_source==="ball_stabilizer_file_v1"&&
        value.frames.some(row=>!Object.hasOwn(row,"ball_before")||
          !Object.hasOwn(row,"ball_rejected")))
        throw Error("Отсутствует сохранённое сравнение траекторий");
      if(s.project!==project||s.video!==video||overlay.raw!==isRaw||
         overlaySourceId()!==run)return;
      if(isRaw){
        const offset=Number(value.analysis_offset_sec??0);
        if(!Number.isFinite(offset)||offset<0)throw Error("Недостоверная граница GPU-отрезка");
        const sourceKey=[project,run,video].join(":");
        overlay.rawOffsets=overlay.rawOffsets||new Map();
        const previous=overlay.rawOffsets.get(sourceKey);
        overlay.rawOffsets.set(sourceKey,offset);
        s.analysisOffset=offset;
        if(previous===undefined&&offset>0&&s.seekAt===0){
          s.seekAt=offset;
          if(ui.video.readyState>=1)ui.video.currentTime=offset;
        }
        overlay.rawFrames.set(sourceKey,value.total_frames);
        overlay.rawDetectors.set(rawDetectorKey(project,run,video),value.person_detector);
        updateRawDetectorLabel();
      }
      overlay.cache.set(key,value);
      while(overlay.cache.size>16)overlay.cache.delete(overlay.cache.keys().next().value);
      if(isRaw&&overlay.stabilizerView){
        const stats=value.ball_stabilizer;
        if(!["ball_stabilizer_v3","ball_stabilizer_v4","ball_stabilizer_v5","ball_stabilizer_v6","ball_stabilizer_v7"].includes(stats?.version)||
           value.frames.some(row=>!Object.hasOwn(row,"stabilized_ball")||
             typeof row.ball_interpolated!=="boolean"||typeof row.ball_rejected!=="boolean"))
          throw Error("Нет результата Ball Stabilizer");
      }
      overlayMessage((isRaw?overlay.rawPoseVisible.checked:overlay.flags.pose)&&
        value.poses_available===false?
          "RTMPose-M: для выбранного GPU-анализа нет сохранённых поз.":"");
      drawOverlay();
    }).catch(error=>{
      if(controller?.signal.aborted||error?.name==="AbortError")return;
      // Transient failures must not permanently disable overlays in a window.
      overlay.failed.set(key,Date.now()+5000);
      if(s.project===project&&s.video===video&&overlay.raw===isRaw&&overlaySourceId()===run)
        overlayMessage("Разметка недоступна: "+error.message);
    }).finally(()=>{
      if(overlay.controllers?.get(key)===controller){
        overlay.controllers.delete(key);
        overlay.fetching.delete(key);
      }
    });
  };
  // Rapid rally navigation should not launch two expensive replay reads per
  // click. Coalesce seeks and request the destination first; drawOverlay()
  // fetches the next window only when playback approaches its boundary.
  let overlaySeekTimer=null;
  const prefetchOverlayAtFrame=frame=>{
    if(overlaySeekTimer!==null)clearTimeout(overlaySeekTimer);
    const project=s.project,sourceId=overlaySourceId(),video=s.video,raw=overlay.raw;
    if(!project||!sourceId||!video||!Number.isSafeInteger(frame)||frame<0)return;
    overlaySeekTimer=setTimeout(()=>{
      overlaySeekTimer=null;
      if(s.project!==project||s.video!==video||overlay.raw!==raw||
         overlaySourceId()!==sourceId)return;
      const total=raw?rawFrameLimit(project,sourceId,video):
        timeline.data?.project_id===project&&
        timeline.data?.rally_run_id===sourceId?timeline.data.total_frames:null;
      const start=overlayWindowStart(frame);
      if(!Number.isSafeInteger(total)||start<total)
        requestOverlayWindow(project,sourceId,video,start);
    },120);
  };
  const cpuTouchStyle={
    attack:["#ff6b6b","≈ Атака"],
    block:["#c38cff","≈ Блок"],
    serve:["#f29d49","≈ Подача"],
    overhead_pass:["#55d6ff","≈ Передача сверху"],
    dig:["#71e49e","≈ Приём снизу"],
    unknown_touch:["#c6bf96","Касание"]
  };
  const cpuTouchKind=kind=>Object.hasOwn(cpuTouchStyle,kind)?kind:"unknown_touch";
  const floorContactColor="#e8b84e";
  const floorContactLabel=outcome=>"Касание пола · "+({
    line_uncertain:"Граница площадки",out:"За пределами площадки",
    near_half:"Ближняя половина",far_half:"Дальняя половина"
  }[outcome]||"Зона не определена");
  const floorContacts=data=>(data?.surface_contact_candidates||[]).filter(event=>event.surface==="floor");
  const validSurfaceContacts=raw=>raw.surface_contact_candidates===undefined||
    (Array.isArray(raw.surface_contact_candidates)&&raw.surface_contact_candidates.every((event,index,all)=>
      Number.isSafeInteger(event.frame_index)&&event.frame_index>=0&&event.frame_index<raw.total_frames&&
      (index===0||all[index-1].frame_index<=event.frame_index)&&event.provisional===true&&
      (event.surface==="net"&&event.outcome==="possible_deflection"||
        event.surface==="floor"&&["near_half","far_half","out","line_uncertain"].includes(event.outcome))&&
      (event.position==null||Array.isArray(event.position)&&event.position.length===2&&
        event.position.every(v=>Number.isFinite(v)&&v>=-0.5&&v<=1.5))));
  // The timeline belongs to the selected saved CPU postprocessing snapshot.
  // Raw GPU detections and limited rally-card motion_events are never Touchs.
  const cpuTouchesForFrame=(project,run,video,sha,frame)=>{
    const data=timeline.data;
    if(!data||data.schema!=="volley-native-rally-timeline/1"||
       data.project_id!==project||data.rally_run_id!==run||
       data.video!==video||data.source_replay_sha256!==sha||
       !Array.isArray(data.touch_annotations))return [];
    const events=data.touch_annotations;
    let left=0,right=events.length;
    while(left<right){
      const mid=(left+right)>>>1;
      if(events[mid].frame_index<frame-8)left=mid+1;else right=mid;
    }
    const near=[];
    for(let i=left;i<events.length&&events[i].frame_index<=frame+8;i++)
      near.push(events[i]);
    return near;
  };
  const drawCpuTouches=(ctx,record,project,run,video,frame,x,y,width,height)=>{
    if(typeof training!=="undefined"&&training.active||!overlay.flags.touches||!record.width||!record.height)return;
    const events=cpuTouchesForFrame(project,run,video,record.replay_sha256,frame);
    for(const event of events){
      const kind=cpuTouchKind(event.kind);
      if(!overlay.actionFlags[kind])continue;
      if(!Array.isArray(event.ball_position_px)||event.ball_position_px.length!==2)continue;
      const [bx,by]=event.ball_position_px;
      if(!Number.isFinite(bx)||!Number.isFinite(by)||
         bx<0||by<0||bx>record.width||by>record.height)continue;
      const [px,py]=[x+bx/record.width*width,y+by/record.height*height];
      const [color,label]=cpuTouchStyle[kind];
      const contact=event.contact_landmark;
      if(contact&&Array.isArray(contact.point_px)&&contact.point_px.length===2){
        const [lx,ly]=contact.point_px;
        if(Number.isFinite(lx)&&Number.isFinite(ly)&&lx>=0&&ly>=0&&
            lx<=record.width&&ly<=record.height){
          const cx=x+lx/record.width*width,cy=y+ly/record.height*height;
          ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(cx,cy);
          ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();
          ctx.beginPath();ctx.arc(cx,cy,4,0,Math.PI*2);
          ctx.fillStyle=color;ctx.fill();
        }
      }
      ctx.beginPath();ctx.arc(px,py,kind!=="unknown_touch"?12:8,0,2*Math.PI);
      ctx.strokeStyle=color;ctx.lineWidth=kind!=="unknown_touch"?3:1.7;ctx.stroke();
      ctx.font="bold 12px Inter, sans-serif";
      ctx.textBaseline="bottom";
      const labelsKey=playerLabelKey(project,run,video,record.replay_sha256);
      const conflict=record.frames?.find(row=>row.frame===frame)?.players?.some(player=>
        player.id===event.provisional_track_id&&player.identity_conflict===true);
      const owner=!conflict&&overlayLabels.loaded&&overlayLabels.key===labelsKey?
        overlayLabels.names.get(String(event.provisional_track_id)):null;
      const touchLabel=label+" · "+(owner||String(event.provisional_track_id??"не определён"));
      const textWidth=ctx.measureText(touchLabel).width;
      const tx=Math.min(x+width-textWidth-9,Math.max(x+5,px+15));
      const ty=Math.max(y+17,Math.min(y+height-5,py-6));
      ctx.fillStyle="rgba(9,20,35,.94)";
      ctx.fillRect(tx-4,ty-14,textWidth+8,17);
      ctx.fillStyle=color;ctx.fillText(touchLabel,tx,ty);
    }
    const data=timeline.data;
    if(data?.project_id!==project||data?.rally_run_id!==run||data?.video!==video||
       data?.source_replay_sha256!==record.replay_sha256)return;
    const surface=data.surface_contact_candidates||[];
    let left=0,right=surface.length;
    while(left<right){
      const mid=(left+right)>>>1;
      if(surface[mid].frame_index<frame-8)left=mid+1;else right=mid;
    }
    for(let i=left;i<surface.length&&surface[i].frame_index<=frame+8;i++){
      const event=surface[i];
      if(event.surface!=="floor"||!Array.isArray(event.position))continue;
      const [bx,by]=event.position;
      if(bx<0||bx>1||by<0||by>1)continue;
      const px=x+bx*width,py=y+by*height;
      ctx.beginPath();ctx.arc(px,py,10,0,2*Math.PI);
      ctx.strokeStyle=floorContactColor;ctx.lineWidth=2.5;ctx.stroke();
      ctx.font="bold 12px Inter, sans-serif";ctx.textBaseline="bottom";
      const label=floorContactLabel(event.outcome);
      const tw=ctx.measureText(label).width;
      const tx=Math.max(x+5,Math.min(x+width-tw-9,px+15));
      const ty=Math.max(y+17,Math.min(y+height-5,py-6));
      ctx.fillStyle="rgba(9,20,35,.94)";ctx.fillRect(tx-4,ty-14,tw+8,17);
      ctx.fillStyle=floorContactColor;ctx.fillText(label,tx,ty);
    }
  };
  const courtPreview={key:"",data:null,loading:false,failed:false};
  const drawCourtPreview=box=>{
    if(!overlay.flags.court||!s.project||!s.video)return;
    const key=s.project+":"+s.video;
    if(courtPreview.key!==key){
      courtPreview.key=key;courtPreview.data=null;
      courtPreview.loading=false;courtPreview.failed=false;
    }
    if(!courtPreview.data&&!courtPreview.loading&&!courtPreview.failed){
      courtPreview.loading=true;
      request(url("/api/video-edit",{project_id:s.project,video:s.video})).then(data=>{
        if(courtPreview.key!==key)return;
        courtPreview.data=data;
        drawOverlay();
      }).catch(()=>{
        if(courtPreview.key===key)courtPreview.failed=true;
      }).finally(()=>{if(courtPreview.key===key)courtPreview.loading=false;});
    }
    const data=courtPreview.data;
    if(!data)return;
    const {ctx,x,y,width,height}=box;
    const position=p=>[x+p[0]*width,y+p[1]*height];
    if(Array.isArray(data.court)&&data.court.length===4){
      ctx.beginPath();
      data.court.forEach((p,i)=>{const [px,py]=position(p);if(i)ctx.lineTo(px,py);else ctx.moveTo(px,py);});
      ctx.closePath();ctx.strokeStyle="#ffd166";ctx.lineWidth=2.25;ctx.stroke();
    }
    if(Array.isArray(data.net)&&data.net.length===2){
      const [ax,ay]=position(data.net[0]),[bx,by]=position(data.net[1]);
      ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);
      ctx.strokeStyle="#35e0ef";ctx.lineWidth=2.5;ctx.stroke();
    }
  };
  // Processed skeletons have the same rally scope as server-filtered player boxes.
  const poseVisibleInRally=frame=>frame?.in_rally===true&&Array.isArray(frame.pose_points);
  const trainingTouchCandidateAtFrame=frame=>{
    if(!showBallCandidates()||!annotationSourceValid()||!annotation.prediction)return null;
    return annotation.prediction.proposals.find(item=>Math.abs(item.frame-frame)<=4)||null;
  };
  const drawTrainingTouchCandidate=(ctx,record,frame,x,y,width,height)=>{
    if(!showBallCandidates()||!annotationSourceValid()||!annotation.prediction||
       record?.replay_sha256&&annotation.prediction.source_replay_sha256&&
       record.replay_sha256!==annotation.prediction.source_replay_sha256)return;
    const ground=annotation.prediction.ground_contacts?.find(item=>Math.abs(item.frame-frame)<=3);
    if(ground){
      const [color,half]=groundContactAppearance[ground.landing]||["#ffd166","неизвестно"];
      const px=x+ground.position[0]*width,py=y+ground.position[1]*height;
      const label="Удар о песок · "+half;
      ctx.save();ctx.lineWidth=3;ctx.strokeStyle=color;ctx.fillStyle=color;
      ctx.beginPath();ctx.arc(px,py,18,0,Math.PI*2);ctx.stroke();
      ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fill();
      ctx.font="bold 13px Inter, sans-serif";ctx.textBaseline="top";
      const tw=ctx.measureText(label).width;
      const tx=Math.max(x+8,Math.min(x+width-tw-14,px+20));
      const ty=Math.max(y+6,Math.min(y+height-23,py-20));
      ctx.fillStyle="rgba(9,20,35,.94)";ctx.fillRect(tx-5,ty-3,tw+10,20);
      ctx.fillStyle=color;ctx.fillText(label,tx,ty);ctx.restore();
      return;
    }
    const candidate=trainingTouchCandidateAtFrame(frame);
    if(!candidate)return;
    const rows=record?.frames||[];
    const detected=rows.find(row=>row.frame===candidate.frame&&Array.isArray(row.ball))||
      rows.find(row=>Math.abs(row.frame-candidate.frame)<=2&&Array.isArray(row.ball));
    const ball=detected?.ball?.length===2&&detected.ball.every(Number.isFinite)?detected.ball:null;
    const px=ball?x+ball[0]*width:x+18;
    const py=ball?y+ball[1]*height:y+24;
    const classified=annotation.typePrediction?.touches.find(item=>item.frame===candidate.frame);
    const label=showBallTouchTypes()?
      (classified?.kind?(trainingNames[classified.kind]||classified.kind):"Касание · тип не определён"):
      "Касание? · по мячу";
    ctx.save();
    if(ball){
      ctx.strokeStyle="#ffbd55";ctx.fillStyle="#ffbd55";ctx.lineWidth=3;
      ctx.beginPath();ctx.arc(px,py,16,0,Math.PI*2);ctx.stroke();
      ctx.beginPath();ctx.arc(px,py,3.5,0,Math.PI*2);ctx.fill();
    }
    ctx.font="bold 13px Inter, sans-serif";ctx.textBaseline="top";
    const message=ball?label:label+" · мяч не виден";
    const textWidth=ctx.measureText(message).width;
    const tx=ball?Math.max(x+5,Math.min(x+width-textWidth-12,px+20)):x+8;
    const ty=ball?Math.max(y+6,Math.min(y+height-22,py-18)):y+8;
    ctx.fillStyle="rgba(9,20,35,.94)";ctx.fillRect(tx-5,ty-3,textWidth+10,20);
    ctx.fillStyle="#ffdb8f";ctx.fillText(message,tx,ty);
    ctx.restore();
  };
  const drawOverlay=()=>{
    const box=overlayContext(),source=overlay.source;
    if(box)drawCourtPreview(box);
    if(!box||!source||(!overlay.raw&&!s.run)||!s.video||!ui.video.src||ui.video.seeking||
       s.videoKey!==s.project+":"+s.video||
       (ui.video.readyState!==undefined&&ui.video.readyState<2))return;
    // Until the next decoded frame arrives, keep the canvas transparent.
    if(!ui.video.paused&&typeof ui.video.requestVideoFrameCallback==="function"&&
       !Number.isFinite(overlay.presentedTime))return;
    if(!overlay.raw&&!overlay.flags.touches&&!overlay.flags.trajectory&&!overlay.flags.players&&
       !overlay.flags.pose&&!showBallCandidates())return;
    // HTMLVideoElement.currentTime can advance before a frame is decoded.
    // requestVideoFrameCallback.mediaTime is the actual frame on screen.
    const shown=Number.isFinite(overlay.presentedTime)?
      overlay.presentedTime:ui.video.currentTime;
    const localTime=shown-(Number(s.analysisOffset)||0);
    if(localTime<0)return;
    const frame=Math.max(0,Math.floor(localTime*fps()+0.001));
    const project=s.project,run=overlaySourceId(),video=s.video;
    if(!run){if(overlay.raw)overlayMessage("Для просмотра сырых данных выберите объединённый GPU-анализ (fusion).");return;}
    const start=overlayWindowStart(frame);
    const key=overlayWindowKey(project,run,video,start);
    const active=overlay.cache.get(key);
    const record=active&&active.start_frame<=frame&&frame<active.end_frame?
      active:cachedOverlayForFrame(project,run,video,frame);
    const total=overlay.raw?rawFrameLimit(project,run,video):timeline.data?.total_frames;
    if(!active&&(!Number.isSafeInteger(total)||start<total))
      requestOverlayWindow(project,run,video,start);
    // Prefetch before a 90-frame boundary; do not request past video end.
    const nextBoundary=(Math.floor(frame/90)+1)*90;
    if(active&&frame>=nextBoundary-45){
      const nextStart=overlayWindowStart(nextBoundary);
      if(nextStart!==start&&(!Number.isSafeInteger(total)||nextStart<total))
        requestOverlayWindow(project,run,video,nextStart);
    }
    if(!record){
      // Saved CPU Touch positions already belong to the validated timeline.
      // Don't make them wait for the much larger, SHA-indexed player replay.
      const index=timeline.data;
      if(!overlay.raw&&index?.project_id===project&&index?.rally_run_id===run&&
         index?.video===video&&typeof index.source_replay_sha256==="string"){
        const {ctx,x,y,width,height}=box;
        if(typeof training!=="undefined"&&training.active)drawTrainingTouchCandidate(ctx,{
          replay_sha256:index.source_replay_sha256
        },frame,x,y,width,height);
        else drawCpuTouches(ctx,{
          width:ui.video.videoWidth,height:ui.video.videoHeight,
          replay_sha256:index.source_replay_sha256
        },project,run,video,frame,x,y,width,height);
      }
      return;
    }
    // The timeline and window must describe the same immutable CPU replay.
    const digest=!overlay.raw&&timeline.data?.project_id===project&&
      timeline.data?.rally_run_id===run&&timeline.data?.video===video?
      timeline.data.source_replay_sha256:null;
    if(digest&&record.replay_sha256!==digest){
      overlayMessage("Наложение не соответствует выбранной CPU-версии.");
      return;
    }
    if(record.width&&record.height&&ui.video.videoWidth&&ui.video.videoHeight&&
      Math.abs((ui.video.videoWidth/ui.video.videoHeight)/
        (record.width/record.height)-1)>0.025){
      overlayMessage("Размеры видео и наложения несовместимы.");
      return;
    }
    if(!overlay.raw&&overlay.flags.players)
      requestOverlayLabels(project,run,video,record.replay_sha256);
    const {ctx,x,y,width,height}=box;
    const point=ball=>[x+ball[0]*width,y+ball[1]*height];
    const frames=record.frames;
    const debugTrajectory=typeof training!=="undefined"&&training.active;
    const canonicalTrajectory=record.ball_trajectory_source==="ball_stabilizer_file_v1";
    const showBefore=debugTrajectory&&overlay.trajectoryVariant==="before";
    const displayedBall=row=>showBefore?row.ball_before:
      (debugTrajectory&&!canonicalTrajectory?null:row.ball);
    if(overlay.raw){
      if(overlay.stabilizerView||(overlay.rawVballVisible.checked&&overlay.rawBallTrail?.checked)){
        const threshold=overlay.stabilizerView?0:Number(overlay.rawBall.value)/100;
        let previous=null;
        ctx.strokeStyle="#47e2ff";ctx.fillStyle="#47e2ff";ctx.lineWidth=2;
        for(const row of frames){
          if(row.frame<frame-(overlay.stabilizerView?44:Math.round(fps()))||row.frame>frame)continue;
          const balls=(row.raw?.balls||[]).filter(ball=>ball.source==="vballnet"&&
            Number.isFinite(ball.score)&&ball.score>=threshold&&
            Array.isArray(ball.point)&&ball.point.length===2&&ball.point.every(Number.isFinite));
          for(const ball of balls){
            const [px,py]=point(ball.point);
            ctx.globalAlpha=0.25+0.6*(1-(frame-row.frame)/Math.max(1,fps()));
            if(!overlay.stabilizerView&&previous&&row.frame===previous.frame+1&&balls.length===1){
              ctx.beginPath();ctx.moveTo(previous.x,previous.y);ctx.lineTo(px,py);ctx.stroke();
            }
            ctx.beginPath();ctx.arc(px,py,2.5,0,Math.PI*2);ctx.fill();
          }
          previous=balls.length===1?{frame:row.frame,x:point(balls[0].point)[0],y:point(balls[0].point)[1]}:null;
        }
        ctx.globalAlpha=1;
      }
      const current=frames.find(row=>row.frame===frame);
      if(!current?.raw)return;
      const raw=current.raw;
      const enough=(score,input)=>Number.isFinite(score)&&score>=Number(input.value)/100;
      ctx.font="bold 11px Inter, sans-serif";ctx.textBaseline="top";ctx.lineWidth=1.8;
      for(const ball of overlay.stabilizerView?[]:(raw.balls||[])){
        if(ball.source!=="vballnet"||!overlay.rawVballVisible.checked||
          !enough(ball.score,overlay.rawBall)||!Array.isArray(ball.point))continue;
        const [px,py]=point(ball.point);
        ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);
        ctx.fillStyle="#47e2ff";ctx.fill();
        ctx.fillText(Math.round(ball.score*100)+"%",px+7,py-6);
      }
      if(overlay.stabilizerView){
        ctx.save();
        ctx.lineWidth=2.5;
        ctx.strokeStyle="#f1bd51";ctx.fillStyle="#f1bd51";
        let previous=null;
        for(const item of frames){
          if(item.frame<frame-44||item.frame>frame||!Array.isArray(item.stabilized_ball)){
            previous=null;
            continue;
          }
          const [sx,sy]=point(item.stabilized_ball);
          const synthetic=item.ball_interpolated===true;
          if(previous&&item.frame-previous.frame<=2){
            ctx.setLineDash(synthetic||previous.synthetic?[5,4]:[]);
            ctx.beginPath();ctx.moveTo(previous.x,previous.y);ctx.lineTo(sx,sy);ctx.stroke();
          }
          if(item.frame===frame){
            ctx.setLineDash(synthetic?[3,3]:[]);
            ctx.beginPath();ctx.arc(sx,sy,6,0,2*Math.PI);ctx.stroke();
            if(!synthetic){ctx.beginPath();ctx.arc(sx,sy,3,0,2*Math.PI);ctx.fill();}
          }
          previous={x:sx,y:sy,frame:item.frame,synthetic};
        }
        ctx.setLineDash([]);
        ctx.restore();
      }
      if(!overlay.stabilizerView&&overlay.rawPeopleVisible.checked){
        for(const player of raw.players||[]){
          if(!Array.isArray(player.box)||player.box.length!==4)continue;
          const [x1,y1,x2,y2]=player.box,px=x+x1*width,py=y+y1*height;
          if(enough(player.score,overlay.rawPlayer)){
            ctx.strokeStyle="#71acff";ctx.strokeRect(px,py,(x2-x1)*width,(y2-y1)*height);
            ctx.fillStyle="#d9eaff";ctx.fillText(Math.round(player.score*100)+"%",px+2,py+2);
          }
        }
        if(overlay.rawPlayerInterpolated?.checked &&
          ["rfdetr-medium","rfdetr-large"].includes(overlay.rawDetectors.get(rawDetectorKey(project,run,video)))){
          ctx.save();
          ctx.strokeStyle="#ffbc69";ctx.fillStyle="#ffe0b0";ctx.lineWidth=1.8;
          ctx.setLineDash([6,4]);
          for(const person of raw.players_interpolated||[]){
            if(!Array.isArray(person.box)||person.box.length!==4||
              !person.box.every(Number.isFinite))continue;
            const [x1,y1,x2,y2]=person.box;
            if(x1>=x2||y1>=y2)continue;
            const px=x+x1*width,py=y+y1*height;
            ctx.strokeRect(px,py,(x2-x1)*width,(y2-y1)*height);
            ctx.fillText("Интерп.",px+2,py+2);
          }
          ctx.restore();
        }
        const poseBoxes=raw.poses?.length?raw.poses:(raw.poses_interpolated||[]);
        if(Array.isArray(poseBoxes)&&poseBoxes.length){
          ctx.save();ctx.setLineDash([5,4]);ctx.strokeStyle="#98a9d3";ctx.lineWidth=1.5;
          for(const person of poseBoxes){
            if(!Array.isArray(person.box)||person.box.length!==4)continue;
            const [x1,y1,x2,y2]=person.box;
            const covered=(raw.players||[]).some(det=>{
              if(!Array.isArray(det.box)||det.box.length!==4)return false;
              const b=det.box;
              const intersection=Math.max(0,Math.min(x2,b[2])-Math.max(x1,b[0]))*
                Math.max(0,Math.min(y2,b[3])-Math.max(y1,b[1]));
              const size=(x2-x1)*(y2-y1)+(b[2]-b[0])*(b[3]-b[1])-intersection;
              return size>0&&intersection/size>0.4;
            });
            if(!covered)ctx.strokeRect(x+x1*width,y+y1*height,(x2-x1)*width,(y2-y1)*height);
          }
          ctx.restore();
        }
      }
      if(!overlay.stabilizerView&&overlay.rawPoseVisible.checked){
        const edges=[[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],
          [11,12],[11,13],[13,15],[12,14],[14,16]];
        const threshold=Number(overlay.rawPose.value)/100;
        // The original RTMPose observations remain in raw.poses. Only draw
        // visual-only interpolation when no observation exists on this frame.
        const interpolated=!raw.poses?.length && Array.isArray(raw.poses_interpolated);
        const poseRows=interpolated?raw.poses_interpolated:(raw.poses||[]);
        ctx.globalAlpha=interpolated?0.78:1;
        for(const person of poseRows){
          const pose=person.keypoints;
          if(!Array.isArray(pose)||pose.length!==17)continue;
          const valid=i=>Array.isArray(pose[i])&&pose[i].length===3&&
            pose[i].every(Number.isFinite)&&pose[i][2]>=threshold;
          ctx.strokeStyle="#6af6ae";ctx.fillStyle="#f9d977";ctx.lineWidth=2;
          for(const [a,b] of edges){
            if(!valid(a)||!valid(b))continue;
            const [ax,ay]=point(pose[a]),[bx,by]=point(pose[b]);
            ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);ctx.stroke();
          }
          for(let i=0;i<17;i++)if(valid(i)){
            const [px,py]=point(pose[i]);
            ctx.beginPath();ctx.arc(px,py,2.5,0,Math.PI*2);ctx.fill();
          }
        }
        ctx.globalAlpha=1;
      }
      return;
    }
    if(overlay.flags.trajectory&&!overlay.raw){
      ctx.save();
      ctx.strokeStyle=debugTrajectory&&!showBefore?"#f1bd51":"#47e2ff";
      ctx.lineWidth=2.5;ctx.lineJoin="round";
      let previous=null;
      for(const row of frames){
        if(row.frame<frame-44||row.frame>frame)continue;
        const sample=displayedBall(row);
        if(!Array.isArray(sample)){previous=null;continue;}
        const [px,py]=point(sample);
        if(previous&&row.frame-previous.frame<=2){
          ctx.setLineDash(debugTrajectory&&!showBefore&&
            (row.ball_interpolated||previous.interpolated)?[5,4]:[]);
          ctx.beginPath();ctx.moveTo(previous.x,previous.y);ctx.lineTo(px,py);ctx.stroke();
        }
        previous={frame:row.frame,x:px,y:py,interpolated:row.ball_interpolated===true};
      }
      ctx.restore();
    }
    const current=frames.find(row=>row.frame===frame);
    // Match the CPU player's verified rally scope; do not draw skeletons between plays.
    if(overlay.flags.pose && poseVisibleInRally(current)){
      const edges=[[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],
        [11,12],[11,13],[13,15],[12,14],[14,16]];
      ctx.strokeStyle="#6af6ae";ctx.fillStyle="#f9d977";
      ctx.lineWidth=2.25;ctx.lineCap="round";
      ctx.globalAlpha=current.pose_points_interpolated===true?0.78:1;
      // Directly visualize RTMPose evidence. A CPU Track ID or a matching
      // bounding box must never be required to display measured joints.
      for(const person of current.pose_points){
        const pose=person.keypoints;
        if(!Array.isArray(pose)||pose.length!==17)continue;
        const valid=i=>Array.isArray(pose[i])&&pose[i].length===3&&
          pose[i].every(Number.isFinite)&&pose[i][2]>=0.3;
        for(const [a,b] of edges){
          if(!valid(a)||!valid(b))continue;
          const [ax,ay]=point(pose[a]),[bx,by]=point(pose[b]);
          ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);ctx.stroke();
        }
        for(let i=0;i<pose.length;i++){
          if(!valid(i))continue;
          const [px,py]=point(pose[i]);
          ctx.beginPath();ctx.arc(px,py,i===9||i===10?3.5:2.5,0,2*Math.PI);ctx.fill();
        }
      }
      ctx.globalAlpha=1;
    }
    // Only CPU-processed mode may display verified timeline touch annotations.
    drawCpuTouches(ctx,record,project,run,video,frame,x,y,width,height);
    if(typeof training!=="undefined"&&training.active)drawTrainingTouchCandidate(ctx,record,frame,x,y,width,height);
    if(overlay.flags.players&&current){
      ctx.strokeStyle="#71acff";ctx.lineWidth=1.8;
      ctx.font="bold 11px Inter, sans-serif";ctx.textBaseline="top";
      for(const player of current.players||[]){
        if(!Array.isArray(player.box)||player.box.length!==4)continue;
        const [x1,y1,x2,y2]=player.box;
        const px=x+x1*width,py=y+y1*height;
        const rw=(x2-x1)*width,rh=(y2-y1)*height;
        ctx.save();
        ctx.strokeStyle=playerTeamColor(player.team_side,overlay.flags.teams);
        if(player.interpolated===true){ctx.setLineDash([5,4]);ctx.globalAlpha=0.78;}
        if(Array.isArray(player.visible_regions)){ctx.setLineDash([5,4]);ctx.globalAlpha=0.78;}
        ctx.strokeRect(px,py,rw,rh);
        if(Array.isArray(player.visible_regions)&&player.visible_regions.length){
          ctx.beginPath();
          for(const [a,b,c,d] of player.visible_regions){
            ctx.rect(x+a*width-1,y+b*height-1,(c-a)*width+2,(d-b)*height+2);
          }
          ctx.clip();ctx.setLineDash([]);ctx.globalAlpha=1;
          ctx.strokeRect(px,py,rw,rh);
        }
        ctx.restore();
        const labelsKey=playerLabelKey(project,run,video,record.replay_sha256);
        const owner=!player.identity_conflict&&overlayLabels.loaded&&overlayLabels.key===labelsKey?
          overlayLabels.names.get(String(player.id)):null;
        // Show the linked player's name, or the numeric ID when unassigned.
        const name=owner||String(player.id);
        const tw=Math.min(220,ctx.measureText(name).width+10);
        ctx.fillStyle="rgba(12,47,91,.88)";
        ctx.fillRect(px,py+rh,tw,16);
        ctx.fillStyle="#f1f8ff";ctx.fillText(name,px+5,py+rh+1,tw-10);
      }
    }
    if(current&&Array.isArray(displayedBall(current))&&overlay.flags.trajectory){
      const [px,py]=point(displayedBall(current));
      ctx.save();
      if(debugTrajectory&&!showBefore&&current.ball_interpolated===true)ctx.setLineDash([4,3]);
      ctx.beginPath();ctx.arc(px,py,5,0,2*Math.PI);
      ctx.fillStyle=debugTrajectory&&!showBefore?"#f1bd51":"#e7fbff";
      ctx.fill();
      ctx.strokeStyle=debugTrajectory&&!showBefore?"#f1bd51":"#21b4fc";
      ctx.lineWidth=2;ctx.stroke();
      ctx.restore();
    }
  };
  const setOverlaySource=item=>{
    overlay.source=item?{events:[]}:null;
    overlayMessage("");
    if(!item){
      invalidateOverlayLabels();
      for(const controller of overlay.controllers.values())controller?.abort();
      overlay.controllers.clear();overlay.fetching.clear();
      overlay.cache.clear();overlay.rawFrames.clear();overlay.failed.clear();overlay.presentedTime=null;
    }
    drawOverlay();
  };
  window.addEventListener("volley:court-calibration-updated",event=>{
    if(s.project===event.detail?.project_id&&s.video===event.detail?.video){
      courtPreview.data=null;courtPreview.failed=false;
      for(const controller of overlay.controllers.values())controller?.abort();
      overlay.controllers.clear();overlay.fetching.clear();overlay.cache.clear();overlay.failed.clear();
      drawOverlay();
    }
  });  window.addEventListener("volley:track-assignments-updated",event=>{
    const detail=event.detail||{};
    if(detail.project_id===s.project&&(!detail.video||detail.video===s.video)){
      invalidateOverlayLabels();
      overlay.cache.clear();
      drawOverlay();
    }
  });
  // The player gallery is refreshed on entry, including server-side automatic
  // links produced by CPU postprocessing. Refresh the preview's name snapshot
  // on re-entry as well; the cached frame coordinates remain SHA-bound.
  window.addEventListener("volley:tab-change",event=>{
    if(event.detail?.tab==="preview"&&s.project&&s.video)
      ensurePreviewVideo();
    if(event.detail?.tab==="preview"&&s.project&&s.run){
      invalidateOverlayLabels();
      drawOverlay();
    }
  });
  for(const eventName of ["volley:player-library-updated","volley:roster-updated"]){
    window.addEventListener(eventName,()=>{
      invalidateOverlayLabels();
      drawOverlay();
    });
  }
  for(const key of ["rallies","touches","trajectory","players","pose","court","teams"]){
    overlay[key].addEventListener("click",()=>{
      overlay.flags[key]=!overlay.flags[key];
      if(key==="teams"&&overlay.flags.teams)overlay.flags.players=true;
      overlay.actionControls.classList.toggle("hidden",overlay.raw||!overlay.flags.touches);
      updateOverlayToggles();
      try{localStorage.setItem("volley:preview-overlays:v2",JSON.stringify(overlay.flags));}catch{}
      drawOverlay();
      drawTimeline();
      if(key==="touches")refreshRallyCardTouchTicks();
    });
  }
  // Manual examples are audited in Rust; the browser only selects the saved
  // CPU frame and player, never sends arbitrary pose coordinates.
  const training={
    panel:$("studio-training-panel"),poseAdvanced:$("studio-training-pose-advanced"),
    status:$("studio-training-status"),label:$("studio-training-label"),
    save:$("studio-training-save"),newExample:$("studio-training-new"),
    refresh:$("studio-training-refresh"),
    list:$("studio-training-list"),count:$("studio-training-count"),
    active:false,track:null,frame:null,busy:false,viewBusy:false,
    revision:0,epoch:0,selectedId:null,editId:null,previousRallyFilter:null
  };
  const trainingMessage=(text,error=false)=>{
    training.status.textContent=text;
    training.status.classList.toggle("is-error",error);
  };
  const trainingNames={attack:"Атака",block:"Блок",serve:"Подача",
    overhead_pass:"Передача сверху",dig:"Приём снизу",low_touch:"Касание"};
  const manualTouchKinds={serve:"Подача",attack:"Атака",block:"Блок",
    overhead_pass:"Передача сверху",dig:"Приём снизу"};
  const trainingEditorReset=()=>{
    training.editId=null;
    training.save.textContent="Сохранить пример";
    training.newExample.classList.add("hidden");
  };
  const openTrainingExample=async item=>{
    if(training.viewBusy)return;
    training.viewBusy=true;
    window.VolleyballTasks?.set("training-open",{label:"Дебаг",stage:"Открытие примера",progress:null});
    trainingMessage("Открываем исходный кадр обучающего примера…");
    try{
      if(s.project!==item.project_id)await projects(item.project_id);
      if(s.project!==item.project_id)
        throw Error("Проект примера недоступен в каталоге");
      if(item.type==="touch"){
        const source=s.chain?.runToAnalysis.get(item.rally_run_id);
        if(!source)throw Error("CPU-версия примера больше не доступна");
        if(s.run!==item.rally_run_id){
          ui.analysis.value=source;
          await setAnalysis(item.rally_run_id);
        }
        if(s.run!==item.rally_run_id)
          throw Error("Не удалось открыть CPU-версию примера");
      }else if(s.video!==item.video){
        const choice=s.chain?.choices.find(row=>row.video===item.video&&row.kind==="fusion");
        if(!choice)throw Error("Для камеры примера нет доступной GPU-версии");
        ui.analysis.value=choice.value;
        await setAnalysis();
      }
      if(s.video!==item.video)throw Error("Исходное видео примера не совпало");
      const second=item.frame_index/fps();
      s.seekAt=second;
      ensurePreviewVideo();
      await loadTimeline();
      if(s.project!==item.project_id||s.video!==item.video)throw Error("Источник изменился");
      ui.video.pause();
      if(timeline.data)seekTimeline(second);
      else if(ui.video.readyState>=1)ui.video.currentTime=second;
      prefetchOverlayAtFrame(item.frame_index);
      training.selectedId=item.id;
      training.track=null;training.frame=null;
      trainingEditorReset();
      if(item.type==="touch"){
        training.editId=item.id;
        training.label.value=item.label;
        training.save.textContent="Перезаписать кадр и игрока";
        training.newExample.classList.remove("hidden");
      }
      for(const row of training.list.querySelectorAll(".studio-training-example"))
        row.classList.toggle("selected",row.dataset.exampleId===item.id);
      trainingMessage(item.type==="touch"?
        "Открыт пример ID "+item.track_id+" · кадр "+item.frame_index+
        ". Тип можно изменить в списке; для замены кадра выберите игрока на паузе и нажмите «Перезаписать».":
        "Открыт сохранённый кадр разметки поля. Геометрию площадки изменяйте в редакторе анализа.");
    }catch(error){trainingMessage("Не удалось открыть пример: "+error.message,true);}
    finally{training.viewBusy=false;window.VolleyballTasks?.clear("training-open");}
  };
  const refreshTraining=async()=>{
    const epoch=++training.epoch;
    if(!s.project){training.list.textContent="Сначала выберите проект.";return;}
    training.list.textContent="Загрузка обучающих примеров…";
    try{
      const data=await request(url("/api/training/examples",{project_id:s.project}));
      if(epoch!==training.epoch)return;
      if(data.schema!=="volley-human-training-library/1"||!Array.isArray(data.examples)||
          !Number.isSafeInteger(data.revision))
        throw Error("Некорректная версия обучающей базы");
      training.revision=data.revision;
      training.list.replaceChildren();
      training.count.textContent=String(data.examples.length);
      for(const item of data.examples){
        const row=document.createElement("div");
        row.className="studio-training-example"+(item.id===training.selectedId?" selected":"");
        row.dataset.exampleId=item.id;
        const label=document.createElement("span");
        label.textContent=(item.type==="court"?"Поле и сетка":trainingNames[item.label]||item.label)+
          " · "+item.video+" · "+(Number(item.frame_index)/30).toFixed(2)+" с"+
          (item.type==="touch"?" · ID "+item.track_id:"")+
          (item.project_id===s.project?"":" · другой проект");
        label.title=item.project_id;
        const actions=document.createElement("div");actions.className="studio-training-example-actions";
        const open=document.createElement("button");
        open.type="button";open.className="secondary";open.textContent="Просмотреть";
        open.addEventListener("click",()=>void openTrainingExample(item));
        actions.append(open);
        if(item.type==="touch"){
          const change=document.createElement("button");
          change.type="button";change.className="secondary";change.textContent="Изменить";
          const editor=document.createElement("div");
          editor.className="studio-training-example-editor hidden";
          const select=document.createElement("select");
          select.setAttribute("aria-label","Исправленный тип касания");
          for(const [value,title] of Object.entries(trainingNames)){
            const option=document.createElement("option");
            option.value=value;option.textContent=title;select.append(option);
          }
          select.value=item.label;
          const accept=document.createElement("button");
          accept.type="button";accept.className="primary";accept.textContent="Сохранить тип";
          accept.addEventListener("click",async()=>{
            accept.disabled=true;
            window.VolleyballTasks?.set("training-relabel",{label:"Дебаг",stage:"Изменение типа касания",progress:null});
            try{
              await post("/api/training/examples",{
                action:"relabel",project_id:item.project_id,
                id:item.id,label:select.value,revision:training.revision
              });
              if(training.editId===item.id)training.label.value=select.value;
              trainingMessage("Тип касания исправлен. Новая метка применяется в следующем CPU-анализе.");
              await refreshTraining();
            }catch(error){
              trainingMessage("Не удалось изменить пример: "+error.message,true);
              await refreshTraining();
            }finally{accept.disabled=false;window.VolleyballTasks?.clear("training-relabel");}
          });
          const cancel=document.createElement("button");
          cancel.type="button";cancel.className="secondary";cancel.textContent="Отмена";
          cancel.addEventListener("click",()=>editor.classList.add("hidden"));
          editor.append(select,accept,cancel);
          change.addEventListener("click",()=>{
            for(const other of training.list.querySelectorAll(".studio-training-example-editor"))
              other.classList.add("hidden");
            editor.classList.remove("hidden");
          });
          actions.append(change);
          row.append(label,actions,editor);
        }else row.append(label,actions);
        const remove=document.createElement("button");
        remove.type="button";remove.className="secondary";remove.textContent="Удалить";
        remove.setAttribute("aria-label","Удалить обучающий пример");
        remove.addEventListener("click",async()=>{
          if(!window.confirm("Удалить этот обучающий пример?"))return;
          remove.disabled=true;
          window.VolleyballTasks?.set("training-delete",{label:"Дебаг",stage:"Удаление примера",progress:null});
          try{
            await post("/api/training/examples",{
              action:"delete",project_id:item.project_id,id:item.id
            });
            if(training.selectedId===item.id){training.selectedId=null;trainingEditorReset();}
            await refreshTraining();
            trainingMessage("Пример удалён. Следующий CPU-анализ будет использовать обновлённую базу.");
          }catch(error){trainingMessage(error.message,true);remove.disabled=false;}
          finally{window.VolleyballTasks?.clear("training-delete");}
        });
        actions.append(remove);
        training.list.append(row);
      }
      if(!data.examples.length)training.list.textContent="Пока нет примеров касаний и разметки площадки.";
    }catch(error){
      if(epoch===training.epoch)training.list.textContent="Не удалось загрузить примеры: "+error.message;
    }
  };
  // Source-scoped human truth. Separate from automatic CPU predictions and
  // from pose examples: training mode never rewrites immutable inference.
  const annotation={
    data:null,epoch:0,busy:false,dirty:false,prediction:null,typePrediction:null,source:"",priorFlags:null,touchId:null,rallyId:null,pendingStart:null,markMode:false,
    markButton:$("studio-debug-touch-mark"),saveQuick:$("studio-debug-touch-save"),
    touchLayerVisible:true,ballLayerVisible:true,
    status:$("studio-annotation-status"),current:$("studio-annotation-current"),
    lane:$("studio-timeline-track"),touchList:$("studio-annotation-touch-list"),
    rallies:$("studio-annotation-rallies"),metrics:$("studio-annotation-metrics"),
    saveState:$("studio-annotation-save-state"),
    calculate:$("studio-annotation-calculate"),classify:$("studio-annotation-classify"),adopt:$("studio-annotation-seed-trajectory"),
    save:$("studio-annotation-save"),method:$("studio-training-method"),
    hideExtra:$("studio-annotation-hide-extra"),
    pending:$("studio-annotation-rally-pending"),
    seed:$("studio-annotation-seed"),add:$("studio-annotation-add-touch"),
    move:$("studio-annotation-move-touch"),remove:$("studio-annotation-delete-touch"),
    next:$("studio-annotation-new-rally"),begin:$("studio-annotation-rally-start"),
    finish:$("studio-annotation-rally-end"),review:$("studio-annotation-review")
  };
  const reference={
    active:false,entries:[],epoch:0,
    analysis:$("studio-debug-mode-analysis"),toggle:$("studio-debug-mode-reference"),
    picker:$("studio-debug-reference-picker"),video:$("studio-debug-reference-video"),
    counts:$("studio-debug-reference-counts")
  };
  const updateReferenceModeUi=()=>{
    reference.analysis.setAttribute("aria-pressed",String(!reference.active));
    reference.toggle.setAttribute("aria-pressed",String(reference.active));
    reference.picker.classList.toggle("hidden",!reference.active);
    if(reference.active)annotation.markMode=false;
    drawAnnotation();
    $("studio-preview-view").classList.toggle("reference-workspace",reference.active);
    const titles=document.querySelector("#studio-preview-view .rallies-panel-titles");
    if(titles){
      titles.querySelector("strong").textContent=reference.active?
        "Эталонные розыгрыши":"Список розыгрышей";
      titles.querySelector("span").textContent=reference.active?
        "Проверенные розыгрыши и касания":"Тестирование алгоритмов в плеере";
    }
    $("studio-overlay-rallies").disabled=reference.active;
    hideAnnotationContext();
    drawTimeline();
    cards();
  };
  reference.analysis.addEventListener("click",()=>{
    if(!training.active||!reference.active)return;
    reference.active=false;reference.epoch++;
    reference.entries=[];
    updateReferenceModeUi();
    annotationFocus();
    updateTrainingRallyLayers();
  });
  reference.toggle.addEventListener("click",()=>{
    if(!training.active||reference.active)return;
    if(annotation.dirty){
      annotationNotice("Сначала сохраните или отмените несохранённую разметку.",true);
      return;
    }
    ui.video.pause();
    reference.active=true;
    updateReferenceModeUi();
    updateTrainingRallyLayers();
    void refreshReferenceCatalog();
  });
  reference.video.addEventListener("change",()=>{
    const entry=reference.entries.find(item=>item.rally_run_id===reference.video.value);
    if(entry)void selectReferenceEntry(entry).catch(error=>{
      reference.counts.textContent="Ошибка выбора видео: "+error.message;
    });
  });
  const annotationSourceTime=frame=>(Number(s.analysisOffset)||0)+frame/fps();
  const annotationLocalFrame=seconds=>Math.round(
    (seconds-(Number(s.analysisOffset)||0))*fps());
  const annotationFrameAtPosition=(fraction,totalFrames)=>Math.max(0,Math.min(
    totalFrames-1,annotationLocalFrame(
      Math.max(0,Math.min(1,fraction))*timelineDuration())));
  const annotationFrame=()=>Math.max(0,Math.min(
    (annotation.data?.total_frames||1)-1,
    annotationLocalFrame(Number.isFinite(ui.video.currentTime)?
      ui.video.currentTime:(Number(s.seekAt)||0))));
  const annotationClock=frame=>annotationSourceTime(frame).toFixed(3).replace(".",",")+" с";
  const annotationNotice=(value,error=false)=>{
    annotation.status.textContent=value;
    annotation.status.classList.toggle("is-error",error);
  };
  const annotationReady=()=>annotation.data&&Array.isArray(annotation.data.touches);
  const updateManualPoint=()=>{
    if(typeof overlay==="undefined")return;
    const marker=$("studio-debug-manual-point");
    if(!marker)return;
    const touch=training.active&&annotationSourceValid()&&
      (annotation.data?.touches||[]).find(item=>item.frame===annotationFrame()&&
        Array.isArray(item.position));
    const rect=overlay.canvas.getBoundingClientRect(),wrap=overlay.canvas.closest(".video-wrap");
    const w=ui.video.videoWidth,h=ui.video.videoHeight;
    if(!touch||!wrap||!w||!h||!rect.width||!rect.height){
      marker.hidden=true;return;
    }
    const scale=Math.min(rect.width/w,rect.height/h);
    const base=wrap.getBoundingClientRect();
    marker.style.left=(rect.left-base.left+(rect.width-w*scale)/2+
      touch.position[0]*w*scale)+"px";
    marker.style.top=(rect.top-base.top+(rect.height-h*scale)/2+
      touch.position[1]*h*scale)+"px";
    marker.title=trainingNames[touch.kind]||"Эталонное касание";
    marker.hidden=false;
  };
  ui.video.addEventListener?.("timeupdate",updateManualPoint);
  ui.video.addEventListener?.("seeked",updateManualPoint);
  if(typeof window!=="undefined")window.addEventListener?.("resize",updateManualPoint);
  const annotationSource=()=>[s.project,s.run,s.video].join("\u0000");
  const annotationSourceValid=()=>annotation.source===annotationSource();
  const trainingTouchCountSummary=()=>{
    const result=annotation.prediction&&annotationSourceValid()?annotation.prediction:null;
    return result?"Касания: "+result.proposals.length+" (кандидаты) · песок: "+
      (result.ground_contacts?.length||0):"";
  };
  const annotationNewId=()=>window.crypto?.randomUUID?.()||
    "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g,c=>{
      const n=Math.floor(Math.random()*16);return (c==="x"?n:(n&3|8)).toString(16);
    });
  const annotationCurrentData=()=>annotation.data&&annotationSourceValid();
  const annotationTrajectoryRallySeeds=()=>{
    const proposed=annotation.prediction;
    if(!proposed||proposed.rally_source!=="calculated_ball_trajectory_rallies"||!Array.isArray(proposed.rally_intervals))return [];
    return proposed.rally_intervals.slice(0,1024)
      .filter(r=>Number.isSafeInteger(r.start)&&Number.isSafeInteger(r.end)&&
        r.start>=0&&r.start<r.end&&r.end<(annotation.prediction?.total_frames||annotation.data?.total_frames||0))
      .map(r=>({id:annotationNewId(),start:r.start,end:r.end}));
  };
  const annotationFromPrediction=()=>({
    ...annotation.data,total_frames:annotation.prediction.total_frames,
    touches:annotation.prediction.proposals.map(p=>({id:annotationNewId(),frame:p.frame})),
    rallies:Array.isArray(annotation.data?.rallies)&&annotation.data.rallies.length?
      annotation.data.rallies.map(r=>({...r})):annotationTrajectoryRallySeeds(),
    reviewed:false,metrics:null
  });
  const annotationSavePresentation=(data,hasRun,busy)=>{
    if(busy)return {text:"Сохранение данных…",disabled:true};
    if(!hasRun)return {text:"Выберите CPU-анализ для разметки.",disabled:true};
    if(!annotationSourceValid())return {text:"Источники сменились. Вернитесь к предыдущей версии для сохранения черновика.",disabled:true};
    if(annotation.dirty)return {text:"Есть несохранённые исправления — сохраните данные касаний перед пересчётом.",disabled:true};
    if(!data||!Array.isArray(data.touches))return {text:"Нажмите «Рассчитать касания по мячу» и возьмите результат за основу.",disabled:true};
    if(!Array.isArray(data.rallies)||!data.rallies.length)return {text:"Данные сохранены. Добавьте хотя бы один розыгрыш для подтверждения видео.",disabled:true};
    if(data.reviewed)return {text:"Разметка сохранена и проверена ✓",disabled:true};
    return {text:"Разметка сохранена. Можно завершить проверку видео.",disabled:false};
  };
  const annotationSorted=type=>[...(annotation.data?.[type]||[])]
    .sort((a,b)=>(a.frame??a.start)-(b.frame??b.start));
  const annotationPosition=frame=>{
    const duration=timelineDuration();
    return 100*Math.max(0,Math.min(1,
      annotationSourceTime(frame)/Math.max(1/fps(),duration)));
  };
  const annotationMarkCurrent=()=>{
    if(!training.active)return;
    annotation.current.textContent=annotationClock(annotationFrame())+
      " · кадр "+annotationFrame();

  };
  const groundContactAppearance={
    near_half:["#34d399","ближняя половина"],
    far_half:["#60a5fa","дальняя половина"],
    line_uncertain:["#ffd166","линия · неуверенно"],
    out:["#ff6b6b","за пределами поля"]
  };
  const drawAnnotation=()=>{
    updateManualPoint();
    if(!training.active)return;
    const data=annotation.data,ready=annotationReady();
    annotation.calculate.disabled=annotation.busy||annotation.dirty||!s.run||!annotationSourceValid();
    annotation.classify.disabled=annotation.busy||annotation.dirty||!annotation.prediction||!annotationSourceValid();
    annotation.adopt.disabled=annotation.busy||!annotation.prediction||!annotationSourceValid();
    annotation.save.disabled=annotation.busy||!annotation.dirty||!ready||!annotationSourceValid();
    annotation.markButton.disabled=annotation.busy||!ready||!annotationSourceValid()||reference.active;
    annotation.markButton.setAttribute("aria-pressed",String(annotation.markMode&&!reference.active));
    annotation.markButton.textContent=annotation.markMode&&!reference.active?"✓ Выберите точку":"✎ Разметить касание";
    annotation.saveQuick.disabled=annotation.save.disabled;
    annotation.seed.disabled=true;
    annotation.seed.textContent=ready?"Сбросить метки к автоматическим":"Взять автоматические метки за основу";
    for(const key of ["add","begin","finish","next"])
      annotation[key].disabled=annotation.busy||!ready||!annotationSourceValid();
    annotation.move.disabled=annotation.busy||!ready||!annotation.touchId;
    annotation.remove.disabled=annotation.move.disabled;
    const save=annotationSavePresentation(data,Boolean(s.run),annotation.busy);
    annotation.review.textContent=data?.reviewed&&!annotation.dirty?"Проверено ✓":"Завершить проверку видео";
    annotation.review.disabled=save.disabled;
    annotation.saveState.textContent=save.text;
    annotation.pending.textContent=annotation.rallyId?
      "Выбран розыгрыш: границы меняются кнопками":
      annotation.pendingStart!==null?
      "Начало "+annotationClock(annotation.pendingStart)+" · установите конец":"";
    annotation.lane.querySelectorAll(".studio-annotation-tick,.studio-trajectory-proposal").forEach(tick=>tick.remove());
    annotation.touchList.replaceChildren();
    annotation.rallies.replaceChildren();
    if(annotation.prediction&&annotationSourceValid()){
      for(const serve of (annotation.prediction.serve_proposals||[]).slice(0,600)){
        const frame=serve.trajectory_start_frame??serve.rally_start_frame;
        const marker=document.createElement("button");
        marker.type="button";marker.className="studio-trajectory-proposal";
        marker.style.background="#f3a350";
        marker.hidden=annotation.ballLayerVisible===false;
        marker.style.left=annotationPosition(frame)+"%";
        marker.title="≈ Подача · начало траектории розыгрыша · кадр "+frame+
          (Number.isSafeInteger(serve.provisional_track_id)?
            " · Track ID "+serve.provisional_track_id:
            " · подающий пока не определён");
        marker.setAttribute("aria-label",marker.title);
        marker.addEventListener("click",event=>{
          event.stopPropagation();seekTimeline(annotationSourceTime(frame));hideAnnotationContext();
        });
        annotation.lane.append(marker);
      }
      for(const item of annotation.prediction.proposals.slice(0,600)){
        const marker=document.createElement("button");
        marker.type="button";marker.className="studio-trajectory-proposal";
        marker.hidden=annotation.ballLayerVisible===false;
        marker.style.left=annotationPosition(item.frame)+"%";
        marker.title="Кандидат касания из отдельного расчёта VBallNet · не подтверждён игроком · кадр "+
          item.frame+" · оценка "+Number(item.score).toFixed(2)+
          (annotation.typePrediction?.touches.find(v=>v.frame===item.frame)?.kind?
            " · "+trainingNames[annotation.typePrediction.touches.find(v=>v.frame===item.frame).kind]: "");
        marker.setAttribute("aria-label",marker.title);
        marker.addEventListener("click",event=>{
          event.stopPropagation();seekTimeline(annotationSourceTime(item.frame));hideAnnotationContext();
        });
        marker.addEventListener("contextmenu",event=>{
          event.preventDefault();event.stopPropagation();
          seekTimeline(annotationSourceTime(item.frame));showAnnotationContext(item.frame,event);
        });
        annotation.lane.append(marker);
      }
      for(const ground of (annotation.prediction.ground_contacts||[]).slice(0,600)){
        const marker=document.createElement("button");
        marker.type="button";
        marker.className="studio-trajectory-proposal studio-trajectory-ground";
        marker.hidden=annotation.ballLayerVisible===false;
        marker.style.left=annotationPosition(ground.frame)+"%";
        const [color,half]=groundContactAppearance[ground.landing]||["#ffd166","неизвестно"];
        marker.style.background=color;
        marker.title="Удар мяча о песок · "+half+" · кадр "+ground.frame;
        marker.setAttribute("aria-label",marker.title);
        marker.addEventListener("click",event=>{
          event.stopPropagation();seekTimeline(annotationSourceTime(ground.frame));hideAnnotationContext();
        });
        annotation.lane.append(marker);
      }
    }
    if(!ready){
      annotation.metrics.textContent=annotation.prediction?
        "По мячу: "+annotation.prediction.proposals.length+" кандидатов · порог "+
        annotation.prediction.calibration.min_score+" · сдвиг "+
        annotation.prediction.calibration.shift_frames+" кадра · ударов о песок "+
        (annotation.prediction.ground_contacts?.length||0)+". Возьмите результат для правок.":
        "Рассчитайте касания по сохранённой траектории VBallNet или возьмите CPU-метки за основу.";
      annotation.rallies.textContent="Нет размеченных розыгрышей";
      annotationMarkCurrent();return;
    }

    // Scrubbable touch lane: individual marks are pointer and keyboard editable.
    for(const item of annotationSorted("touches")){
      const tick=document.createElement("button");
      tick.type="button";tick.className="studio-annotation-tick"+
        (item.id===annotation.touchId?" selected":"");
      tick.style.left=annotationPosition(item.frame)+"%";
      tick.hidden=annotation.touchLayerVisible===false;
      tick.dataset.trainingTouchId=item.id;
      tick.title="Касание "+annotationClock(item.frame)+
        " · "+(trainingNames[item.kind]||"тип не выбран")+
        ". Перетащите или выберите и нажмите «Перенести»";
      tick.setAttribute("aria-label",tick.title);
      let pointer=null;
      tick.addEventListener("pointerdown",e=>{
        if(e.button!==0||annotation.busy||reference.active)return;
        e.preventDefault();e.stopPropagation();
        annotation.touchId=item.id;
        pointer={id:e.pointerId,initialX:e.clientX,frame:item.frame};
        tick.setPointerCapture?.(e.pointerId);
        tick.classList.add("selected");
        annotation.move.disabled=false;annotation.remove.disabled=false;
      });
      tick.addEventListener("pointermove",e=>{
        if(!pointer||pointer.id!==e.pointerId)return;
        e.stopPropagation();
        const box=annotation.lane.getBoundingClientRect();
        if(!box.width)return;
        pointer.frame=annotationFrameAtPosition(
          (e.clientX-box.left)/box.width,data.total_frames);
        tick.style.left=annotationPosition(pointer.frame)+"%";
      });
      tick.addEventListener("pointerup",e=>{
        if(!pointer||pointer.id!==e.pointerId)return;
        e.stopPropagation();
        const destination=pointer.frame;
        pointer=null;
        if(destination!==item.frame)editAnnotation("move_touch",{id:item.id,frame:destination});
        else {
          seekTimeline(annotationSourceTime(item.frame));drawAnnotation();hideAnnotationContext();
        }
      });
      tick.addEventListener("pointercancel",e=>{e.stopPropagation();pointer=null;drawAnnotation();});
      tick.addEventListener("click",e=>{
        e.stopPropagation();
        if(reference.active)seekTimeline(annotationSourceTime(item.frame));
      });
      tick.addEventListener("contextmenu",e=>{
        e.preventDefault();e.stopPropagation();
        if(reference.active)return;
        annotation.touchId=item.id;seekTimeline(annotationSourceTime(item.frame));
        showAnnotationContext(item.frame,e,item.id);
      });
      tick.addEventListener("keydown",e=>{
        if(reference.active)return;
        if(e.key==="ContextMenu"||(e.shiftKey&&e.key==="F10")){
          e.preventDefault();annotation.touchId=item.id;
          showAnnotationContext(item.frame,{},item.id);return;
        }
        if(e.key==="ArrowLeft"||e.key==="ArrowRight"){
          e.preventDefault();
          annotation.touchId=item.id;
          editAnnotation("move_touch",{id:item.id,frame:Math.max(0,Math.min(
            data.total_frames-1,item.frame+(e.key==="ArrowRight"?1:-1)))});
        }
      });
      annotation.lane.append(tick);
    }
    const touches=annotationSorted("touches");
    for(const item of touches){
      const row=document.createElement("button");row.type="button";
      row.className="studio-annotation-touch"+(item.id===annotation.touchId?" selected":"");
      row.textContent="● "+annotationClock(item.frame)+" · "+
        (trainingNames[item.kind]||"Тип не выбран");
      row.title="Выбрать касание и перейти к нему";
      row.addEventListener("click",()=>{
        annotation.touchId=item.id;seekTimeline(annotationSourceTime(item.frame));drawAnnotation();
        hideAnnotationContext();
      });
      row.addEventListener("contextmenu",event=>{
        if(reference.active)return;
        event.preventDefault();annotation.touchId=item.id;
        seekTimeline(annotationSourceTime(item.frame));showAnnotationContext(item.frame,event,item.id);
      });
      annotation.touchList.append(row);
    }
    const rallies=annotationSorted("rallies");
    for(let index=0;index<rallies.length;index++){
      const item=rallies[index];
      const row=document.createElement("div");
      row.className="studio-annotation-rally"+(item.id===annotation.rallyId?" selected":"");
      const jump=document.createElement("button");
      jump.type="button";jump.className="studio-annotation-rally-jump";
      jump.textContent="№"+(index+1)+" · "+annotationClock(item.start)+" — "+
        annotationClock(item.end);
      jump.addEventListener("click",()=>{
        annotation.rallyId=item.id;annotation.pendingStart=null;
        seekTimeline(annotationSourceTime(item.start));drawAnnotation();hideAnnotationContext();
      });
      jump.addEventListener("contextmenu",event=>{
        if(reference.active)return;
        event.preventDefault();annotation.rallyId=item.id;
        seekTimeline(annotationSourceTime(item.start));showAnnotationContext(item.start,event);
      });
      const remove=document.createElement("button");remove.type="button";
      remove.className="secondary";remove.textContent="×";remove.title="Удалить розыгрыш";
      remove.setAttribute("aria-label","Удалить размеченный розыгрыш");
      remove.addEventListener("click",()=>editAnnotation("delete_rally",{id:item.id}));
      remove.hidden=reference.active;
      row.append(jump,remove);annotation.rallies.append(row);
    }
    if(!rallies.length)annotation.rallies.textContent="Розыгрышей пока нет.";
    const m=annotation.dirty?null:data.metrics?.touches;
    annotation.metrics.textContent=m?
      (data.reviewed?"ПРОВЕРЕНО · ":"ЧЕРНОВИК · ")+
      "Совпадений "+m.matched+" · лишних "+m.extra+" · пропущено "+m.missed+
      " · F1 "+(100*m.f1).toFixed(1)+"% · ошибка времени "+
      (m.matched?Number(m.mean_error_ms).toFixed(1)+" мс":"—")+" · окно ±1 кадр (33 мс)"+
      " · границы "+(data.metrics.rallies.boundary_mean_error_ms===null?"—":
      Number(data.metrics.rallies.boundary_mean_error_ms).toFixed(1)+" мс"):
      "Нет результатов сравнения.";
    const overall=data.overall;
    if(overall?.reviewed_videos>0)annotation.metrics.textContent+=
      " | Вся проверенная база: "+overall.reviewed_videos+" видео · F1 "+
      (100*overall.f1).toFixed(1)+"% · лишних "+overall.extra+
      " · пропущено "+overall.missed+" · средняя ошибка "+
      (overall.matched?Number(overall.mean_error_ms).toFixed(1)+" мс":"—");
    if(annotation.prediction){
      const p=annotation.prediction;
      annotation.metrics.textContent+=" | По мячу "+p.proposals.length+" кандидатов · песок "+
        (p.ground_contacts?.length||0)+
        (p.court_filter_active?" · фильтр площадки":" · площадка не размечена") +
        (p.method==="touch_detector_v4"?
          " · угол излома "+p.impact_settings.min_turn_angle_deg+"°"+
          " · разрыв дуг до "+p.impact_settings.max_join_error_px+" пикс.":
          " · обучающих видео "+p.calibration.training_videos+
          " · порог "+p.calibration.min_score+
          " · сдвиг "+p.calibration.shift_frames+" кадра");
      if(p.metrics_on_labeled_video){
        annotation.metrics.textContent+=" · F1 "+(100*p.metrics_on_labeled_video.f1).toFixed(1)+"% (на обучающем видео)";
      }
    }
    if(annotation.dirty)annotation.metrics.textContent+=" | НЕ СОХРАНЕНО";
    annotationMarkCurrent();
  };
  const resetTrainingComparison=()=>{
    comparison.epoch++;
    comparison.data=null;
    comparison.sourceKey="";
    comparison.simulated=false;
    if(comparison.status)comparison.status.textContent="Сравнение выполняется отдельно по кнопке «Рассчитать розыгрыши».";
    if(typeof renderComparison==="function")renderComparison();
    window.VolleyballTasks?.clear("training-simulation");
  };
  const refreshAnnotation=async()=>{
    resetTrainingComparison();
    const project=s.project,run=s.run;
    if(annotation.dirty){
      annotationNotice("Есть несохранённая разметка. Сохраните данные касаний перед сменой CPU-версии.",true);
      drawAnnotation();return;
    }
    const epoch=++annotation.epoch;
    annotation.data=null;annotation.prediction=null;annotation.typePrediction=null;annotation.touchId=null;annotation.rallyId=null;
    annotation.pendingStart=null;annotation.source=annotationSource();
    drawOverlay();
    if(!training.active||!project||!run){
      annotationNotice("Для разметки выберите CPU-анализ и видео.");
      drawAnnotation();return;
    }
    annotationNotice("Загрузка эталонных меток…");
    try{
      const data=await request(url("/api/training/annotations",{
        project_id:project,rally_run_id:run
      }));
      if(epoch!==annotation.epoch||project!==s.project||run!==s.run)return;
      if(data.schema!=="volley-touch-annotations/1"||data.video!==s.video)
        throw Error("Неверный источник эталонной разметки");
      annotation.data={...data,touches:Array.isArray(data.touches)?data.touches:[],
        rallies:Array.isArray(data.rallies)?data.rallies:[]};drawAnnotation();
      if(reference.active)cards();

      annotationNotice(data.reviewed?"Эталон проверен. Метрики рассчитаны по исходным CPU-детекциям.":
        "Корректируйте касания и границы, затем завершите проверку видео.");
    }catch(error){
      if(epoch!==annotation.epoch)return;
      annotationNotice("Не удалось загрузить метки: "+error.message,true);drawAnnotation();
      if(reference.active)cards();

    }
  };
  const editAnnotation=async(action,fields={})=>{
    if(reference.active||annotation.busy||!annotationSourceValid()||!s.project||!s.run)return;
    if(action==="review"){
      if(annotation.dirty||!annotationReady())return;
      const project=s.project,run=s.run,epoch=annotation.epoch;
      annotation.busy=true;drawAnnotation();
      window.VolleyballTasks?.set("training-annotation",{label:"Дебаг",stage:"Проверка разметки",progress:null});
      try{
        const result=await post("/api/training/annotations",{
          action:"review",project_id:project,rally_run_id:run,revision:annotation.data.revision
        });
        if(epoch!==annotation.epoch||project!==s.project||run!==s.run)return;
        annotation.data=result;
        annotationNotice("Видео подтверждено. Эталон сохранён независимо от CPU-детектора.");
        resetTrainingComparison();
      }catch(error){annotationNotice("Ошибка подтверждения: "+error.message,true);}
      finally{annotation.busy=false;window.VolleyballTasks?.clear("training-annotation");drawAnnotation();}
      return;
    }
    if(!annotationReady())return;
    const data=annotation.data,max=data.total_frames;
    if(fields.frame!==undefined&&(!Number.isSafeInteger(fields.frame)||fields.frame<0||fields.frame>=max)){
      annotationNotice("Кадр вне пределов видео.",true);return;
    }
    if(action==="add_touch"||action==="update_touch"){
      const kind=fields.kind??null;
      const position=fields.position??null;
      if(kind!==null&&!Object.hasOwn(manualTouchKinds,kind)||
        position!==null&&(!Array.isArray(position)||position.length!==2||
          !position.every(n=>Number.isFinite(n)&&n>=0&&n<=1))){
        annotationNotice("Недопустимый тип или точка касания.",true);return;
      }
      if(action==="add_touch"){
        if(data.touches.length>=4096||data.touches.some(t=>t.frame===fields.frame)){
          annotationNotice("Касание уже существует на этом кадре или достигнут лимит.",true);return;
        }
        const id=annotationNewId();
        data.touches.push({id,frame:fields.frame,kind,position});
        annotation.touchId=id;
      }else{
        const touch=data.touches.find(t=>t.id===fields.id);
        if(!touch)return;
        touch.kind=kind;
        if(position!==null)touch.position=position;
        annotation.touchId=touch.id;
      }
    }else if(action==="move_touch"){
      const touch=data.touches.find(t=>t.id===fields.id);
      if(!touch||data.touches.some(t=>t.id!==fields.id&&t.frame===fields.frame))return;
      touch.frame=fields.frame;touch.position=null;annotation.touchId=touch.id;
    }else if(action==="delete_touch"){
      if(!data.touches.some(t=>t.id===fields.id))return;
      data.touches=data.touches.filter(t=>t.id!==fields.id);annotation.touchId=null;
    }else if(action==="add_rally"||action==="move_rally"){
      const {start,end}=fields;
      if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<0||start>=end||end>=max){
        annotationNotice("Неверные границы розыгрыша.",true);return;
      }
      if(action==="add_rally"){
        if(data.rallies.length>=1024)return;
        const id=annotationNewId();data.rallies.push({id,start,end});annotation.rallyId=id;
      }else{
        const row=data.rallies.find(r=>r.id===fields.id);if(!row)return;
        row.start=start;row.end=end;
      }
      annotation.pendingStart=null;
    }else if(action==="delete_rally"){
      if(!data.rallies.some(r=>r.id===fields.id))return;
      data.rallies=data.rallies.filter(r=>r.id!==fields.id);annotation.rallyId=null;
    }else return;
    annotation.dirty=true;data.reviewed=false;data.metrics=null;
    annotation.typePrediction=null;
    drawAnnotation();drawTimeline();drawOverlay();
    annotationNotice("Черновик изменён. Сохраните данные касаний, затем повторите расчёт по траектории.");
  };
  const impactInputIds=[
    "studio-impact-angle","studio-impact-join",
    "studio-impact-fit-error","studio-impact-min-separation"
  ];
  const impactSettingsFromPlayer=()=>{
    const [angle,join,fit,separation]=impactInputIds.map(id=>Number($(id).value));
    if(!Number.isFinite(angle)||angle<15||angle>170||
      !Number.isFinite(join)||join<8||join>80||
      !Number.isFinite(fit)||fit<1||fit>20||
      !Number.isSafeInteger(separation)||separation<3||separation>30||
      impactInputIds.some(id=>$(id).value.trim()===""))return null;
    return {
      min_turn_angle_deg:angle,
      max_join_error_px:join,
      max_fit_error_px:fit,
      min_separation_frames:separation
    };
  };
  for(const id of impactInputIds)$(id).addEventListener("change",()=>{
    if(annotation.prediction)invalidateTrainingTouchPrediction();
  });
  const trainingTouchRallySettings=()=>{
    const rally={mode:comparison.mode.value,merge_gap_seconds:trainingGapSeconds()};
    if(rally.mode==="continuous"){
      rally.min_trajectory_seconds=Number(comparison.minSeconds.value);
      rally.stationary_ball_ms=Number(comparison.stationaryMs.value);
    }else{
      rally.min_impacts=Number(comparison.minImpacts.value);
      rally.context_frames=Number(comparison.context.value);
      rally.min_duration_frames=Number(comparison.duration.value);
      rally.max_touch_gap_frames=Number(comparison.gap.value);
    }
    return rally;
  };
  const invalidateTrainingTouchPrediction=()=>{
    if(!annotation.prediction)return;
    annotation.prediction=null;annotation.typePrediction=null;
    drawAnnotation();drawTimeline();drawOverlay();
    annotationNotice("Настройки розыгрышей или столкновений изменены. Повторите расчёт касаний.");
  };
  const calculateTrajectory=async()=>{
    if(annotation.busy||annotation.dirty||!s.run||!annotationSourceValid())return;
    const impact=impactSettingsFromPlayer();
    if(!impact){annotationNotice("Проверьте параметры поиска столкновений.",true);return;}
    const rallies=comparisonMatches(comparison.data)&&comparison.simulated&&
      comparison.data.trajectory_status==="ready"?comparison.data:null;
    if(!rallies){
      annotationNotice("Сначала нажмите «Рассчитать розыгрыши», затем «Рассчитать касания».",true);
      return;
    }
    if(comparison.recalculate.disabled){
      annotationNotice("Расчёт розыгрышей ещё выполняется. Касания доступны после завершения.",true);
      return;
    }
    const rally=trainingTouchRallySettings(),parameters=rallies.parameters;
    if(!parameters||parameters.min_trajectory_seconds!==rally.min_trajectory_seconds||
      parameters.stationary_ball_ms!==rally.stationary_ball_ms||
      parameters.merge_gap_seconds!==rally.merge_gap_seconds){
      annotationNotice("Параметры изменились. Повторно рассчитайте розыгрыши перед поиском касаний.",true);
      return;
    }
    if(!Array.isArray(rallies.trajectory_rallies)){
      annotationNotice("Некорректные интервалы розыгрышей. Повторите расчёт.",true);
      return;
    }
    const rally_intervals=rallies.trajectory_rallies.map(({start,end})=>({start,end}));
    const source_replay_sha256=rallies.source_replay_sha256;
    if(!/^[0-9a-f]{64}$/.test(source_replay_sha256)){
      annotationNotice("Некорректный источник рассчитанных розыгрышей. Повторите расчёт.",true);
      return;
    }
    const epoch=annotation.epoch,project=s.project,run=s.run,video=s.video;
    annotation.busy=true;drawAnnotation();
    window.VolleyballTasks?.set("training-trajectory",{label:"Дебаг",stage:"Touch Detector · розыгрыши по мячу",progress:null});
    annotationNotice("Ищем изменения траектории мяча только внутри рассчитанных розыгрышей, без GPU…");
    const settingsKey=JSON.stringify({rally,impact});
    try{
      const result=await post("/api/training/trajectory/calculate",{
        project_id:project,rally_run_id:run,source_replay_sha256,rally_intervals,impact
      });
      if(comparison.data!==rallies||settingsKey!==JSON.stringify({
        rally:trainingTouchRallySettings(),impact:impactSettingsFromPlayer()
      }))
        throw Error("Настройки розыгрышей изменились. Повторите расчёт касаний.");
      if(epoch!==annotation.epoch||project!==s.project||run!==s.run||video!==s.video)return;
      if(result.schema!=="volley-trajectory-touch-proposals/1"||
          result.project_id!==project||result.video!==video||result.rally_run_id!==run||
          !Array.isArray(result.proposals)||!Array.isArray(result.serve_proposals)||
          !Array.isArray(result.ground_contacts)||
          typeof result.court_filter_active!=="boolean"||
          !Number.isSafeInteger(result.total_frames)||
          result.method!=="touch_detector_v4"||
          result.ball_motion_interpolation?.stage!=="ball_motion_interpolation_v1"||
          !Number.isSafeInteger(result.ball_motion_interpolation?.result?.rejected_outliers)||
          !Number.isSafeInteger(result.ball_motion_interpolation?.result?.interpolated_frames)||
          !Number.isSafeInteger(result.ball_motion_interpolation?.result?.processed_frames)||
          result.rally_source!=="calculated_ball_trajectory_rallies"||
          result.source_replay_sha256!==source_replay_sha256||
          Object.entries(impact).some(([key,value])=>result.impact_settings?.[key]!==value)||
          !Array.isArray(result.rally_intervals)||
          result.rally_intervals.length!==rally_intervals.length||
          result.rally_intervals.some((interval,i)=>interval.start!==rally_intervals[i].start||
            interval.end!==rally_intervals[i].end)||
          result.rally_count!==result.rally_intervals.length||
          result.serve_proposals.length!==result.rally_intervals.length||
          result.serve_proposals.some((item,i)=>{
            const interval=result.rally_intervals[i];
            const frame=item.trajectory_start_frame??item.rally_start_frame;
            return item.kind!=="serve"||item.identity_verified!==false||
              item.source!=="rally_trajectory_start_v1"||
              item.rally_start_frame!==interval.start||
              !Number.isSafeInteger(frame)||frame<interval.start||
              frame>Math.min(interval.end,interval.start+29)||
              (item.provisional_track_id!==null&&
                (!Number.isSafeInteger(item.provisional_track_id)||item.provisional_track_id<=0));
          })||
          result.ball_motion_interpolation.result.processed_frames!==
            result.rally_intervals.reduce((sum,r)=>sum+r.end-r.start+1,0)||
          result.rally_intervals.some(r=>!Number.isSafeInteger(r.start)||!Number.isSafeInteger(r.end)||
            r.start<0||r.start>=r.end||r.end>=result.total_frames)||
          result.proposals.some(p=>!Number.isSafeInteger(p.frame)||p.frame<0||
            p.frame>=result.total_frames||
            !result.rally_intervals.some(r=>r.start<=p.frame&&p.frame<=r.end))||
          result.ground_contacts.some(g=>!Number.isSafeInteger(g.frame)||g.frame<0||
            g.frame>=result.total_frames||!groundContactAppearance[g.landing]||
            !Array.isArray(g.position)||g.position.length!==2||
            !g.position.every(Number.isFinite)||
            !result.rally_intervals.some(r=>r.start<=g.frame&&g.frame<=r.end)||
            result.proposals.some(p=>Math.abs(p.frame-g.frame)<=2)))
        throw Error("Касания выходят за рассчитанные розыгрыши по мячу");
      annotation.prediction=result;annotation.typePrediction=null;
      timelineRallyLayers.candidates.checked=true;
      updateTrainingRallyLayers();
      drawTimeline();drawOverlay();
      if(epoch!==annotation.epoch||project!==s.project||run!==s.run||video!==s.video)return;
      annotationNotice("Touch Detector: обработано "+result.ball_motion_interpolation.result.processed_frames+
        " кадров в "+result.rally_count+" рассчитанных по мячу розыгрышах (вне розыгрышей касания не искались; всего в видео "+result.total_frames+" кадров). Интерполяция мяча: отброшено "+result.ball_motion_interpolation.result.rejected_outliers+
        " ложных точек, восстановлено "+result.ball_motion_interpolation.result.interpolated_frames+
        " пропущенных кадров. Найдено "+
        result.proposals.length+" кандидатов касаний, "+result.serve_proposals.length+
        " начальных подач и "+
        result.ground_contacts.length+" отдельных ударов о песок. "+
        (result.court_filter_active?"Половины определены по разметке площадки. ":
        "Нет разметки площадки — фильтр ударов о песок отключён. ")+
        "Вне розыгрышей поиск не выполняется. Кандидаты столкновений не подтверждены игроком.");
    }catch(error){
      if(epoch===annotation.epoch)annotationNotice("Ошибка расчёта касаний: "+error.message,true);
    }finally{
      annotation.busy=false;window.VolleyballTasks?.clear("training-trajectory");drawAnnotation();
    }
  };
  const calculateTouchTypes=async()=>{
    if(annotation.busy||annotation.dirty||!annotation.prediction||!annotationSourceValid())return;
    const epoch=annotation.epoch,project=s.project,run=s.run,video=s.video;
    const candidates=annotation.prediction;
    const impact=impactSettingsFromPlayer();
    if(!impact){annotationNotice("Проверьте параметры поиска столкновений.",true);return;}
    const rally=trainingTouchRallySettings(),settingsKey=JSON.stringify({rally,impact});
    annotation.busy=true;drawAnnotation();
    window.VolleyballTasks?.set("training-types",{
      label:"Дебаг",stage:"Определение типов касаний по RTMPose",progress:null
    });
    annotationNotice("Проверяем движение рук RTMPose только в найденных по мячу касаниях…");
    try{
      const result=await post("/api/training/trajectory/types",{
        project_id:project,rally_run_id:run,
        source_replay_sha256:candidates.source_replay_sha256,
        rally_intervals:candidates.rally_intervals,impact,frames:candidates.proposals.map(item=>item.frame)
      });
      if(epoch!==annotation.epoch||s.project!==project||s.run!==run||s.video!==video||
        annotation.prediction!==candidates)return;
      if(JSON.stringify({
        rally:trainingTouchRallySettings(),impact:impactSettingsFromPlayer()
      })!==settingsKey)
        throw Error("Настройки розыгрышей изменились, рассчитайте типы заново.");
      if(result.schema!=="volley-trajectory-touch-types/1"||
        result.project_id!==project||result.video!==video||result.rally_run_id!==run||
        result.source_replay_sha256!==candidates.source_replay_sha256||
        result.total_frames!==candidates.total_frames||
        !Array.isArray(result.touches)||
        result.touches.length!==candidates.proposals.length||
        result.touches.some((item,i)=>item.frame!==candidates.proposals[i].frame||
          ![null,"attack","overhead_pass","dig","serve","block"].includes(item.kind)||
          item.kind!==null&&(!Number.isSafeInteger(item.track_id)||item.track_id<=0)))
        throw Error("Классификация не совпадает с найденными касаниями по мячу");
      annotation.typePrediction=result;
      timelineRallyLayers.candidates.checked=true;
      timelineRallyLayers.types.checked=true;
      updateTrainingRallyLayers();
      annotationNotice("Тип определён у "+result.classified+" из "+result.total+
        " касаний. Остальные остаются «Касание»; результаты не являются подтверждённой разметкой.");
    }catch(error){
      if(epoch===annotation.epoch)annotationNotice("Ошибка определения типов: "+error.message,true);
    }finally{
      annotation.busy=false;
      window.VolleyballTasks?.clear("training-types");
      drawAnnotation();
    }
  };
  annotation.classify.addEventListener("click",calculateTouchTypes);
  annotation.calculate.addEventListener("click",calculateTrajectory);
  annotation.adopt.addEventListener("click",()=>{
    if(!annotation.prediction||annotation.busy||!annotationSourceValid())return;
    if(annotationReady()&&!window.confirm("Заменить текущую разметку расчётом? Несохранённые изменения будут потеряны."))return;
    annotation.data=annotationFromPrediction();
    annotation.touchId=null;annotation.rallyId=null;annotation.dirty=true;
    drawAnnotation();drawTimeline();
    annotationNotice("Прогноз взят в черновик. Исправьте метки и сохраните данные касаний.");
  });
  annotation.saveQuick.addEventListener("click",()=>annotation.save.click());
  annotation.markButton.addEventListener("click",()=>{
    if(annotation.markButton.disabled)return;
    annotation.markMode=!annotation.markMode;
    ui.video.pause();hideAnnotationContext();drawAnnotation();
    if(annotation.markMode)annotationNotice("Пауза: нажмите на точку контакта с мячом, выберите тип и сохраните эталон.");
  });
  annotation.save.addEventListener("click",async()=>{
    if(annotation.busy||!annotation.dirty||!annotationReady()||!annotationSourceValid())return;
    const epoch=annotation.epoch,project=s.project,run=s.run,video=s.video;
    annotation.busy=true;drawAnnotation();
    window.VolleyballTasks?.set("training-annotation",{label:"Дебаг",stage:"Сохранение меток и признаков траектории",progress:null});
    annotationNotice("Сохраняем метки и признаки траектории…");
    try{
      const result=await post("/api/training/annotations",{
        action:"save",project_id:project,rally_run_id:run,revision:annotation.data.revision||0,
        touches:annotation.data.touches,rallies:annotation.data.rallies
      });
      if(epoch!==annotation.epoch||project!==s.project||run!==s.run||video!==s.video)return;
      if(result.schema!=="volley-touch-annotations/1"||result.video!==video)throw Error("Неверное подтверждение сохранения");
      annotation.data=result;annotation.dirty=false;annotation.prediction=null;annotation.typePrediction=null;
      drawTimeline();drawOverlay();resetTrainingComparison();
      annotationNotice("Данные сохранены. Повторите расчёт касаний — алгоритм использует исправления для подбора порога и сдвига.");
    }catch(error){
      if(epoch===annotation.epoch)annotationNotice("Ошибка сохранения: "+error.message,true);
    }finally{
      annotation.busy=false;window.VolleyballTasks?.clear("training-annotation");drawAnnotation();
    }
  });
  annotation.seed.addEventListener("click",()=>{
    if(annotation.busy||!annotationSourceValid()||!timeline.data)return;
    if(annotationReady()&&!window.confirm("Взять текущие CPU-метки за основу черновика?"))return;
    const touches=[...new Set((timeline.data.touch_frames||[]).filter(n=>Number.isSafeInteger(n)&&n>=0))]
      .slice(0,4096).map(frame=>({id:annotationNewId(),frame}));
    annotation.data={...annotation.data,touches,rallies:annotationTrajectoryRallySeeds(),reviewed:false,metrics:null};
    annotation.dirty=true;annotation.touchId=null;annotation.rallyId=null;
    drawAnnotation();drawTimeline();annotationNotice("CPU-метки скопированы в черновик. Исправьте и сохраните данные.");
  });
  annotation.add.addEventListener("click",()=>editAnnotation("add_touch",{frame:annotationFrame()}));
  annotation.move.addEventListener("click",()=>{
    if(annotation.touchId)editAnnotation("move_touch",{id:annotation.touchId,frame:annotationFrame()});
  });
  annotation.remove.addEventListener("click",()=>{
    if(annotation.touchId)editAnnotation("delete_touch",{id:annotation.touchId});
  });
  annotation.next.addEventListener("click",()=>{
    annotation.rallyId=null;annotation.pendingStart=null;
    annotationNotice("Новый розыгрыш: выберите кадр начала, затем кадр конца.");
    drawAnnotation();
  });
  annotation.begin.addEventListener("click",()=>{
    const frame=annotationFrame();
    const row=annotation.data?.rallies?.find(x=>x.id===annotation.rallyId);
    if(row)editAnnotation("move_rally",{id:row.id,start:frame,end:row.end});
    else{annotation.pendingStart=frame;drawAnnotation();}
  });
  annotation.finish.addEventListener("click",()=>{
    const frame=annotationFrame();
    const row=annotation.data?.rallies?.find(x=>x.id===annotation.rallyId);
    if(row)editAnnotation("move_rally",{id:row.id,start:row.start,end:frame});
    else if(annotation.pendingStart!==null){
      const start=annotation.pendingStart;
      editAnnotation("add_rally",{start,end:frame}).then(()=>{
        if(annotation.data?.rallies?.some(x=>x.start===start&&x.end===frame)){
          annotation.pendingStart=null;drawAnnotation();
        }
      });
    }else annotationNotice("Сначала установите начало нового розыгрыша.",true);
  });
  annotation.review.addEventListener("click",()=>editAnnotation("review"));
  ui.video.addEventListener("timeupdate",annotationMarkCurrent);
  ui.video.addEventListener("seeked",annotationMarkCurrent);
  window.addEventListener("beforeunload",event=>{
    if(annotation.dirty){event.preventDefault();event.returnValue="";}
  });
  const annotationFocus=()=>{
    if(!training.active)return;
    const byPose=annotation.method.value==="pose";
    training.poseAdvanced.open=byPose;
    overlay.canvas.closest(".video-wrap")?.classList.toggle("training-select",byPose);
    overlay.canvas.closest(".video-wrap")?.classList.toggle("training-annotation",!byPose);
    if(annotation.hideExtra.checked){
      if(!annotation.priorFlags)annotation.priorFlags={...overlay.flags};
      overlay.flags.players=byPose;overlay.flags.pose=byPose;
      overlay.flags.court=false;overlay.flags.rallies=false;
      overlay.flags.touches=false;overlay.flags.trajectory=true;
    }else if(annotation.priorFlags){
      Object.assign(overlay.flags,annotation.priorFlags);
    }
    updateOverlayToggles();drawOverlay();drawTimeline();
  };
  annotation.method.addEventListener("change",annotationFocus);
  annotation.hideExtra.addEventListener("change",annotationFocus);
  const comparison={
    data:null,sourceKey:"",epoch:0,shown:200,simulated:false,
    status:$("studio-comparison-status"),
    summary:$("studio-simulation-summary"),touches:$("studio-comparison-touches"),
    metrics:$("studio-comparison-metrics"),
    differences:$("studio-comparison-differences"),
    filter:$("studio-comparison-filter"),
    recalculate:$("studio-comparison-recalculate"),
    calibrate:$("studio-comparison-calibrate"),
    minImpacts:$("studio-comparison-min-impacts"),
    context:$("studio-comparison-context"),
    duration:$("studio-comparison-duration"),
    gap:$("studio-comparison-gap"),
    mode:$("studio-comparison-mode"),
    impactSettings:$("studio-comparison-impact-settings"),
    continuousSettings:$("studio-comparison-continuous-settings"),
    minSeconds:$("studio-comparison-min-seconds"),
    stationaryMs:$("studio-comparison-stationary-ms")
  };
  const continuousComparisonMode=()=>comparison.mode.value==="continuous";
  const updateComparisonModeSettings=()=>{
    comparison.impactSettings.hidden=continuousComparisonMode();
    comparison.continuousSettings.hidden=!continuousComparisonMode();
    comparison.recalculate.textContent="Рассчитать розыгрыши";
  };
  const comparisonMatches=data=>data&&
    data.schema==="volley-rally-training-comparison/1"&&
    data.project_id===s.project&&data.rally_run_id===s.run&&data.video===s.video;
  const mergeTrainingIntervals=(items,gapFrames,stops=[])=>{
    const sorted=items.filter(item=>Number.isSafeInteger(item.start)&&
      Number.isSafeInteger(item.end)&&item.start>=0&&item.end>=item.start)
      .sort((a,b)=>a.start-b.start||a.end-b.end);
    const merged=[];
    for(const item of sorted){
      const previous=merged[merged.length-1];
      const hold=previous&&stops.some(stop=>Number(stop[0])<item.start&&
        Number(stop[1])>previous.end);
      if(previous&&!hold&&(item.start<=previous.end+1||
        item.start-previous.end-1<gapFrames)){
        previous.end=Math.max(previous.end,item.end);
        previous.members.push(...item.members);
      }else merged.push({start:item.start,end:item.end,members:[...item.members]});
    }
    return merged;
  };
  const comparisonTime=frame=>(Number(frame)/fps()).toFixed(2)+" с";
  const comparisonRange=value=>value?
    comparisonTime(value.start)+" — "+comparisonTime(value.end):"—";
  const renderComparison=()=>{
    const data=comparisonMatches(comparison.data)?comparison.data:null;
    const ready=data?.trajectory_status==="ready";
    const selected=Boolean(training.active&&s.project&&s.run&&s.video);
    comparison.metrics.replaceChildren();
    comparison.differences.replaceChildren();
    comparison.touches.replaceChildren();
    updateComparisonModeSettings();
    for(const input of [comparison.minImpacts,comparison.context,comparison.duration,comparison.gap])
      input.disabled=!selected||continuousComparisonMode();
    comparison.minSeconds.disabled=!selected||!continuousComparisonMode();
    comparison.stationaryMs.disabled=!selected||!continuousComparisonMode();
    comparison.recalculate.disabled=!selected;
    comparison.calibrate.disabled=!data || continuousComparisonMode() || !data.reference_reviewed;
    if(!data){
      comparison.summary.textContent="Результат моделирования ещё не рассчитан";
      comparison.metrics.textContent="Сравнение ещё не загружено.";
      comparison.differences.textContent=selected?
        "Для сравнения отдельно нажмите «Рассчитать розыгрыши».":
        "Выберите CPU-анализ для просмотра расхождений.";
      return;
    }
    if(!ready){
      comparison.summary.textContent="Нажмите «Рассчитать розыгрыши» для расчёта по сохранённому треку мяча";
      comparison.metrics.textContent="Розыгрыши по траектории ещё не рассчитаны.";
      comparison.differences.textContent="Непрерывная траектория будет рассчитана из сохранённого трека мяча без повторного GPU-инференса.";
      return;
    }
    const addMetric=(title,body)=>{
      const card=document.createElement("div");
      card.className="studio-training-compare-metric";
      const label=document.createElement("small");label.textContent=title;
      const value=document.createElement("strong");value.textContent=body;
      card.append(label,value);comparison.metrics.append(card);
    };

    const continuous=data.mode==="continuous";
    const simulated=Array.isArray(data.simulated_touch_candidates)?data.simulated_touch_candidates:[];
    const unassigned=Array.isArray(data.unassigned_touch_candidates)?data.unassigned_touch_candidates:[];
    comparison.summary.textContent=comparison.simulated?
      continuous?"Устойчивая траектория: "+data.trajectory_rallies.length+" отрезков до объединения":
      "Алгоритм по мячу: "+simulated.length+
      " возможных касаний внутри · "+unassigned.length+
      " перегибов вне розыгрышей":"Нажмите «Смоделировать»";
    const touchCounts=trainingTouchCountSummary();
    if(touchCounts)comparison.summary.textContent+=" · "+touchCounts;

    addMetric("По мячу до объединения",String(data.trajectory_rallies.length));
    if(!continuous){
      addMetric("Кандидаты внутри розыгрышей",String(simulated.length));
      addMetric("Перегибы вне розыгрышей",String(unassigned.length));
    }



    if(data.calibration){
      const before=data.calibration.baseline.f1*100;
      const after=data.calibration.fitted.f1*100;
      addMetric("Подбор по этому видео · F1",before.toFixed(1)+"% → "+after.toFixed(1)+"%");
    }
    if(data.reference_reviewed&&data.metrics.trajectory_reference){
      addMetric("Мяч против эталона · F1",(data.metrics.trajectory_reference.f1*100).toFixed(1)+"%");
    }else{
      addMetric("Ручной эталон",data.reference_rallies===null?"Нет разметки":
        "Черновик · подтвердите разметку видео");
    }
    if(comparison.simulated&&!continuous&&
      typeof timelineRallyLayers!=="undefined"&&timelineRallyLayers.candidates.checked){
      for(const frame of simulated.slice(0,160)){
        const button=document.createElement("button");button.type="button";
        button.className="studio-simulated-touch";
        button.textContent="◉ "+comparisonTime(frame);
        button.title="Перейти к кандидату касания (не подтверждён)";
        button.addEventListener("click",()=>seekTimeline(frame/fps()));
        button.addEventListener("contextmenu",event=>{
          event.preventDefault();seekTimeline(frame/fps());showAnnotationContext(frame,event);
        });
        comparison.touches.append(button);
      }
      if(unassigned.length){
        const label=document.createElement("small");
        label.textContent="Перегибы вне розыгрышей · возможные, но не подтверждённые касания:";
        comparison.touches.append(label);
        for(const frame of unassigned.slice(0,160)){
          const button=document.createElement("button");button.type="button";
          button.className="studio-simulated-touch";
          button.textContent="◌ "+comparisonTime(frame);
          button.title="Проверить перегиб мяча вне розыгрыша (не подтверждённое касание)";
          button.addEventListener("click",()=>seekTimeline(frame/fps()));
          button.addEventListener("contextmenu",event=>{
            event.preventDefault();seekTimeline(frame/fps());showAnnotationContext(frame,event);
          });
          comparison.touches.append(button);
        }
        if(unassigned.length>160){
          const count=document.createElement("small");
          count.textContent="Показаны первые 160 из "+unassigned.length+" перегибов вне розыгрышей";
          comparison.touches.append(count);
        }
      }
      if(simulated.length>160){
        const count=document.createElement("small");
        count.textContent="Показаны первые 160 из "+simulated.length+" · все точки доступны на шкале";
        comparison.touches.append(count);
      }
    }
    const kindLabels={agreed:"Совпадают",shifted:"Разные границы",
      cpu_only:"Только CPU",trajectory_only:"Только траектория",
      both_missed:"Пропущено обоими"};
    const filter=comparison.filter.value;
    const rows=data.differences.filter(item=>filter==="all"||
      filter==="disagreement"&&item.kind!=="agreed"||item.kind===filter);
    if(!rows.length){
      comparison.differences.textContent=comparison.simulated&&data.trajectory_rallies.length===0?
        continuous?
        "Не найден устойчивый движущийся трек мяча дольше выбранного порога. Попробуйте уменьшить длительность.":
        data.trajectory_impacts.length===0?
        "Перегибы мяча вниз→вверх не обнаружены в сохранённом треке. Проверьте качество исходных наблюдений мяча.":
        "Перегибы есть, но ни один не образовал розыгрыш. Уменьшите порог перегибов или увеличьте паузу между ними и смоделируйте снова.":
        "Нет расхождений по выбранному фильтру.";
      return;
    }
    const total=document.createElement("small");
    total.className="studio-training-compare-counter";
    total.textContent="Найдено: "+rows.length+" · нажмите на строку для перехода к эпизоду";
    comparison.differences.append(total);
    for(const item of rows.slice(0,comparison.shown)){
      const row=document.createElement("button");
      row.type="button";
      row.className="studio-training-compare-row "+item.kind;
      const name=document.createElement("strong");
      name.textContent=kindLabels[item.kind]||item.kind;
      const range=document.createElement("span");
      range.textContent="CPU: "+comparisonRange(item.cpu)+"  ·  Мяч: "+
        comparisonRange(item.trajectory)+(item.reference?"  ·  Эталон: "+comparisonRange(item.reference):"")+
        (item.iou===null?"":"  ·  IoU "+item.iou.toFixed(2));
      row.append(name,range);
      row.addEventListener("click",()=>{
        ui.video.pause();
        seekTimeline(item.seek_frame/fps());
      });
      comparison.differences.append(row);
    }
    if(rows.length>comparison.shown){
      const more=document.createElement("button");
      more.type="button";more.className="secondary";
      more.textContent="Показать ещё ("+(rows.length-comparison.shown)+")";
      more.addEventListener("click",()=>{comparison.shown+=200;renderComparison();});
      comparison.differences.append(more);
    }
  };
  const refreshComparison=async(useControls=false,calibrate=false)=>{
    const epoch=++comparison.epoch,project=s.project,run=s.run,video=s.video;
    if(!training.active||!project||!run||!video){
      comparison.data=null;comparison.sourceKey="";comparison.simulated=false;
      window.VolleyballTasks?.clear("training-simulation");
      comparison.status.textContent="Выберите видео и CPU-постобработку.";
      renderComparison();cards();return;
    }
    const key=project+":"+run+":"+video;
    const sameSource=comparison.sourceKey===key;
    const selectedMode="continuous";
    const simulate=useControls||calibrate||selectedMode==="continuous"||
      (sameSource&&comparison.simulated);
    if(!sameSource)comparison.simulated=false;
    comparison.status.textContent="Рассчитываем розыгрыши по сохранённой траектории мяча…";
    comparison.recalculate.disabled=true;
    comparison.calibrate.disabled=true;
    const options={project_id:project,rally_run_id:run};
    if(simulate){
      options.simulate=true;
      options.mode=selectedMode;
      if(selectedMode==="continuous"){
        options.min_trajectory_seconds=Number(comparison.minSeconds.value);
        options.merge_gap_seconds=trainingGapSeconds();
        options.stationary_ball_ms=Number(comparison.stationaryMs.value);
      }else{
        options.min_impacts=Number(comparison.minImpacts.value);
        options.context_frames=Number(comparison.context.value);
        options.min_duration_frames=Number(comparison.duration.value);
        options.max_touch_gap_frames=Number(comparison.gap.value);
      }
    }
    if(calibrate)options.calibrate=true;
    window.VolleyballTasks?.set("training-simulation",{
      label:"Дебаг",
      stage:calibrate?"Подбор параметров по эталону":simulate?"Моделирование розыгрышей и касаний":"Сравнение алгоритмов",
      progress:null
    });
    try{
      const data=await request(url("/api/training/rally-comparison",options));
      if(epoch!==comparison.epoch||project!==s.project||run!==s.run||
        video!==s.video||!training.active)return;
      if(!comparisonMatches(data)||(data.mode&&data.mode!==selectedMode)||
        !Array.isArray(data.cpu_rallies)||
        !Array.isArray(data.trajectory_rallies)||!Array.isArray(data.differences)||
        !Array.isArray(data.trajectory_impacts))
        throw Error("Источник сравнения не совпадает с выбранным анализом");
      comparison.data=data;comparison.sourceKey=key;comparison.shown=200;
      if(simulate)comparison.simulated=true;
      if(data.parameters){
        if(selectedMode==="continuous"){
          comparison.minSeconds.value=data.parameters.min_trajectory_seconds;
          comparison.stationaryMs.value=data.parameters.stationary_ball_ms;
        }else{
          comparison.minImpacts.value=data.parameters.min_impacts;
          comparison.context.value=data.parameters.context_frames;
          comparison.duration.value=data.parameters.min_duration_frames;
          comparison.gap.value=data.parameters.max_touch_gap_frames;
        }
      }
      comparison.status.textContent=data.calibration?
        "Параметры подобраны по подтверждённой разметке текущего видео. Это обучение на примере, не независимая оценка точности.":
        data.mode==="continuous"?
        "Устойчивое движение VBallNet с короткими пропусками; модель розыгрышей не подтверждает касания игроков.":
        data.trajectory_status==="ready"?
        simulate&&data.trajectory_impacts.length===0?
        "В сохранённом треке мяча не найдено перегибов вниз→вверх. Проверьте данные траектории.":
        simulate&&data.trajectory_rallies.length===0?
        "Перегибы есть, но пороги не сформировали розыгрыши. Измените число перегибов или паузу между ними.":
        "Результат по сохранённой траектории VBallNet; возможные касания не подтверждены игроками и не классифицированы.":
        "Траекторных интервалов нет в этой CPU-версии.";
      renderComparison();
      drawTimeline();cards();
    }catch(error){
      if(epoch!==comparison.epoch)return;
      comparison.data=null;
      comparison.simulated=false;
      comparison.status.textContent="Ошибка сравнения: "+error.message;
      renderComparison();drawTimeline();cards();
    }finally{
      if(epoch===comparison.epoch)window.VolleyballTasks?.clear("training-simulation");
    }
  };
  comparison.mode.addEventListener("change",()=>{
    invalidateTrainingTouchPrediction();
    updateComparisonModeSettings();
    comparison.data=null;
    comparison.sourceKey="";
    comparison.simulated=false;
    renderComparison();cards();
    void refreshComparison(true);
  });
  updateComparisonModeSettings();
  for(const input of [comparison.minSeconds,comparison.stationaryMs,comparison.minImpacts,comparison.context,
    comparison.duration,comparison.gap])
    input.addEventListener("change",invalidateTrainingTouchPrediction);
  comparison.recalculate.addEventListener("click",()=>{
    invalidateTrainingTouchPrediction();
    void refreshComparison(true);
  });
  comparison.calibrate.addEventListener("click",()=>void refreshComparison(true,true));
  comparison.filter.addEventListener("change",()=>{comparison.shown=200;renderComparison();});
  const annotationContext={
    root:$("studio-annotation-context"),
    label:$("studio-annotation-context-label"),
    actions:$("studio-annotation-context-actions")
  };
  const hideAnnotationContext=()=>annotationContext.root.classList.add("hidden");
  const showAnnotationContext=(atFrame,anchor={},touchId=null,position=null)=>{
    if(reference.active||!training.active||!s.run||!s.video)return;
    const maxFrame=(annotation.data?.total_frames||timeline.data?.total_frames||1)-1;
    const frame=Math.max(0,Math.min(maxFrame,Math.round(atFrame)));
    const ready=Boolean(annotationReady());
    const touch=(annotation.data?.touches||[]).find(item=>item.id===touchId)||
      (annotation.data?.touches||[]).find(item=>Math.abs(item.frame-frame)<=1);
    const selected=(annotation.data?.touches||[]).find(item=>item.id===annotation.touchId);
    const rally=(annotation.data?.rallies||[]).find(item=>item.id===annotation.rallyId)||
      (annotation.data?.rallies||[]).find(item=>item.start<=frame&&item.end>=frame);
    const trajectoryProposal=annotation.prediction?.proposals?.some(p=>Math.abs(p.frame-frame)<=1);
    const rawImpact=Boolean(comparison.simulated&&comparisonMatches(comparison.data)&&
      comparison.data.unassigned_touch_candidates?.includes(frame));
    const simulated=Boolean(trajectoryProposal)||rawImpact||(comparison.simulated&&comparisonMatches(comparison.data)&&
      comparison.data.simulated_touch_candidates.includes(frame));
    ui.video.pause();
    annotationContext.actions.replaceChildren();
    annotationContext.label.textContent=(rawImpact?"Перегиб вне розыгрыша · ":simulated?"Кандидат касания · ":
      touch?"Размеченное касание · ":"Кадр ")+frame+
      " · "+comparisonTime(frame);
    const action=(label,callback,destructive=false)=>{
      const button=document.createElement("button");button.type="button";
      button.className=destructive?"secondary studio-context-destructive":"secondary";
      button.textContent=label;
      button.disabled=annotation.busy;
      button.addEventListener("click",()=>{hideAnnotationContext();callback();});
      annotationContext.actions.append(button);
    };
    const edit=(name,fields)=>void editAnnotation(name,fields);
    const typeField=document.createElement("label");
    typeField.className="studio-manual-touch-kind";
    typeField.textContent="Тип касания";
    const typeSelect=document.createElement("select");
    typeSelect.setAttribute("aria-label","Тип ручного касания");
    for(const [value,title] of [["","Тип не определён"],...Object.entries(manualTouchKinds)]){
      const option=document.createElement("option");
      option.value=value;option.textContent=title;typeSelect.append(option);
    }
    typeSelect.value=touch?.kind||"";
    typeField.append(typeSelect);
    if(ready)annotationContext.actions.append(typeField);
    if(!ready){
      if(annotation.prediction)action("✓ Взять прогноз касаний для правок",()=>annotation.adopt.click());
      action("Взять CPU-разметку за основу",()=>annotation.seed.click());
    }else{
      if(!touch){
        action(simulated?"✓ Добавить кандидат в эталон":"+ Касание здесь",
          ()=>edit("add_touch",{frame,kind:typeSelect.value||null,position}));
      }else{
        annotation.touchId=touch.id;
        action("Обновить тип касания",()=>edit("update_touch",{
          id:touch.id,kind:typeSelect.value||null,position}));
        action("← Сдвинуть на кадр",()=>edit("move_touch",{id:touch.id,
          frame:Math.max(0,touch.frame-1)}));
        action("Сдвинуть на кадр →",()=>edit("move_touch",{id:touch.id,
          frame:Math.min(maxFrame,touch.frame+1)}));
        action("Удалить касание",()=>edit("delete_touch",{id:touch.id}),true);
      }
      if(selected&&selected.id!==touch?.id&&selected.frame!==frame){
        action("Переместить выбранное сюда",()=>edit("move_touch",{id:selected.id,frame}));
      }
      if(annotation.pendingStart!==null){
        if(frame>annotation.pendingStart){
          action("Закончить новый розыгрыш",()=>{
            const start=annotation.pendingStart;
            void editAnnotation("add_rally",{start,end:frame}).then(()=>{
              if(annotation.data?.rallies?.some(r=>r.start===start&&r.end===frame)){
                annotation.pendingStart=null;drawAnnotation();
              }
            });
          });
        }
        action("Отменить начало",()=>{annotation.pendingStart=null;drawAnnotation();});
      }else{
        action("Начало нового розыгрыша",()=>{
          annotation.rallyId=null;annotation.pendingStart=frame;drawAnnotation();
        });
      }
      if(rally){
        annotation.rallyId=rally.id;
        if(frame<rally.end){
          action("Начало розыгрыша здесь",()=>edit("move_rally",{
            id:rally.id,start:frame,end:rally.end}));
        }
        if(frame>rally.start){
          action("Конец розыгрыша здесь",()=>edit("move_rally",{
            id:rally.id,start:rally.start,end:frame}));
        }
        action("Удалить розыгрыш",()=>edit("delete_rally",{id:rally.id}),true);
      }
    }
    if(annotation.dirty)action("Сохранить данные касаний",()=>annotation.save.click());
    action("Закрыть",()=>{});
    const wrap=overlay.canvas.closest(".video-wrap");
    const bounds=wrap.getBoundingClientRect();
    const px=Number.isFinite(anchor.clientX)?anchor.clientX-bounds.left:bounds.width/2;
    const py=Number.isFinite(anchor.clientY)?anchor.clientY-bounds.top:bounds.height*0.7;
    annotationContext.root.style.left=Math.max(8,Math.min(bounds.width-276,px))+"px";
    annotationContext.root.style.top=Math.max(8,Math.min(bounds.height-315,py))+"px";
    annotationContext.root.classList.remove("hidden");
  };
  annotationContext.root.addEventListener("pointerdown",event=>event.stopPropagation());
  annotationContext.root.addEventListener("click",event=>event.stopPropagation());
  document.addEventListener("keydown",event=>{
    if(event.key==="Escape")hideAnnotationContext();
  });
  overlay.canvas.addEventListener("click",event=>{
    if(!training.active||training.poseAdvanced.open)return;
    if(reference.active||!annotation.markMode||!annotationReady()||
      !annotationSourceValid()||annotation.busy){hideAnnotationContext();return;}
    if(!ui.video.paused){ui.video.pause();return;}
    const rect=overlay.canvas.getBoundingClientRect();
    const w=ui.video.videoWidth,h=ui.video.videoHeight;
    if(!rect.width||!rect.height||!w||!h)return;
    const scale=Math.min(rect.width/w,rect.height/h);
    const x=(event.clientX-rect.left-(rect.width-w*scale)/2)/(w*scale);
    const y=(event.clientY-rect.top-(rect.height-h*scale)/2)/(h*scale);
    if(x<0||x>1||y<0||y>1){
      annotationNotice("Выберите точку внутри изображения видео.",true);return;
    }
    showAnnotationContext(annotationFrame(),event,null,[
      Number(x.toFixed(4)),Number(y.toFixed(4))
    ]);
  });
  overlay.canvas.addEventListener("contextmenu",event=>{
    if(reference.active||!training.active||training.poseAdvanced.open||!s.run||!s.video)return;
    event.preventDefault();
    showAnnotationContext(annotationFrame(),event);
  });
  const pickTrainingPlayer=event=>{
    if(reference.active||!training.active||!training.poseAdvanced.open||!ui.video.paused||!s.run||!s.video)return;
    const frame=Math.max(0,Math.floor(ui.video.currentTime*fps()+0.001));
    const data=cachedOverlayForFrame(s.project,s.run,s.video,frame);
    const row=data?.frames?.find(item=>item.frame===frame);
    if(!row){trainingMessage("Дождитесь загрузки данных CPU и RTMPose для этого кадра.",true);return;}
    const rect=overlay.canvas.getBoundingClientRect();
    const w=ui.video.videoWidth,h=ui.video.videoHeight;
    const scale=Math.min(rect.width/w,rect.height/h);
    const rx=(rect.width-w*scale)/2,ry=(rect.height-h*scale)/2;
    const px=(event.clientX-rect.left-rx)/(w*scale);
    const py=(event.clientY-rect.top-ry)/(h*scale);
    const candidates=(row.players||[]).filter(player=>{
      const b=player.box;
      return Array.isArray(b)&&px>=b[0]&&px<=b[2]&&py>=b[1]&&py<=b[3];
    });
    candidates.sort((a,b)=>
      (a.box[2]-a.box[0])*(a.box[3]-a.box[1])-
      (b.box[2]-b.box[0])*(b.box[3]-b.box[1]));
    const picked=candidates[0];
    training.track=picked?.id||null;training.frame=picked?frame:null;
    if(!picked)trainingMessage("В этом месте нет рамки игрока. Выберите игрока внутри рамки.",true);
    else if(row.pose_points_interpolated)
      trainingMessage("ID "+picked.id+" выбран. Поза интерполирована — выберите соседний кадр с измерением.",true);
    else trainingMessage("ID "+picked.id+" · кадр "+frame+" · выберите действительное касание и нажмите «Сохранить пример».");
  };
  overlay.canvas.addEventListener("click",pickTrainingPlayer);
  training.poseAdvanced.addEventListener("toggle",()=>{
    if(training.poseAdvanced.open)annotation.markMode=false;
    drawAnnotation();
    overlay.canvas.closest(".video-wrap")?.classList.toggle("training-select",training.active&&training.poseAdvanced.open);
    overlay.canvas.closest(".video-wrap")?.classList.toggle("training-annotation",training.active&&!training.poseAdvanced.open);
    hideAnnotationContext();
  });
  training.refresh.addEventListener("click",refreshTraining);
  training.newExample.addEventListener("click",()=>{
    training.selectedId=null;
    training.track=null;training.frame=null;
    trainingEditorReset();
    for(const row of training.list.querySelectorAll(".studio-training-example"))
      row.classList.remove("selected");
    trainingMessage("Новый пример: выберите кадр и коснувшегося мяча игрока.");
  });
  training.save.addEventListener("click",async()=>{
    if(reference.active||!training.active||training.busy)return;
    if(!ui.video.paused){trainingMessage("Поставьте видео на паузу.",true);return;}
    const frame=Math.max(0,Math.floor(ui.video.currentTime*fps()+0.001));
    if(!s.run||!training.track||training.frame!==frame){
      trainingMessage("Выберите касавшегося мяча игрока на текущем кадре.",true);return;
    }
    training.busy=true;training.save.disabled=true;
    window.VolleyballTasks?.set("training-save",{label:"Дебаг",stage:"Сохранение примера",progress:null});
    try{
      const editing=training.editId;
      await post("/api/training/examples",{
        action:editing?"replace_touch":"touch",
        project_id:s.project,video:s.video,rally_run_id:s.run,
        frame_index:frame,track_id:training.track,label:training.label.value,
        ...(editing?{id:editing,revision:training.revision}:{})
      });
      training.track=null;training.frame=null;
      if(editing)trainingEditorReset();
      trainingMessage(editing?
        "Пример перезаписан из нового проверенного кадра и позы.":
        "Пример сохранён: точки RTMPose и контакт с мячом. Влияет на следующий CPU-анализ.");
      await refreshTraining();
    }catch(error){
      trainingMessage("Не удалось сохранить пример: "+error.message,true);
      if(training.editId)await refreshTraining();
    }finally{training.busy=false;training.save.disabled=false;
      window.VolleyballTasks?.clear("training-save");}
  });
  for(const key of ["seeked","loadedmetadata"]){
    ui.video.addEventListener(key,()=>{training.track=null;training.frame=null;
      if(training.active)trainingMessage("Пауза → щёлкните по игроку, коснувшемуся мяча.");});
  }
  if(overlay.mode){
    const selectMode=raw=>{
      if(overlay.raw===raw&&!overlay.stabilizerView&&!training.active)return;
      const restoreFilter=training.active?training.previousRallyFilter:null;
      training.active=false;training.track=null;training.frame=null;
      training.previousRallyFilter=null;
      reference.active=false;reference.epoch++;reference.entries=[];
      reference.picker.classList.add("hidden");
      reference.analysis.setAttribute("aria-pressed","true");
      reference.toggle.setAttribute("aria-pressed","false");
      $("studio-preview-view").classList.remove("reference-workspace");
      $("studio-overlay-rallies").disabled=false;
      if(restoreFilter&&restoreFilter!==ui.filter.value){
        ui.filter.value=restoreFilter;
        void loadPage().catch(failed);
      }
      if(s.current.startsWith("ball-"))s.current="";
      comparison.epoch++;comparison.data=null;comparison.sourceKey="";comparison.simulated=false;
      hideAnnotationContext();
      training.panel.classList.add("hidden");
      if(annotation.priorFlags){Object.assign(overlay.flags,annotation.priorFlags);annotation.priorFlags=null;}
      annotation.lane.querySelectorAll(".studio-annotation-tick,.studio-trajectory-proposal").forEach(tick=>tick.remove());
      overlay.canvas.closest(".video-wrap")?.classList.remove("training-annotation");
      training.selectedId=null;trainingEditorReset();
      overlay.canvas.closest(".video-wrap")?.classList.remove("training-select");
      overlay.raw=raw;
      overlay.stabilizerView=false;
      updateRawDetectorLabel();
      overlay.mode.setAttribute("aria-selected",String(!raw));
      overlay.rawMode.setAttribute("aria-selected",String(raw));
      overlay.stabilizerMode.setAttribute("aria-pressed","false");
      overlay.debugTrajectoryLegend.classList.add("hidden");
      overlay.rawControls.classList.toggle("hidden",!raw);
      overlay.debugTrajectorySwitch.classList.add("hidden");
      updateOverlayToggles();
      overlay.mode.closest(".video-overlay-toggles").classList.toggle("raw-selected",raw);
      setOverlaySource(raw||s.run?{}:null);
      if(raw&&s.video){ensurePreviewVideo();prefetchOverlayAtFrame(0);}
      if(raw&&!rawFusion())overlayMessage("У выбранного GPU-источника нет сохранённого объединённого fusion.");
      if(!raw&&!s.run)overlayMessage("Для обработанного режима выберите CPU-постобработку.");
      if(raw)timeline.status.textContent="Сырые GPU-данные · без CPU-разметки";
      else if(!s.run)timeline.status.textContent="Выберите CPU-версию";
      else if(timeline.data)timeline.status.textContent=
        timeline.data.rallies.length+" розыгрышей · "+timeline.data.touch_annotations.length+" касаний (CPU)";
      updateReferenceModeUi();
      placeTrainingTimeline();
      drawTimeline();
      cards();
      refreshRallyCardTouchTicks();
      drawOverlay();
    };
    overlay.mode.addEventListener("click",()=>selectMode(false));
    overlay.rawMode.addEventListener("click",()=>selectMode(true));
    const selectDebugComparison=comparison=>{
      if(!training.active)return;
      if(comparison&&!rawFusion()){
        overlayMessage("Для сравнения нужен сохранённый GPU-анализ.");
        return;
      }
      overlay.raw=comparison;
      overlay.stabilizerView=comparison;
      overlay.stabilizerMode.setAttribute("aria-pressed",String(comparison));
      overlay.debugTrajectoryLegend.classList.toggle("hidden",!comparison);
      if(comparison)overlay.flags.trajectory=true;
      overlay.debugBefore.setAttribute("aria-pressed",String(!comparison&&overlay.trajectoryVariant==="before"));
      overlay.debugAfter.setAttribute("aria-pressed",String(!comparison&&overlay.trajectoryVariant==="after"));
      overlay.rawControls.classList.add("hidden");
      overlay.mode.closest(".video-overlay-toggles").classList.remove("raw-selected");
      updateOverlayToggles();
      setOverlaySource(comparison||s.run?{}:null);
      if(comparison&&s.video){
        ensurePreviewVideo();
        prefetchOverlayAtFrame(0);
      }
      if(comparison)timeline.status.textContent="Обе траектории · голубая: VBallNet, золотая: Ball Stabilizer";
      else if(!s.run)timeline.status.textContent="Выберите CPU-версию";
      else if(timeline.data)timeline.status.textContent=
        "Розыгрыши по мячу: "+trainingRallyGroups().ball.length+
        " · пауза < "+trainingGapSeconds()+" с";
      drawTimeline();
      drawOverlay();
    };
    overlay.stabilizerMode.addEventListener("click",()=>selectDebugComparison(!overlay.stabilizerView));
    const selectTrajectoryVariant=variant=>{
      overlay.trajectoryVariant=variant;
      if(overlay.stabilizerView)selectDebugComparison(false);
      overlay.flags.trajectory=true;
      updateOverlayToggles();
      overlay.debugBefore.setAttribute("aria-pressed",String(variant==="before"));
      overlay.debugAfter.setAttribute("aria-pressed",String(variant==="after"));
      overlayMessage(variant==="after"?
        "После стабилизации · используется сохранённая траектория 2-й фазы":
        "До стабилизации · исходные точки VBallNet из CPU replay");
      drawOverlay();
    };
    overlay.debugBefore.addEventListener("click",()=>selectTrajectoryVariant("before"));
    overlay.debugAfter.addEventListener("click",()=>selectTrajectoryVariant("after"));
    const selectTrainingMode=()=>{
      if(training.active)return;
      ui.video.pause();
      overlay.raw=false;overlay.stabilizerView=false;training.active=true;training.track=null;training.frame=null;
      reference.active=false;reference.epoch++;reference.entries=[];
      updateReferenceModeUi();
      s.current="";s.stopAt=null;s.clipStart=null;s.clipEnd=null;
      wipe(ui.events);
      ui.viewTitle.textContent="Розыгрыш по мячу не выбран";
      ui.viewDescription.textContent="Выберите розыгрыш по траектории. CPU-касания в дебаге не отображаются.";
      training.previousRallyFilter=ui.filter.value;
      if(ui.filter.value!=="all"){
        ui.filter.value="all";
        void loadPage().catch(failed);
      }
      overlay.flags.pose=true;overlay.flags.players=true;
      overlay.mode.setAttribute("aria-selected","false");
      overlay.rawMode.setAttribute("aria-selected","false");
      overlay.stabilizerMode.setAttribute("aria-pressed","false");
      overlay.debugTrajectoryLegend.classList.add("hidden");
      overlay.debugBefore.setAttribute("aria-pressed",String(overlay.trajectoryVariant==="before"));
      overlay.debugAfter.setAttribute("aria-pressed",String(overlay.trajectoryVariant==="after"));
      overlay.mode.closest(".video-overlay-toggles").classList.remove("raw-selected");
      overlay.debugTrajectorySwitch.classList.remove("hidden");
      overlay.rawControls.classList.add("hidden");
      training.panel.classList.remove("hidden");
      placeTrainingTimeline();
      annotationFocus();
      updateTrainingRallyLayers();
      updateOverlayToggles();
      setOverlaySource(s.run?{}:null);
      drawTimeline();drawOverlay();cards();
      trainingMessage("Остановите видео в момент удара и щёлкните по рамке игрока. Нужны сохранённые RTMPose и VBallNet.");
      refreshTraining();
      refreshAnnotation();
    };
    window.addEventListener("volley:tab-change",event=>{
      if(event.detail?.tab==="training")selectTrainingMode();
      else if(training.active)selectMode(false);
    });
    for(const [input,value] of [[overlay.rawBall,$("studio-raw-ball-value")],
      [overlay.rawPlayer,$("studio-raw-player-value")],
      [overlay.rawPose,$("studio-raw-pose-value")]]){
      input.addEventListener("input",()=>{value.textContent=input.value+"%";drawOverlay();});
    }
    for(const input of [overlay.rawVballVisible,overlay.rawPeopleVisible,overlay.rawPlayerInterpolated,overlay.rawPoseVisible,overlay.rawBallTrail].filter(Boolean))
      input.addEventListener("change",drawOverlay);
    for(const input of overlay.actionControls.querySelectorAll("[data-overlay-action]")){
      input.addEventListener("change",()=>{
        overlay.actionFlags[input.dataset.overlayAction]=input.checked;
        drawOverlay();drawTimeline();refreshRallyCardTouchTicks();
      });
    }
  }
  const startOverlayFrameLoop=()=>{
    const token=++overlay.loop;
    if(typeof ui.video.requestVideoFrameCallback!=="function")return;
    const tick=(_now,metadata)=>{
      if(token!==overlay.loop||ui.video.seeking)return;
      if(Number.isFinite(metadata?.mediaTime))
        overlay.presentedTime=metadata.mediaTime;
      drawOverlay();updateRallyCardProgress();
      if(!ui.video.paused)ui.video.requestVideoFrameCallback(tick);
    };
    ui.video.requestVideoFrameCallback(tick);
  };
  ui.video.addEventListener("playing",startOverlayFrameLoop);
  ui.video.addEventListener("pause",()=>{
    overlay.loop++;overlay.presentedTime=null;drawOverlay();
  });
  ui.video.addEventListener("seeking",()=>{
    updateRallyCardProgress();
    overlay.loop++;overlay.presentedTime=null;
    overlayContext(); // Never leave previous-frame boxes on a pending seek.
  });
  ui.video.addEventListener("seeked",()=>{
    overlay.presentedTime=null;drawOverlay();updateRallyCardProgress();
    if(!ui.video.paused)startOverlayFrameLoop();
  });
  ui.video.addEventListener("timeupdate",()=>{
    if(typeof ui.video.requestVideoFrameCallback!=="function"&&!ui.video.seeking)
      drawOverlay();
  });
  ui.video.addEventListener("loadedmetadata",()=>{
    overlay.presentedTime=null;drawOverlay();
  });
  window.addEventListener("resize",drawOverlay);
  const wipe=node=>{while(node.firstChild)node.firstChild.remove();};
  const message=(text,mood)=>{
    ui.banner.className="banner"+(mood?" "+mood:"");
    ui.banner.textContent=String(text);
  };
  const failed=e=>message(e instanceof Error?e.message:String(e),"error");
  const fps=()=>window.VolleyballUi.engine.volley_evidence_fps(s.fps||0);
  const clock=frames=>{
    const sec=Math.max(0,Math.floor((Number(frames)||0)/fps()));
    return String(Math.floor(sec/60)).padStart(2,"0")+":"+String(sec%60).padStart(2,"0");
  };
  const length=item=>(Math.max(0,
    (Number(item.detected_end_frame)-Number(item.detected_start_frame)+1)/fps())).toFixed(1)+" с";
  const second=frame=>(Math.max(0,Number(frame)||0)/fps())
    .toLocaleString("ru-RU",{minimumFractionDigits:1,maximumFractionDigits:1});
  const rallyPlayerCount=item=>Number.isSafeInteger(item.stable_track_count)&&item.stable_track_count>=0?
    item.stable_track_count:null;
  const valid=id=>typeof id==="string"&&/^[a-f0-9]{24}$/.test(id);
  // The Rust timeline index is source-verified and spans the complete original
  // video; unlike rally cards it is independent of pagination and gold filters.
  const timeline={
    track:$("studio-timeline-track"),canvas:$("studio-timeline-canvas"),
    playhead:$("studio-timeline-playhead"),status:$("studio-timeline-status"),
    middle:$("studio-timeline-middle"),end:$("studio-timeline-end"),
    data:null,epoch:0,dragging:false
  };
  const timelineRallyLayers={
    touches:$("studio-training-rallies-by-touches"),
    ball:$("studio-training-rallies-by-trajectory"),
    candidates:$("studio-training-ball-touch-candidates"),
    types:$("studio-training-ball-touch-types"),
    typeStatus:$("studio-training-ball-touch-types-status"),
    gap:$("studio-training-merge-gap-seconds"),
    cpuSummary:$("studio-training-cpu-summary"),
    ballSummary:$("studio-training-ball-summary"),
    provenance:$("studio-training-ball-touch-provenance")
  };
  const trainingGapSeconds=()=>Math.min(60,Math.max(0,
    Number.isFinite(Number(timelineRallyLayers.gap.value))?
      Number(timelineRallyLayers.gap.value):4));
  const trainingRallyGroups=()=>{
    const ball=comparisonMatches(comparison.data)&&
      comparison.data?.trajectory_status==="ready"?
      comparison.data.trajectory_rallies:[];
    const ballIntervals=ball.map(item=>({
      start:Number(item.start),end:Number(item.end),members:[item]
    }));
    const gapFrames=trainingGapSeconds()*fps();
    return {
      touches:[],
      ball:mergeTrainingIntervals(ballIntervals,gapFrames,[
        ...(comparison.data?.parameters?.stationary_intervals||[]),
        ...(comparison.data?.parameters?.unobserved_intervals||[])
      ])
    };
  };
  const trainingRallyNoun=count=>{
    const tail=count%10,two=count%100;
    return tail===1&&two!==11?"розыгрыш":
      tail>=2&&tail<=4&&(two<12||two>14)?"розыгрыша":"розыгрышей";
  };
  const durationTrainingGroups=groups=>{
    const frames=groups.reduce((sum,group)=>sum+group.end-group.start+1,0);
    const seconds=Math.round(frames/fps());
    const clockText=seconds>=3600?
      Math.floor(seconds/3600)+":"+String(Math.floor(seconds%3600/60)).padStart(2,"0"):
      String(Math.floor(seconds/60));
    return clockText+":"+String(seconds%60).padStart(2,"0");
  };
  const updateTrainingSummaries=groups=>{
    timelineRallyLayers.cpuSummary.textContent="Устаревший способ отключён";
    timelineRallyLayers.ballSummary.textContent=comparisonMatches(comparison.data)&&
      comparison.data?.trajectory_status==="ready"?
      groups.ball.length+" "+trainingRallyNoun(groups.ball.length)+" · "+
        durationTrainingGroups(groups.ball):
      "Розыгрыши ещё не рассчитаны";
    const prediction=annotation.prediction&&annotationSourceValid()?
      annotation.prediction.proposals:null;
    if(training.active&&timeline.data)timeline.status.textContent=
      "Розыгрыши по мячу: "+groups.ball.length+
      " · пауза < "+trainingGapSeconds()+" с"+
      (trainingTouchCountSummary()?" · "+trainingTouchCountSummary():"");
    timelineRallyLayers.provenance.textContent=prediction?
      prediction.length+" кандидатов · в "+annotation.prediction.rally_count+" розыгрышах VBallNet":
      "Нет расчёта касаний · используется только VBallNet";
  };
  const showBallCandidates=()=>training.active&&!reference.active&&timelineRallyLayers.ball.checked&&
    timelineRallyLayers.candidates.checked;
  const showBallTouchTypes=()=>showBallCandidates()&&timelineRallyLayers.types.checked;
  const updateTrainingRallyLayers=()=>{
    annotation.touchLayerVisible=reference.active||timelineRallyLayers.touches.checked;
    annotation.ballLayerVisible=showBallCandidates();
    timelineRallyLayers.candidates.disabled=!timelineRallyLayers.ball.checked;
    timelineRallyLayers.types.disabled=!showBallCandidates();
    timelineRallyLayers.typeStatus.textContent=!showBallCandidates()?
      "Сначала включите определение касаний":
      annotation.typePrediction?
      annotation.typePrediction.classified+" типов из "+annotation.typePrediction.total+" касаний · RTMPose":
      "Типы ещё не рассчитаны · нажмите «Определить тип касания»";
    for(const [selector,visible] of [
      [".studio-annotation-tick",annotation.touchLayerVisible],
      [".studio-trajectory-proposal",annotation.ballLayerVisible]
    ])annotation.lane.querySelectorAll(selector).forEach(marker=>{marker.hidden=!visible;});
    drawTimeline();drawOverlay();
    ui.list.scrollTop=0;
    cards();
    renderComparison();
  };
  annotation.touchLayerVisible=timelineRallyLayers.touches.checked;
  annotation.ballLayerVisible=showBallCandidates();
  for(const layer of [timelineRallyLayers.touches,timelineRallyLayers.ball,
    timelineRallyLayers.candidates,timelineRallyLayers.types])
    layer.addEventListener("change",updateTrainingRallyLayers);
  timelineRallyLayers.gap.addEventListener("input",()=>{
    if(timelineRallyLayers.gap.value.trim()!==""){
      invalidateTrainingTouchPrediction();
      updateTrainingRallyLayers();
    }
  });
  timelineRallyLayers.gap.addEventListener("change",()=>{
    timelineRallyLayers.gap.value=String(trainingGapSeconds());
    invalidateTrainingTouchPrediction();
    updateTrainingRallyLayers();
  });
  const rallyCardProgresses=new Map();
  const placeTrainingTimeline=()=>{
    const element=$("studio-timeline");
    const layout=document.querySelector("#studio-preview-view .viewer-layout");
    if(element.parentElement!==layout)layout.append(element);
    requestAnimationFrame(drawTimeline);
  };
  const timeLabel=value=>{
    const seconds=Math.max(0,Math.floor(Number(value)||0));
    const minutes=Math.floor(seconds/60),s=String(seconds%60).padStart(2,"0");
    return minutes>=60?Math.floor(minutes/60)+":"+
      String(minutes%60).padStart(2,"0")+":"+s:
      String(minutes).padStart(2,"0")+":"+s;
  };
  const timelineDuration=()=>{
    const data=timeline.data;
    const actual=s.videoKey===s.project+":"+s.video?ui.video.duration:0;
    // The decoded media is authoritative even when no CPU timeline exists.
    // Otherwise raw-only GPU previews could play but never seek.
    if(Number.isFinite(actual)&&actual>0)return actual;
    if(!data)return 0;
    const source=Number(data.video_duration_sec);
    return Number.isFinite(source)&&source>0?source:
      Math.max(0,Number(data.total_frames)/fps()||0);
  };
  const updateTimelineProgress=()=>{
    const duration=timelineDuration();
    const position=Math.min(duration,Math.max(0,
      Number.isFinite(ui.video.currentTime)&&s.videoKey===s.project+":"+s.video?
        ui.video.currentTime:s.seekAt));
    timeline.playhead.style.left=duration?(100*position/duration)+"%":"0%";
    timeline.middle.textContent=timeLabel(Math.floor(position*10+0.5)/10)+"."+String(Math.floor(position*10+0.5)%10);
    timeline.track.setAttribute("aria-valuemax",String(Math.ceil(duration)));
    timeline.track.setAttribute("aria-valuenow",String(Math.floor(position)));
    timeline.track.setAttribute("aria-valuetext",
      duration?timeLabel(position)+" из "+timeLabel(duration):"Нет видео");
  };
  const drawTimeline=()=>{
    const canvas=timeline.canvas,ctx=canvas.getContext("2d");
    if(!ctx)return;
    const width=canvas.clientWidth,height=canvas.clientHeight;
    if(width<=0||height<=0)return;
    const dpr=Math.min(window.devicePixelRatio||1,2);
    const dw=Math.round(width*dpr),dh=Math.round(height*dpr);
    if(canvas.width!==dw||canvas.height!==dh){canvas.width=dw;canvas.height=dh;}
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,width,height);
    const y=3,bandHeight=Math.max(1,height-6);
    ctx.fillStyle="#192e49";ctx.fillRect(0,y,width,bandHeight);
    const data=timeline.data,duration=timelineDuration();
    if(!overlay.raw&&data&&duration){
      // A single shared seek lane: Touch ticks are overlaid on rally bands.
      if((training.active?timelineRallyLayers.touches.checked:overlay.flags.rallies)){
        for(const rally of (training.active?trainingRallyGroups().touches:
          data.rallies)){
          const start=(Number(s.analysisOffset)||0)+Number(rally.start_frame??rally.start)/fps(),
            end=(Number(s.analysisOffset)||0)+(Number(rally.end_frame??rally.end)+1)/fps();
          const x=Math.max(0,Math.min(width,start/duration*width));
          const x2=Math.max(0,Math.min(width,end/duration*width));
          const selected=s.current===rally.candidate_id;
          const gold=s.gold.has(rally.candidate_id);
          ctx.fillStyle=gold?"#cfaf6959":selected?"#57c9e099":"#37b9ce80";
          ctx.fillRect(x,y,Math.max(1,x2-x),bandHeight);
          ctx.fillStyle=gold?"#eac77dc7":selected?"#8ae4f6":"#4dd5e0bb";
          ctx.fillRect(x,y,Math.max(1,x2-x),Math.min(3,bandHeight));
        }
      }
      if(training.active&&!reference.active&&timelineRallyLayers.ball.checked&&
        comparisonMatches(comparison.data)&&comparison.data.trajectory_status==="ready"){
        for(const rally of trainingRallyGroups().ball){
          const x=((Number(s.analysisOffset)||0)+rally.start/fps())/duration*width;
          const x2=((Number(s.analysisOffset)||0)+(rally.end+1)/fps())/duration*width;
          ctx.fillStyle="#efb15773";
          ctx.fillRect(x,y,Math.max(2,x2-x),bandHeight);
          ctx.fillStyle="#f4c576c2";
          ctx.fillRect(x,y+Math.max(0,bandHeight-3),Math.max(2,x2-x),Math.min(3,bandHeight));
        }
      }
      if(showBallCandidates()&&annotation.prediction&&annotationSourceValid()){
        for(const serve of (annotation.prediction.serve_proposals||[])){
          const frame=serve.trajectory_start_frame??serve.rally_start_frame;
          const x=((Number(s.analysisOffset)||0)+frame/fps())/duration*width;
          ctx.fillStyle="#f3a350";
          ctx.fillRect(Math.floor(x),y+bandHeight-12,Math.max(2,Math.ceil(dpr)),8);
        }
        for(const proposal of annotation.prediction.proposals){
          const x=((Number(s.analysisOffset)||0)+proposal.frame/fps())/duration*width;
          const kind=showBallTouchTypes()&&annotation.typePrediction?.touches.find(v=>v.frame===proposal.frame)?.kind;
          ctx.fillStyle=kind?({attack:"#ff6b6b",overhead_pass:"#66c8ff",dig:"#56dfb9",serve:"#f3a350",block:"#ac92ff"}[kind]||"#ffbd55"):"#ffbd55";
          ctx.fillRect(Math.floor(x),y+bandHeight-8,1,5);
        }
        for(const ground of (annotation.prediction.ground_contacts||[])){
          const x=((Number(s.analysisOffset)||0)+ground.frame/fps())/duration*width;
          ctx.fillStyle=(groundContactAppearance[ground.landing]||["#ffd166"])[0];
          ctx.fillRect(Math.floor(x),y+bandHeight-16,2,6);
        }
      }
      if(training.active&&(reference.active||timelineRallyLayers.touches.checked)&&annotationReady()&&
        (!reference.active||annotation.data.reviewed&&annotationSourceValid())){
        for(const mark of annotation.data.rallies){
          const x=((Number(s.analysisOffset)||0)+mark.start/fps())/duration*width;
          const x2=((Number(s.analysisOffset)||0)+(mark.end+1)/fps())/duration*width;
          ctx.fillStyle="#00ad9470";ctx.fillRect(x,y,Math.max(1,x2-x),bandHeight);
        }
        for(const mark of annotation.data.touches){
          const x=((Number(s.analysisOffset)||0)+mark.frame/fps())/duration*width;
          ctx.fillStyle=({attack:"#ff6b6b",block:"#ac92ff",serve:"#f3a350",
            overhead_pass:"#66c8ff",dig:"#56dfb9"}[mark.kind]||"#10deb2");
          ctx.fillRect(x,y,Math.max(2,Math.ceil(dpr)),bandHeight);
        }
      }
      if(!training.active&&overlay.flags.touches&&overlay.actionFlags.serve&&
        Array.isArray(data.serve_proposals)){
        for(const serve of data.serve_proposals){
          const frame=Number(serve.trajectory_start_frame??serve.rally_start_frame);
          const x=Math.floor(((Number(s.analysisOffset)||0)+frame/fps())/duration*width);
          if(Number.isFinite(x)&&x>=0&&x<width){
            ctx.fillStyle="#f3a350";
            ctx.fillRect(x,y,Math.max(3,Math.ceil(dpr)),bandHeight);
          }
        }
      }
      if(!training.active&&overlay.flags.touches){
        for(const event of floorContacts(data)){
          const x=Math.floor(((Number(s.analysisOffset)||0)+event.frame_index/fps())/duration*width);
          if(!Number.isFinite(x)||x<0||x>=width)continue;
          ctx.fillStyle=floorContactColor;
          ctx.fillRect(x,y,Math.max(3,Math.ceil(dpr)),bandHeight);
        }
      }
      if(!training.active&&overlay.flags.touches&&Array.isArray(data.touch_annotations)){
        for(const classified of [false,true]){
          for(const touch of data.touch_annotations){
            const kind=cpuTouchKind(touch.kind),style=cpuTouchStyle[kind];
            if((kind!=="unknown_touch")!==classified||!overlay.actionFlags[kind])continue;
            const x=Math.floor(((Number(s.analysisOffset)||0)+Number(touch.frame_index)/fps())/duration*width);
            if(!Number.isFinite(x)||x<0||x>=width)continue;
            ctx.fillStyle=style[0];
            ctx.fillRect(x,y,Math.max(2,Math.ceil(dpr)),bandHeight);
          }
        }
      }
    }
    $("studio-timeline-rally-key").hidden=overlay.raw||(training.active?!timelineRallyLayers.touches.checked:!overlay.flags.rallies);
    $("studio-timeline-ball-key").hidden=!training.active||overlay.raw||!timelineRallyLayers.ball.checked||
      !comparisonMatches(comparison.data)||comparison.data.trajectory_status!=="ready";
    $("studio-timeline-touch-key").hidden=overlay.raw||training.active||!overlay.flags.touches;
    $("studio-timeline-manual-key").hidden=!training.active||
      !(reference.active||timelineRallyLayers.touches.checked);
    $("studio-timeline-candidates-key").hidden=!showBallCandidates()||
      !(annotation.prediction&&annotationSourceValid());
    updateTimelineProgress();
  };
  const clearTimeline=()=>{
    timeline.epoch++;timeline.dragging=false;timeline.data=null;
    if(training.active&&!annotation.dirty){annotation.epoch++;annotation.data=null;annotation.prediction=null;annotation.typePrediction=null;annotation.touchId=null;
      comparison.epoch++;comparison.data=null;comparison.sourceKey="";
      annotation.rallyId=null;annotation.pendingStart=null;drawAnnotation();}
    timeline.status.textContent="Выберите CPU-версию";
    timeline.middle.textContent="—";timeline.end.textContent="—";
    timeline.track.title="";drawTimeline();refreshRallyCardTouchTicks();
  };
  const loadTimeline=async()=>{
    const project=s.project,run=s.run,epoch=++timeline.epoch;
    if(!project||!run)return;
    timeline.status.textContent="Загрузка временных меток…";
    try{
      const raw=await previewCache.load(url("/api/cpu/native/rallies/timeline",{
        project_id:project,rally_run_id:run
      }),900000,true);
      if(epoch!==timeline.epoch||s.project!==project||s.run!==run)return;
      if(raw.schema!=="volley-native-rally-timeline/1"||
        raw.project_id!==project||raw.rally_run_id!==run||
        typeof raw.video!=="string"||
        !Number.isSafeInteger(raw.total_frames)||raw.total_frames<=0||
        !Number.isFinite(raw.video_duration_sec)||raw.video_duration_sec<=0||
        !Array.isArray(raw.rallies)||!Array.isArray(raw.touch_frames)||
        !Array.isArray(raw.touch_annotations)||
        raw.touch_annotations.length!==raw.touch_frames.length||
        !raw.touch_annotations.every((event,index)=>
          event.frame_index===raw.touch_frames[index]&&
          (event.kind===null||Object.hasOwn(cpuTouchStyle,event.kind)))||
        !validSurfaceContacts(raw)||raw.authoritative_game_events!==false||
        (s.video&&raw.video!==s.video))
        throw Error("Недостоверный индекс временной шкалы");
      timeline.data=raw;
      s.analysisOffset=Number(raw.analysis_offset_sec)||0;
      s.fps=Number(raw.fps)||0;
      if(!s.video)s.video=raw.video;
      ensurePreviewVideo();
      const actual=raw.touch_annotations;
      const attacks=actual.filter(e=>e.kind==="attack").length;
      const blocks=actual.filter(e=>e.kind==="block").length;
      timeline.status.textContent=raw.rallies.length+" розыгрышей · "+
        actual.length+" касаний (CPU) · "+floorContacts(raw).length+" касаний пола · "+
        attacks+" атак · "+blocks+" блоков";
      const duration=timelineDuration();
      timeline.middle.textContent="00:00.0";
      timeline.end.textContent=timeLabel(duration);
      drawTimeline();syncFavoritesSelection();drawOverlay();
      refreshRallyCardTouchTicks();updateRallyCardProgress();
      if(training.active)cards();
      if(training.active)refreshAnnotation();
      if(s.current){
        const selected=s.items.find(item=>item.candidate_id===s.current);
        if(selected){
          const {start,end}=rallyClipFrames(selected);
          renderRallyTouches(selected,start,end);
        }
      }
    }catch(error){
      if(epoch!==timeline.epoch)return;
      timeline.data=null;drawTimeline();refreshRallyCardTouchTicks();
      timeline.status.textContent="Шкала недоступна: "+String(error?.message||error);
    }
  };
  const rallyAtTimelineTime=seconds=>{
    if(training.active){
      const frame=(seconds-(Number(s.analysisOffset)||0))*fps();
      if(reference.active){
        const match=annotationSourceValid()&&annotation.data?.reviewed?
          annotation.data.rallies.find(rally=>rally.start<=frame&&rally.end>=frame):null;
        return match?{start_frame:match.start,end_frame:match.end}:null;
      }
      const groups=trainingRallyGroups();
      const match=(timelineRallyLayers.touches.checked?groups.touches:[])
        .find(group=>group.start<=frame&&group.end>=frame)||
        (timelineRallyLayers.ball.checked?groups.ball:[])
        .find(group=>group.start<=frame&&group.end>=frame);
      return match?{start_frame:match.start,end_frame:match.end}:null;
    }
    const rows=timeline.data?.rallies||[];
    let left=0,right=rows.length;
    const frame=(seconds-(Number(s.analysisOffset)||0))*fps();
    while(left<right){
      const mid=(left+right)>>>1;
      if(rows[mid].start_frame<=frame)left=mid+1;else right=mid;
    }
    return left>0&&rows[left-1].end_frame>=frame?rows[left-1]:null;
  };
  const schedulePreviewRecovery=key=>{
    if(s.videoKey!==key||previewLoad.key!==key||previewLoad.pending||
       previewLoad.timer||previewLoad.retries>=2)return;
    const delay=800*(++previewLoad.retries);
    previewLoad.timer=setTimeout(()=>{
      previewLoad.timer=null;
      if(s.videoKey===key&&previewLoad.key===key&&!previewLoad.pending)
        loadPreviewVideo();
    },delay);
  };
  const loadPreviewVideo=(autoplay=false)=>{
    const project=s.project,video=s.video,key=project+":"+video;
    if(!project||!video)return;
    if(previewLoad.timer)clearTimeout(previewLoad.timer);
    previewLoad.timer=null;
    if(previewLoad.key!==key){previewLoad.key=key;previewLoad.retries=0;}
    previewLoad.pending=true;
    const epoch=++previewLoad.epoch;
    const onLoaded=()=>{
      if(epoch!==previewLoad.epoch||s.videoKey!==key)return;
      previewLoad.pending=false;
      if(!ui.video.getAttribute("src"))schedulePreviewRecovery(key);
    };
    const onFailed=error=>{
      if(epoch!==previewLoad.epoch||s.videoKey!==key)return;
      previewLoad.pending=false;
      message("Не удалось загрузить видео: "+String(error?.message||error),"error");
      schedulePreviewRecovery(key);
    };
    if(window.VolleyballBrowserVideo?.load){
      try{
        Promise.resolve(window.VolleyballBrowserVideo.load(ui.video,project,video,{
          autoplay,seek:s.seekAt
        })).then(onLoaded,onFailed);
      }catch(error){onFailed(error);}
    }else{
      ui.video.src=url("/api/video-media",{project_id:project,video});
      ui.video.load();
      onLoaded();
      if(autoplay)ui.video.play().catch(()=>{});
    }
  };
  const ensurePreviewVideo=()=>{
    if(!s.project||!s.video)return;
    const key=s.project+":"+s.video;
    if(s.videoKey===key){
      if(!previewLoad.pending&&!previewLoad.timer&&
         (!ui.video.getAttribute("src")||ui.video.error))
        loadPreviewVideo();
      return;
    }
    s.videoKey=key;
    ui.emptyVideo.classList.add("hidden");
    loadPreviewVideo();
  };
  window.addEventListener("volley:playback-status",event=>{
    if(event.detail?.video===ui.video&&event.detail?.state==="failed")
      schedulePreviewRecovery(s.project+":"+s.video);
  });
  const seekTimeline=seconds=>{
    if((!overlay.raw&&!timeline.data)||!s.video||(!overlay.raw&&!s.run))return;
    const duration=timelineDuration();
    if(!duration)return;
    const position=Math.max(0,Math.min(Math.max(0,duration-0.05),seconds));
    s.stopAt=null;s.clipStart=null;s.clipEnd=null;s.seekAt=position;
    timeline.playhead.style.left=(100*position/duration)+"%";
    if(s.current){s.current="";cards();}
    // Preserve player boxes and trajectory when seeking outside the selected card.
    if(!overlay.source)setOverlaySource({motion_events:[]});
    if(position>=(Number(s.analysisOffset)||0))
      prefetchOverlayAtFrame(Math.floor((position-(Number(s.analysisOffset)||0))*fps()));
    const rally=rallyAtTimelineTime(position);
    ui.viewTitle.textContent=rally?
      "Розыгрыш · "+timeLabel(rally.start_frame/fps())+" – "+
        timeLabel(rally.end_frame/fps()):
      "Просмотр полного видео";
    ui.viewDescription.textContent=rally?
      "Положение на общей шкале. Розыгрыш предварительный; касания не подтверждены.":
      "Просмотр исходного видео вне выделенного розыгрыша.";
    ui.videoTime.textContent=timeLabel(position)+" / "+timeLabel(duration);
    wipe(ui.events);ui.emptyVideo.classList.add("hidden");
    const key=s.project+":"+s.video;
    if(s.videoKey!==key){
      s.videoKey=key;
      loadPreviewVideo();
    }else if(ui.video.readyState>=1){
      try{ui.video.currentTime=position;}catch{}
    }
    drawTimeline();
  };
  const timelinePosition=event=>{
    const box=timeline.track.getBoundingClientRect();
    return box.width?Math.max(0,Math.min(1,(event.clientX-box.left)/box.width))*
      timelineDuration():0;
  };
  timeline.track.addEventListener("pointerdown",event=>{
    if(event.button!==0||(!timeline.data&&!overlay.raw))return;
    timeline.dragging=true;
    timeline.contextPress={x:event.clientX,y:event.clientY,moved:false};
    if(training.active)hideAnnotationContext();
    timeline.track.setPointerCapture?.(event.pointerId);
    seekTimeline(timelinePosition(event));
  });
  timeline.track.addEventListener("pointermove",event=>{
    if(timeline.dragging){
      if(timeline.contextPress&&Math.abs(event.clientX-timeline.contextPress.x)>5)
        timeline.contextPress.moved=true;
      seekTimeline(timelinePosition(event));return;
    }
    const row=rallyAtTimelineTime(timelinePosition(event));
    timeline.track.title=row?"Розыгрыш: "+
      timeLabel(row.start_frame/fps())+" – "+timeLabel(row.end_frame/fps()):
      "Нажмите для перехода к моменту видео";
  });
  timeline.track.addEventListener("pointerup",()=>{
    timeline.dragging=false;timeline.contextPress=null;
  });
  timeline.track.addEventListener("contextmenu",event=>{
    if(!training.active||!s.run)return;
    event.preventDefault();
    let frame=Math.max(0,Math.round((timelinePosition(event)-
      (Number(s.analysisOffset)||0))*fps()));
    if(showBallCandidates()&&comparison.simulated&&
      comparisonMatches(comparison.data)&&comparison.data.mode==="impacts"&&
      !annotation.prediction){
      const candidates=comparison.data.simulated_touch_candidates;
      const tolerance=Math.max(2,Math.ceil(10*timelineDuration()*fps()/
        Math.max(1,timeline.track.getBoundingClientRect().width)));
      let low=0,high=candidates.length;
      while(low<high){
        const mid=(low+high)>>>1;
        if(candidates[mid]<frame)low=mid+1;else high=mid;
      }
      const nearest=[candidates[low-1],candidates[low]]
        .filter(Number.isFinite)
        .sort((a,b)=>Math.abs(a-frame)-Math.abs(b-frame))[0];
      if(nearest!==undefined&&Math.abs(nearest-frame)<=tolerance){
        frame=nearest;seekTimeline((Number(s.analysisOffset)||0)+frame/fps());
      }
    }
    showAnnotationContext(frame,event);
  });
  for(const type of ["pointercancel","lostpointercapture"]){
    timeline.track.addEventListener(type,()=>{
      timeline.dragging=false;timeline.contextPress=null;
    });
  }
  timeline.track.addEventListener("keydown",event=>{
    if(training.active&&(event.key==="ContextMenu"||(event.shiftKey&&event.key==="F10"))){
      event.preventDefault();showAnnotationContext(annotationFrame(),{});return;
    }
    const steps={PageDown:-30,PageUp:30};
    let target;
    if(event.key==="Home")target=0;
    else if(event.key==="End")target=timelineDuration();
    else if(Object.hasOwn(steps,event.key))
      target=(Number(ui.video.currentTime)||s.seekAt)+
        steps[event.key]*(event.shiftKey?0.2:1);
    else return;
    event.preventDefault();seekTimeline(target);
  });
  window.addEventListener("resize",drawTimeline);
  if(typeof ResizeObserver==="function"){
    new ResizeObserver(drawTimeline).observe(timeline.track);
  }
  // Pin the right rally rail to the video's lower edge, excluding the
  // playback controls and timeline. A flexible grid row alone grows with cards.
  const previewLayout=document.querySelector("#studio-preview-view .viewer-layout");
  const previewPanel=previewLayout?.querySelector(".viewer-panel");
  const previewFrame=previewPanel?.querySelector(".video-wrap");
  const previewHead=previewPanel?.querySelector(".viewer-head");
  const previewRail=previewLayout?.querySelector(".rallies-panel");
  const syncPreviewRailHeight=()=>{
    if(!previewPanel||!previewFrame||!previewRail)return;
    if(window.matchMedia("(max-width: 780px)").matches){
      previewRail.style.removeProperty("--preview-rally-height");
      return;
    }
    const panel=previewPanel.getBoundingClientRect();
    const frame=previewFrame.getBoundingClientRect();
    if(!frame.width||!frame.height)return;
    const next=Math.max(1,Math.round(frame.bottom-panel.top))+"px";
    if(previewRail.style.getPropertyValue("--preview-rally-height")!==next)
      previewRail.style.setProperty("--preview-rally-height",next);
  };
  window.addEventListener("resize",syncPreviewRailHeight);
  if(typeof ResizeObserver==="function"&&previewFrame){
    const railObserver=new ResizeObserver(syncPreviewRailHeight);
    railObserver.observe(previewFrame);
    if(previewHead)railObserver.observe(previewHead);
  }
  requestAnimationFrame(syncPreviewRailHeight);
  // Manual selection is stored and ordered inside real Rust/WebAssembly.
  // JavaScript stores only the human-readable labels for selected IDs.
  let wasmReady=false;
  class RustSelection {
    constructor(exports){
      this.exports=exports;
      this.labels=new Map();
      this.encoder=new TextEncoder();
      this.decoder=new TextDecoder();
      if(exports.volley_web_api_version()!==1||
        !exports.memory ||exports.volley_selection_limit()!==80){
        throw Error("Неверная версия модуля выбора фрагментов");
      }
      exports.volley_selection_clear();
    }
    use(id,fn){
      if(!valid(id))return 0;
      const ptr=this.exports.volley_id_buffer_ptr();
      const destination=new Uint8Array(this.exports.memory.buffer,ptr,24);
      destination.set(this.encoder.encode(id));
      return fn();
    }
    get size(){return this.exports.volley_selection_count();}
    has(id){return this.use(id,()=>this.exports.volley_selection_contains())===1;}
    set(id,label){
      const rc=this.use(id,()=>this.exports.volley_selection_add());
      if(rc!==1)throw Error(rc===-1?
        "Максимум 80 розыгрышей для одного монтажа":"Неверный ID розыгрыша");
      this.labels.set(id,label);return this;
    }
    delete(id){
      const found=this.has(id);
      this.use(id,()=>this.exports.volley_selection_remove());
      this.labels.delete(id);return found;
    }
    clear(){this.exports.volley_selection_clear();this.labels.clear();}
    keys(){
      const ids=[];
      for(let i=0;i<this.size;i++){
        if(this.exports.volley_selection_at(i)!==1)throw Error("Ошибка чтения выбранных фрагментов");
        const ptr=this.exports.volley_id_buffer_ptr();
        const text=this.decoder.decode(new Uint8Array(this.exports.memory.buffer,ptr,24));
        ids.push(text);
      }
      return ids.values();
    }
    values(){return Array.from(this.keys(),id=>this.labels.get(id)).values();}
  }
  const loadRustSelection=async()=>{
    await window.VolleyballUi.ready;
    s.selected=new RustSelection(window.VolleyballUi.engine);
    wasmReady=true;syncFavoritesSelection();cards();
  };
  const clearVideo=()=>{
    previewLoad.epoch++;previewLoad.key="";previewLoad.pending=false;previewLoad.retries=0;
    if(previewLoad.timer)clearTimeout(previewLoad.timer);
    previewLoad.timer=null;
    if(window.VolleyballBrowserVideo?.cancel)window.VolleyballBrowserVideo.cancel(ui.video);
    else{ui.video.pause();ui.video.removeAttribute("src");ui.video.load();}
    s.videoKey="";s.current="";s.stopAt=null;s.clipStart=null;s.clipEnd=null;s.seekAt=0;
    setOverlaySource(null);
    ui.emptyVideo.classList.remove("hidden");
    ui.viewTitle.textContent="Розыгрыш не выбран";
    ui.viewDescription.textContent="Откройте розыгрыш для просмотра.";
    ui.videoTime.textContent="—";wipe(ui.events);
    updateTimelineProgress();
  };
  const stopPoll=()=>{if(s.timer)clearInterval(s.timer);s.timer=null;};
  // Persisted favorites are the only source of montage candidates. WebAssembly
  // remains the bounded (80 IDs) source-order selection engine, never a second
  // user-facing toggle. A timeline index supplies labels for all preview pages.
  const syncFavoritesSelection=()=>{
    if(!wasmReady)return;
    s.selected.clear();
    if(s.marksLoaded&&s.run&&s.gold.size<=80){
      const all=timeline.data?.project_id===s.project&&
        timeline.data?.rally_run_id===s.run?timeline.data.rallies:[];
      const indexed=new Map(all.map(rally=>[rally.candidate_id,rally]));
      const page=new Map(s.items.map(item=>[item.candidate_id,item]));
      const start=id=>Number(indexed.get(id)?.start_frame??
        page.get(id)?.detected_start_frame??NaN);
      const ordered=window.VolleyStudioChain.call(2,[...s.gold].map(id=>({id,start:Number.isFinite(start(id))?start(id):null})));
      for(const id of ordered){
        const rally=indexed.get(id),item=page.get(id);
        const first=start(id);
        const last=Number(rally?.end_frame??item?.detected_end_frame??NaN);
        s.selected.set(id,{
          time:Number.isFinite(first)?clock(first):"ID "+id.slice(0,8),
          duration:Number.isFinite(first)&&Number.isFinite(last)?
            (Math.max(0,last-first+1)/fps()).toFixed(1)+" с":"—"
        });
      }
    }
    renderSelection();
  };
  const renderSelection=()=>{
    wipe(ui.picked);
    const count=s.marksLoaded?s.gold.size:0;
    ui.selectedCount.textContent=count+" / 80";
    for(const item of s.selected.values()){
      ui.picked.append(make("span","picked",item.time+" · "+item.duration));
    }
    ui.selectionText.textContent=!s.marksLoaded?
      "Загрузка сохранённого избранного…":
      count>80?
        "В избранном "+count+" розыгрышей. Для одного монтажа допускается не больше 80 — уберите лишние отметки.":
      count?
        "В избранном "+count+" розыгрышей. Добавьте нужные эпизоды в монтажную ленту и настройте порядок.":
        "Отмечайте розыгрыши как «Избранное» в предпросмотре — они появятся здесь для монтажа.";
    const draft=window.VolleyballMontageDraft?.ids(s.project,s.run)||[];
    ui.render.disabled=!wasmReady||s.busy||!s.run||!s.marksLoaded||
      count===0||count>80||s.selected.size!==count||
      draft.length===0||draft.some(id=>!s.selected.has(id));
    const material=timeline.data?.project_id===s.project&&
      timeline.data?.rally_run_id===s.run?timeline.data:null;
    const verified=new Map((Array.isArray(material?.montage_plan)?material.montage_plan:[])
      .filter(item=>typeof item.candidate_id==="string")
      .map(item=>[item.candidate_id,item]));
    const items=[...s.selected.keys()].map(id=>{
      const clip=verified.get(id);
      const start=Number(clip?.start_sec),end=Number(clip?.end_sec);
      if(!clip||!Number.isFinite(start)||!Number.isFinite(end)||
        start<0||end<=start||clip.video!==s.video)return null;
      const label=s.selected.labels?.get(id);
      return {id,video:clip.video,start,end,
        time:label?.time||"Розыгрыш "+id.slice(0,8),
        duration:(end-start).toFixed(1)+" с"};
    }).filter(Boolean);
    window.dispatchEvent(new CustomEvent("volley:montage-favorites",{
      detail:{project_id:s.project,rally_run_id:s.run,ready:Boolean(material),
        video:s.video,items}
    }));
  };
  window.addEventListener("volley:montage-draft-change",()=>{
    if(wasmReady)renderSelection();
  });
  const reset=()=>{
    s.generation++;s.run="";s.video="";s.items=[];s.total=0;s.filtered=0;s.offset=0;s.analysisOffset=0;
    s.loadingMore=false;s.canLoadMore=true;s.pageLoading=false;s.pageError="";s.fps=0;
    s.gold=new Set();s.revision="absent";s.writable=false;s.marksLoaded=false;s.selected.clear();
    s.jobId="";stopPoll();clearVideo();clearTimeline();wipe(ui.list);
    ui.job.textContent="";ui.cancel.classList.add("hidden");
    ui.download.classList.add("hidden");renderSelection();
  };
  const options=(select,rows,value,label,placeholder,allowEmpty=true)=>{
    wipe(select);
    // A project with saved primary analyses must never offer a blank selection.
    if(allowEmpty||!rows.length){
      const blank=make("option","",placeholder);blank.value="";select.append(blank);
    }
    for(const item of rows){
      const entry=make("option","",label(item));entry.value=String(item[value]);
      select.append(entry);
    }
  };
  // The linked source chain is project -> GPU analysis (video + lineage) -> CPU snapshot.
  // Preserve selections on refresh; reset only dependent selections when an upstream
  // source changes. Stale async responses must never reselect a different project.
  const setRuns=async(preferred="")=>{
    reset();
    const source=s.chain?.choices.find(row=>row.value===s.analysis);
    if(source?.video){
      s.video=source.video;
      s.fps=30;
      if(overlay.raw)setOverlaySource({});
    }
    const rows=s.chain?.byAnalysis.get(s.analysis)||[];
    options(ui.run,rows,"rally_run_id",row=>
      (window.VolleyStudioChain.stamp(row.created_at)||"CPU "+row.rally_run_id.slice(0,8))+
      " · "+(row.display_name||((row.candidate_count||0)+" розыгрышей")),
      !s.analysis?"Сначала выберите анализ":
        rows.length?"Выберите постобработку":"Нет CPU-версий для анализа");
    ui.run.disabled=!s.analysis||!rows.length;
    ui.run.value=window.VolleyStudioChain.latestRun(rows,preferred);
    refreshPicker(ui.run);
    s.run=ui.run.value;
    s.video=rows.find(row=>row.rally_run_id===s.run)?.video||source?.video||"";
    if(s.video){s.fps=30;ensurePreviewVideo();}
    // Overlay playback follows the selected CPU snapshot, not a rally card.
    // reset() clears the canvas source; restore it for full-video preview too.
    setOverlaySource(overlay.raw||s.run?{}:null);
    updateRawDetectorLabel();
    if(s.run&&s.video)
      prefetchOverlayAtFrame(Math.max(0,Math.floor(((ui.video.currentTime||0)-(Number(s.analysisOffset)||0))*fps())));
    if(s.analysis)sourceSelection.save(s.project,s.analysis,s.run);
    window.dispatchEvent(new CustomEvent("volley:postprocessing-change",{
      detail:{project_id:s.project,rally_run_id:s.run}
    }));
    // SHA-verifying a long Touch file may take seconds. Never block the
    // already-resolved header and source chain on that independent operation.
    if(s.run){
      void loadPage().catch(failed);
      void loadTimeline().catch(failed);
    }
    else message(!s.analysis?"Выберите первичный GPU-анализ.":
      overlay.raw&&rawFusion()?"Сырые GPU-данные доступны без CPU-постобработки.":
      rows.length?"Выберите версию постобработки.":
      "У выбранного анализа пока нет связанной CPU-постобработки.");
    if(overlay.raw&&rawFusion())prefetchOverlayAtFrame(0);
  };
  const setAnalysis=async(preferredRun="")=>{
    s.analysis=ui.analysis.value;
    const source=s.chain?.choices.find(row=>row.value===s.analysis);
    window.dispatchEvent(new CustomEvent("volley:chain-change",{
      detail:{project_id:s.project,video:source?.video||"",analysis:s.analysis}
    }));
    await setRuns(preferredRun);
  };
  const selectReferenceEntry=async entry=>{
    if(!reference.active||!entry||annotation.dirty)return;
    const choice=s.chain?.choices.find(candidate=>candidate.video===entry.video&&
      (s.chain.byAnalysis.get(candidate.value)||[]).some(run=>run.rally_run_id===entry.rally_run_id));
    if(!choice){
      reference.counts.textContent="У эталона больше нет связанной CPU-версии";
      return;
    }
    if(s.run!==entry.rally_run_id||s.video!==entry.video){
      ui.analysis.value=choice.value;
      await setAnalysis(entry.rally_run_id);
    }
    if(!reference.active||s.video!==entry.video||s.run!==entry.rally_run_id||
      reference.video.value!==entry.rally_run_id)return;
    reference.video.value=entry.rally_run_id;
    reference.counts.textContent="Загрузка проверенной разметки…";
    await refreshAnnotation();
  };
  const refreshReferenceCatalog=async()=>{
    if(!reference.active)return;
    const epoch=++reference.epoch,project=s.project;
    reference.video.disabled=true;
    reference.counts.textContent="Поиск проверенных видео…";
    reference.entries=[];
    wipe(reference.video);
    try{
      const result=await request(url("/api/training/annotations/catalog",{project_id:project}));
      if(epoch!==reference.epoch||!reference.active||project!==s.project)return;
      if(result.schema!=="volley-touch-annotation-catalog/1"||result.project_id!==project||
        !Array.isArray(result.videos))throw Error("Неверный формат каталога эталонов");
      reference.entries=result.videos.filter(item=>
        typeof item.video==="string"&&typeof item.rally_run_id==="string"&&
        s.projectVideos.includes(item.video)&&s.chain?.choices.some(choice=>
          choice.video===item.video&&
          (s.chain.byAnalysis.get(choice.value)||[]).some(run=>
            run.rally_run_id===item.rally_run_id)));
      for(const item of reference.entries){
        const name=item.video.split(/[\\/]/).pop()||item.video;
        const option=make("option","",name);
        option.value=item.rally_run_id;
        reference.video.append(option);
      }
      reference.video.disabled=!reference.entries.length;
      if(!reference.entries.length){
        const option=make("option","","Нет проверенных видео");
        option.value="";reference.video.append(option);
        reference.counts.textContent="В этом проекте пока нет проверенных эталонов";
        cards();return;
      }
      const current=reference.entries.find(item=>item.rally_run_id===s.run&&item.video===s.video);
      const selected=current||reference.entries[0];
      reference.video.value=selected.rally_run_id;
      await selectReferenceEntry(selected);
    }catch(error){
      if(epoch!==reference.epoch||!reference.active)return;
      reference.video.disabled=true;
      const option=make("option","","Эталоны недоступны");option.value="";
      reference.video.append(option);
      reference.counts.textContent="Ошибка загрузки: "+error.message;
      cards();
    }
  };
  const chainSignature=chain=>JSON.stringify([
    chain.choices,[...chain.byAnalysis].map(([key,rows])=>[key,rows])
  ]);
  const newestPrimaryAnalysis=choices=>{
    // Only real GPU sources take precedence over legacy orphan CPU snapshots.
    // Stage listings are sorted separately per video, not across the project.
    const primary=choices.filter(row=>row.kind!=="unlinked");
    const candidates=primary.length?primary:choices;
    return [...candidates].sort((a,b)=>
      (Number(b.created_at)||0)-(Number(a.created_at)||0)||
      String(b.id||"").localeCompare(String(a.id||"")))[0]?.value||"";
  };
  const renderChain=async(chain,preferredAnalysis="",preferredRun="",retainCurrent=false)=>{
    const unchanged=retainCurrent&&s.analysis===preferredAnalysis&&s.run===preferredRun&&
      chain.choices.some(row=>row.value===s.analysis)&&
      (!s.run||(chain.byAnalysis.get(s.analysis)||[]).some(row=>row.rally_run_id===s.run));
    s.chain=chain;
    options(ui.analysis,chain.choices,"value",item=>item.label,
      chain.choices.length?"Выберите 1 фазу":"Нет результатов 1 фазы",
      !chain.choices.length);
    ui.analysis.disabled=!chain.choices.length;
    // Restore a valid explicit choice; otherwise select the most recently
    // created GPU analysis, including after the selected version is deleted.
    ui.analysis.value=chain.choices.some(row=>row.value===preferredAnalysis)?
      preferredAnalysis:newestPrimaryAnalysis(chain.choices);
    refreshPicker(ui.analysis);
    if(unchanged){
      // Only labels and available versions changed. Do not discard the active
      // preview, playback position, marks or in-flight requests.
      const rows=chain.byAnalysis.get(s.analysis)||[];
      options(ui.run,rows,"rally_run_id",row=>
        (window.VolleyStudioChain.stamp(row.created_at)||"CPU "+row.rally_run_id.slice(0,8))+
        " · "+(row.display_name||((row.candidate_count||0)+" розыгрышей")),
        rows.length?"Выберите постобработку":"Нет CPU-версий для анализа");
      ui.run.disabled=!rows.length;
      ui.run.value=s.run;
      refreshPicker(ui.run);
      return;
    }
    // If the selected source was deleted (including dependent data), switch
    // to the remaining available versions using the normal reset flow.
    await setAnalysis(preferredRun);
  };
  const loadChain=async(preferredAnalysis="",preferredRun="",retainCurrent=false,freshOnly=false)=>{
    const token=++s.chainEpoch,project=s.project;
    if(!retainCurrent){
      reset();
      s.analysis="";s.chain=null;
      options(ui.analysis,[],"value",x=>x,"Загрузка анализов…");
      options(ui.run,[],"rally_run_id",x=>x,"Сначала выберите анализ");
      ui.analysis.disabled=true;ui.run.disabled=true;
      refreshPicker(ui.analysis);refreshPicker(ui.run);
    }
    await window.VolleyballUi?.ready;
    if(token!==s.chainEpoch||s.project!==project)return;
    if(!project){
      options(ui.analysis,[],"value",x=>x,"Сначала выберите проект");
      message("Выберите проект для просмотра цепочки обработок.");
      return;
    }
    const videos=s.projectVideos.slice(0,40);
    const cached=freshOnly?null:chainSnapshot.read(project,videos);
    if(cached){
      await renderChain(cached,retainCurrent?s.analysis:preferredAnalysis,
        retainCurrent?s.run:preferredRun,retainCurrent);
      if(token!==s.chainEpoch||s.project!==project)return;
    }else{
      message("Первичная загрузка списка GPU- и CPU-версий…");
    }
    try{
      const runs=await previewCache.load(
        url("/api/cpu/native/rallies/runs",{project_id:project}),300000,true,true);
      if(token!==s.chainEpoch||s.project!==project)return;
      const overviews=new Map();
      let unavailable=0;
      // Index requests are metadata-only, with a maximum of four in flight.
      for(let offset=0;offset<videos.length;offset+=4){
        const batch=await Promise.all(videos.slice(offset,offset+4).map(async video=>{
          try{
            const value=await previewCache.load(url("/api/studio/analysis-overview",{
              project_id:project,video
            }),300000,true,true);
            if(value.schema!=="volley-native-analysis-overview/1"||
               value.project_id!==project||value.video!==video||value.metadata_only!==true)
              throw Error("Несовместимая версия индекса анализа");
            return [video,value];
          }catch{return [video,null];}
        }));
        if(token!==s.chainEpoch||s.project!==project)return;
        for(const [video,value] of batch){
          if(value)overviews.set(video,value);else unavailable++;
        }
      }
      if(token!==s.chainEpoch||s.project!==project)return;
      const builder=window.VolleyStudioChain;
      if(!builder?.build)throw Error("Не удалось загрузить связанную цепочку данных");
      if(unavailable){
        const error=Error("Не удалось загрузить полный индекс стадий анализа");
        if(cached){message(error.message,"error");return;}
        throw error;
      }
      const fresh=builder.build(videos,overviews,runs.runs);
      chainSnapshot.save(project,videos,overviews,runs.runs);
      if(!s.chain||chainSignature(s.chain)!==chainSignature(fresh)){
        // Preserve edits to the current selection made while revalidating.
        await renderChain(fresh,retainCurrent?s.analysis:
          cached?(s.analysis||preferredAnalysis):preferredAnalysis,
          retainCurrent?s.run:cached?(s.run||preferredRun):preferredRun,
          retainCurrent);
      }
    }catch(error){
      if(token!==s.chainEpoch||s.project!==project)return;
      if(!cached)throw error;
      console.warn("Обновление индекса обработок отложено:",error);
    }
  };
  const projects=async(preferredProject="")=>{
    const prev=s.project,selectedAnalysis=s.analysis,selectedRun=s.run;
    const stored=sourceSelection.read();
    const epoch=++s.catalogEpoch;
    const raw=await window.VolleyballStudioProjectCatalog();
    if(epoch!==s.catalogEpoch)return;
    const rows=(Array.isArray(raw.projects)?raw.projects:[]).
      filter(row=>typeof row.id==="string");
    s.projectCatalog=rows;
    options(ui.project,rows,"id",row=>row.name||row.title||row.id,"Выберите проект");
    ui.project.value=rows.some(row=>row.id===preferredProject)?preferredProject:
      rows.some(row=>row.id===prev)?prev:
      rows.some(row=>row.id===stored.project)?stored.project:
      rows.length?rows[rows.length-1].id:"";
    refreshPicker(ui.project);
    s.project=ui.project.value;
    const current=rows.find(row=>row.id===s.project);
    s.projectVideos=Array.isArray(current?.videos)?
      current.videos.filter(video=>typeof video==="string"):[];
    const loading=loadChain(s.project===prev?selectedAnalysis:
      s.project===stored.project?stored.analysis:"",
      s.project===prev?selectedRun:
      s.project===stored.project?stored.run:"",false,!!prev&&s.project!==prev);
    window.dispatchEvent(new Event("volley:project-ready"));
    await loading;
    if(epoch!==s.catalogEpoch)return;
    if(reference.active)void refreshReferenceCatalog();
  };
  const loadMarks=async(token)=>{
    const raw=await previewCache.load(url("/api/cpu/native/rallies/marks",{
      project_id:s.project,rally_run_id:s.run
    }),7000);
    if(token!==s.generation)return;
    s.gold=new Set((Array.isArray(raw.candidate_ids)?raw.candidate_ids:[]).filter(valid));
    s.revision=typeof raw.revision==="string"?raw.revision:"absent";
    s.writable=raw.writable===true;s.marksLoaded=true;
    syncFavoritesSelection();drawTimeline();
  };
  const rallyListQuery=()=>{
    const choice=ui.filter.value;
    return {
      gold_filter:choice==="marked"||choice==="unmarked"?choice:"all",
      sort_by:choice==="duration_desc"||choice==="touches_desc"?choice:"time"
    };
  };
  const loadPage=async()=>{
    if(!s.project||!s.run)return;
    s.offset=0;s.loadingMore=false;s.canLoadMore=true;
    s.pageLoading=true;s.pageError="";
    const token=++s.generation;
    const path=url("/api/cpu/native/rallies/preview",{
      project_id:s.project,rally_run_id:s.run,offset:s.offset,limit:s.limit,
      ...rallyListQuery()
    });
    // Reopening a tab reuses the rendered list and the cached page silently.
    if(previewCache.peek(path)===null)
      ui.list.replaceChildren(make("div","empty-state","Проверка источников и загрузка розыгрышей…"));
    try{
      // Bookmarks have a separate revision and a separate metadata request.
      // A slow or unavailable bookmark endpoint must never block cached rally
      // cards. Apply the marks as soon as they arrive, without reloading video.
      void loadMarks(token).then(()=>{
        if(token===s.generation){cards();drawTimeline();}
      }).catch(error=>{
        if(token===s.generation)
          message("Избранное недоступно: "+String(error?.message||error),"error");
      });
      const raw=await previewCache.load(path,900000,true);
      if(token!==s.generation)return;
      if(raw.schema!=="volley-native-rally-preview/1"||
        !validSurfaceContacts(raw)||raw.authoritative_game_events!==false||
        raw.automatic_montage_eligible!==false||
        !Array.isArray(raw.items))
        throw Error("Неподдерживаемая версия предварительного CPU-анализа");
      s.pageLoading=false;
      s.fps=Number(raw.fps)||0;
      s.items=Array.isArray(raw.items)?raw.items:[];
      s.analysisOffset=Number(raw.analysis_offset_sec)||0;
      s.total=Number(raw.total_candidates)||0;
      s.filtered=Number(raw.filtered_candidates??s.total)||0;
      s.canLoadMore=s.items.length<s.filtered;
      s.video=typeof raw.video==="string"?raw.video:"";
      ensurePreviewVideo();
      if(typeof raw.source_replay_sha256==="string")requestOverlayLabels(s.project,s.run,s.video,raw.source_replay_sha256);
      if(s.items.length&&previewIsVisible()){
        const first=Number(s.items[0].detected_start_frame);
        if(Number.isSafeInteger(first)&&first>=0)
          prefetchOverlayAtFrame(first);
      }
      syncFavoritesSelection();cards();
      if(!$("studio-preview-view").classList.contains("hidden"))
        message("CPU-данные загружены. Розыгрыши предварительные; проверьте их в видеоплеере.","ok");
    }catch(e){
      if(token!==s.generation)return;
      s.pageLoading=false;
      s.pageError=e instanceof Error?e.message:String(e);
      s.items=[];s.total=0;s.canLoadMore=false;cards();
      failed(e);
    }
  };
  const toggleGold=async id=>{
    if(!valid(id))return;
    if(!s.writable){
      message("Избранное можно сохранять только в рабочем хранилище.","error");return;
    }
    if(s.busy)return;
    s.busy=true;
    const marking=!s.gold.has(id);
    const run=s.run,project=s.project;
    try{
      const raw=await post("/api/cpu/native/rallies/marks",{
        project_id:project,rally_run_id:run,candidate_id:id,
        marked:marking,expected_revision:s.revision
      });
      if(run!==s.run||project!==s.project)return;
      invalidatePreviewRun(project,run);
      previewCache.drop(path=>path.startsWith("/api/cpu/native/rallies/marks?"));
      s.gold=new Set((Array.isArray(raw.candidate_ids)?raw.candidate_ids:[]).filter(valid));
      s.revision=String(raw.revision);s.marksLoaded=true;
      syncFavoritesSelection();drawTimeline();
      if(ui.filter.value==="marked"||ui.filter.value==="unmarked"){
        await loadPage();
      }else cards();
      message(marking?"Добавлено в избранное.":"Удалено из избранного.","ok");
    }catch(e){
      try{
        previewCache.drop(path=>path.startsWith("/api/cpu/native/rallies/marks?"));
        await loadMarks(s.generation);cards();
      }catch{}
      failed(e);
    }finally{s.busy=false;renderSelection();cards();}
  };
  // The *complete* server-side counts, not the 14-event card preview,
  // determine which action types are shown on a rally card.
  const estimatedTypes=item=>{
    const labels={attack:"Атака",block:"Блок",serve:"Подача",
      overhead_pass:"Передача сверху",dig:"Приём снизу"};
    const counts=item?.estimated_action_counts;
    const known=counts&&typeof counts==="object"&&!Array.isArray(counts)?
      Object.entries(labels).filter(([kind])=>
        Number.isSafeInteger(counts[kind])&&counts[kind]>0)
        .map(([kind,label])=>label+" ×"+counts[kind]):[];
    const unknown=(Number(item?.unclassified_action_count)||0)+
      (Number(counts?.low_touch)||0);
    if(Number.isSafeInteger(unknown)&&unknown>0)
      known.push("Касание ×"+unknown);
    return known.length?"≈ "+known.join(" · "):"Типы не определены";
  };
  // Card timelines use the same authoritative CPU Touchs and palette as the video.
  // The short motion_events preview is not a source of contact markers.
  // Preview follows the highlighted detection on the full-video timeline.
  // manual_clip carries deliberate montage padding and must not shift playback.
  const rallyClipFrames=item=>({
    start:Number(item.detected_start_frame??item.start_frame??
      item.manual_clip?.start_frame??0),
    end:Number(item.detected_end_frame??item.end_frame??
      item.manual_clip?.end_frame??0)
  });
  const refreshRallyCardTouchTicks=()=>{
    const data=timeline.data;
    const validTimeline=Boolean(data)&&data.project_id===s.project&&
      data.rally_run_id===s.run&&data.video===s.video&&
      Array.isArray(data.touch_annotations);
    const serves=validTimeline&&!overlay.raw&&overlay.flags.touches&&
      overlay.actionFlags.serve&&Array.isArray(data.serve_proposals)?
      data.serve_proposals:[];
    const touches=validTimeline&&(typeof training==="undefined"||!training.active)&&
      !overlay.raw&&overlay.flags.touches?data.touch_annotations.filter(event=>
      overlay.actionFlags[cpuTouchKind(event.kind)]):[];
    const ground=validTimeline&&!overlay.raw&&overlay.flags.touches?floorContacts(data):[];
    for(const {progress,markers,start,end} of rallyCardProgresses.values()){
      wipe(markers);
      if(!validTimeline){
        progress.setAttribute("aria-label","Шкала розыгрыша. Ожидание касаний CPU");
        continue;
      }
      const duration=Math.max(1,end-start+1);
      let count=0;
      for(const event of touches){
        if(event.frame_index<start||event.frame_index>end)continue;
        const [color,label]=cpuTouchStyle[cpuTouchKind(event.kind)];
        const tick=make("span","rally-card-touch-tick");
        tick.style.left=(100*(event.frame_index-start)/duration)+"%";
        tick.style.backgroundColor=color;
        tick.style.color=color;
        tick.title=second(event.frame_index)+" с · "+label.replace(/^≈ /,"");
        tick.setAttribute("aria-hidden","true");
        markers.append(tick);
        count++;
      }
      for(const event of ground){
        if(event.frame_index<start||event.frame_index>end)continue;
        const tick=make("span","rally-card-touch-tick");
        tick.style.left=(100*(event.frame_index-start)/duration)+"%";
        tick.style.backgroundColor=floorContactColor;
        tick.style.color=floorContactColor;
        tick.title=second(event.frame_index)+" с · "+floorContactLabel(event.outcome);
        tick.setAttribute("aria-hidden","true");
        markers.append(tick);
      }
      for(const serve of serves){
        const frame=Number(serve.trajectory_start_frame??serve.rally_start_frame);
        if(!Number.isSafeInteger(frame)||frame<start||frame>end)continue;
        const tick=make("span","rally-card-touch-tick");
        tick.style.left=(100*(frame-start)/duration)+"%";
        tick.style.backgroundColor="#f3a350";
        tick.style.color="#f3a350";
        tick.title=second(frame)+" с · ≈ Подача"+
          (Number.isSafeInteger(serve.provisional_track_id)?
            " · Track ID "+serve.provisional_track_id:" · игрок пока не определён");
        tick.setAttribute("aria-hidden","true");
        markers.append(tick);
      }
      progress.setAttribute("aria-label",
        "Выбрать момент воспроизведения · "+count+" касаний из CPU-постобработки");
    }
  };
  const updateRallyCardProgress=()=>{
    const active=rallyCardProgresses.get(s.current);
    if(!active)return;
    const {progress,start,end}=active;
    const current=s.videoKey===s.project+":"+s.video&&
      Number.isFinite(ui.video.currentTime)?
        (Number.isFinite(overlay.presentedTime)&&!ui.video.paused?
          overlay.presentedTime:ui.video.currentTime):s.seekAt;
    const fraction=(current-(Number(s.analysisOffset)||0)-start/fps())/(Math.max(1,end-start+1)/fps());
    progress.style.setProperty("--rally-progress",
      Math.max(0,Math.min(100,fraction*100))+"%");
    const relative=Math.max(0,Math.min((end-start+1)/fps(),current-(Number(s.analysisOffset)||0)-start/fps()));
    progress.setAttribute("aria-valuenow",relative.toFixed(1));
    progress.setAttribute("aria-valuetext",
      second(Math.max(start,Math.min(end,(current-(Number(s.analysisOffset)||0))*fps()))+(Number(s.analysisOffset)||0)*fps())+" с на видео");
  };
  // Event pills and summary are derived from the same selected postprocessing
  // timeline as the canvas. The 14-event card sample is never used here.
  const renderRallyTouches=(item,start,end)=>{
    const data=timeline.data;
    const valid=data?.project_id===s.project&&data?.rally_run_id===s.run&&
      data?.video===s.video&&Array.isArray(data.touch_annotations);
    wipe(ui.events);
    if(training.active){
      ui.viewDescription.textContent="Дообучение: только кандидаты по мячу; типы не рассчитаны.";
      return;
    }
    if(!valid){
      ui.viewDescription.textContent=
        "Загрузка касаний выбранной CPU-постобработки…";
      return;
    }
    const serves=Array.isArray(data.serve_proposals)?data.serve_proposals.filter(item=>{
      const frame=Number(item.trajectory_start_frame??item.rally_start_frame);
      return Number.isSafeInteger(frame)&&frame>=start&&frame<=end;
    }):[];
    const touches=data.touch_annotations.filter(event=>
      event.frame_index>=start&&event.frame_index<=end);
    const ground=floorContacts(data).filter(event=>event.frame_index>=start&&event.frame_index<=end);
    const labels={attack:"Атака",block:"Блок",serve:"Подача",
      overhead_pass:"Передача сверху",dig:"Приём снизу"};
    const counts=new Map();
    for(const touch of touches){
      const kind=cpuTouchKind(touch.kind);
      counts.set(kind,(counts.get(kind)||0)+1);
    }
    const breakdown=Object.entries(labels)
      .filter(([kind])=>counts.has(kind))
      .map(([kind,label])=>label+" ×"+counts.get(kind));
    if(counts.has("unknown_touch"))
      breakdown.push("Касание ×"+counts.get("unknown_touch"));
    ui.viewDescription.textContent="Фрагмент "+clock(start)+" – "+clock(end)+
      ". Касаний из CPU-постобработки: "+touches.length+
      ". Касаний пола: "+ground.length+
      ". Оценка типов: "+(breakdown.join(" · ")||"нет данных")+
      ". Классификация автоматическая, не подтверждена вручную.";
    for(const serve of serves){
      const frame=Number(serve.trajectory_start_frame??serve.rally_start_frame);
      const pill=make("span","event-pill",clock(frame)+" · ≈ Подача");
      pill.title="Начало траектории розыгрыша. "+
        (Number.isSafeInteger(serve.provisional_track_id)?
          "Кандидат Track ID "+serve.provisional_track_id:
          "Подающий не идентифицирован")+". Не подтверждено вручную.";
      pill.style.borderColor="#f3a350";
      ui.events.append(pill);
    }
    for(const touch of touches.slice(0,32)){
      const kind=cpuTouchKind(touch.kind);
      const label=kind!=="unknown_touch"?"≈ "+labels[kind]:"Касание";
      const pill=make("span","event-pill",clock(touch.frame_index)+" · "+label);
      pill.title="Результат выбранной CPU-постобработки. Автоматическая оценка.";
      pill.style.borderColor=cpuTouchStyle[kind][0];
      ui.events.append(pill);
    }
    for(const event of ground){
      const pill=make("span","event-pill",clock(event.frame_index)+" · "+floorContactLabel(event.outcome));
      pill.title="Предполагаемый контакт с поверхностью по траектории мяча. Не решение судьи.";
      pill.style.borderColor=floorContactColor;
      ui.events.append(pill);
    }
  };
  const view=(item,requestedTime=null)=>{
    if(!s.project||!s.video)return;
    if(s.current!==item.candidate_id){
      s.current=item.candidate_id;cards();
    }
    // Full timeline entries may not belong to the currently loaded (or sorted) card page.
    const {start,end}=rallyClipFrames(item);
    const key=s.project+":"+s.video;
    s.clipStart=(Number(s.analysisOffset)||0)+start/fps();
    s.clipEnd=(Number(s.analysisOffset)||0)+(end+1)/fps();
    s.stopAt=s.playbackMode==="free"?null:s.clipEnd;
    ui.viewTitle.textContent="Розыгрыш · "+clock(s.clipStart*fps())+" – "+clock(s.clipEnd*fps());
    ui.videoTime.textContent=clock(s.clipStart*fps())+" – "+clock(s.clipEnd*fps());
    if(item.ballRally===true||item.trainingMerged===true){
      wipe(ui.events);
      ui.viewDescription.textContent=item.ballRally===true?
        "Предварительный розыгрыш по траектории мяча. Касания и игроки не подтверждены.":
        "Тестовое объединение CPU-интервалов. Касания и игроки не пересчитывались.";
    }else renderRallyTouches(item,start,end);
    ui.emptyVideo.classList.add("hidden");
    // Direct seeking on the card must not jump back to the rally start.
    s.seekAt=Number.isFinite(requestedTime)?
      Math.max(s.clipStart,Math.min(s.clipEnd-1/fps(),requestedTime)):s.clipStart;
    rallyCardProgresses.get(s.current)?.progress.style.setProperty("--rally-progress",
      (100*(s.seekAt-s.clipStart)/(s.clipEnd-s.clipStart))+"%");
    overlay.presentedTime=null;
    setOverlaySource(item);
    prefetchOverlayAtFrame(Math.max(0,Math.floor((s.seekAt-(Number(s.analysisOffset)||0))*fps())));
    let deferredPlay=false;
    if(s.videoKey!==key){
      s.videoKey=key;
      loadPreviewVideo(previewIsVisible());
      deferredPlay=true;
    }else if(ui.video.readyState>=1){
      try{ui.video.currentTime=s.seekAt;}catch{}
    }
    // Draw only after seeking; otherwise the playhead temporarily shows the
    // previously played position instead of the selected rally start.
    drawTimeline();
    // Metadata/seek is restored by loadedmetadata without background playback.
    if(previewIsVisible()&&!deferredPlay)ui.video.play().catch(error=>
      message("Воспроизведение недоступно: "+error.message,"error"));
  };
  const renderReferenceCards=()=>{
    const chosen=reference.entries.find(item=>item.video===s.video&&
      item.rally_run_id===s.run);
    const data=chosen&&annotationSourceValid()&&annotationReady()&&
      annotation.data.reviewed?annotation.data:null;
    if(!data){
      ui.list.append(make("div","empty-state",!reference.entries.length?
        "В проекте нет доступных проверенных эталонов.":
        "Для выбранного видео пока нет доступной проверенной разметки."));
      return;
    }
    const offset=Number(s.analysisOffset)||0;
    const label=frame=>timeLabel(offset+frame/fps());
    const rallies=annotationSorted("rallies"),touches=annotationSorted("touches");
    reference.counts.textContent="Розыгрыши: "+rallies.length+" · касания: "+touches.length;
    const touchButton=touch=>{
      const button=make("button","studio-reference-touch","● "+label(touch.frame)+" · "+
        (trainingNames[touch.kind]||"Касание"));
      button.type="button";
      button.title="Перейти к размеченному касанию · "+
        (trainingNames[touch.kind]||"тип не определён");
      button.addEventListener("click",event=>{
        event.stopPropagation();
        annotation.touchId=touch.id;
        seekTimeline(offset+touch.frame/fps());
      });
      return button;
    };
    for(const [index,rally] of rallies.entries()){
      const selected=annotation.rallyId===rally.id;
      const card=make("article","rally-card studio-reference-card"+(selected?" is-selected":""));
      card.tabIndex=0;card.setAttribute("role","button");
      card.setAttribute("aria-label","Эталонный розыгрыш "+(index+1)+
        " с "+label(rally.start)+" по "+label(rally.end));
      const jump=()=>{
        annotation.rallyId=rally.id;
        seekTimeline(offset+rally.start/fps());
        cards();
      };
      card.addEventListener("click",event=>{
        if(!event.target.closest("button"))jump();
      });
      card.addEventListener("keydown",event=>{
        if((event.key==="Enter"||event.key===" ")&&event.target===card){
          event.preventDefault();jump();
        }
      });
      const top=make("div","card-top");
      top.append(make("span","id-label","Эталон · №"+(index+1)),
        make("strong","rally-card-time-value",label(rally.start)+"–"+label(rally.end)));
      const hits=touches.filter(touch=>touch.frame>=rally.start&&touch.frame<=rally.end);
      const info=make("div","card-stats");
      info.append(make("span","","Длительность "+((rally.end-rally.start+1)/fps()).toFixed(1)+" с"),
        make("strong","",hits.length+" касаний"));
      const marks=make("div","studio-reference-touches");
      for(const touch of hits)marks.append(touchButton(touch));
      if(!hits.length)marks.append(make("span","card-timespan","Касания не размечены"));
      card.append(top,info,marks);
      ui.list.append(card);
    }
    const unassigned=touches.filter(touch=>
      !rallies.some(rally=>touch.frame>=rally.start&&touch.frame<=rally.end));
    if(unassigned.length){
      const card=make("article","rally-card studio-reference-card");
      card.append(make("strong","id-label","Касания вне розыгрышей"));
      const marks=make("div","studio-reference-touches");
      for(const touch of unassigned)marks.append(touchButton(touch));
      card.append(marks);ui.list.append(card);
    }
    if(!rallies.length&&!unassigned.length)
      ui.list.append(make("div","empty-state","В этом эталоне нет размеченных розыгрышей и касаний."));
  };
  const cards=()=>{
    const previousScroll=ui.list.scrollTop;
    rallyCardProgresses.clear();
    wipe(ui.list);
    const trainingList=typeof training!=="undefined"&&training.active;
    if(trainingList&&reference.active){
      renderReferenceCards();
      ui.list.scrollTop=previousScroll;
      return;
    }
    const cpuItems=trainingList?[]:s.items;
    const ballItems=trainingList&&timelineRallyLayers.ball.checked&&
      comparisonMatches(comparison.data)&&comparison.data?.trajectory_status==="ready"?
      comparison.data.trajectory_rallies:[];
    const groups=trainingList?trainingRallyGroups():null;
    if(groups)updateTrainingSummaries(groups);
    const visible=trainingList?
      (timelineRallyLayers.ball.checked?groups.ball:[]).map(group=>({
        kind:"ball",item:group,start:group.start
      })):
      cpuItems.map(item=>({kind:"touch",item}));
    if(!visible.length){
      const text=trainingList&&!timelineRallyLayers.ball.checked?
        "Включите розыгрыши по траектории мяча.":
        trainingList&&!comparisonMatches(comparison.data)?
          "Розыгрыши по мячу ещё не рассчитаны. Нажмите «Смоделировать розыгрыши».":
        s.pageLoading?"Проверка источников и загрузка розыгрышей…":
        s.pageError?"Предпросмотр не загружен: "+s.pageError+
          " · Выбранная CPU-версия не удалена.":
        !s.run?(overlay.raw&&rawFusion()?
          "Сырые GPU-данные доступны в плеере. Розыгрыши появятся после CPU-постобработки.":
          "Выберите версию CPU-постобработки.") :
        ui.filter.value==="marked"&&s.filtered===0?
          "В этой версии пока нет избранных розыгрышей. Выберите «Все».":
        ui.filter.value==="unmarked"&&s.filtered===0?
          "Все розыгрыши этой версии добавлены в избранное. Выберите «Все».":
        s.total===0?"В выбранной CPU-версии нет обнаруженных розыгрышей.":
        "Не удалось отобразить карточки этой CPU-версии.";
      ui.list.append(make("div","empty-state",text));
      return;
    }
    for(const entry of visible){
      if(entry.kind==="ball"||entry.kind==="merged"){
        const {start,end,members}=entry.item;
        const byBall=entry.kind==="ball";
        const id=(byBall?"ball-":"merged-cpu-")+start+"-"+end;
        const candidate={ballRally:byBall,trainingMerged:!byBall,candidate_id:id,
          detected_start_frame:start,detected_end_frame:end};
        const card=make("article","rally-card "+(byBall?"training-ball":"training-touch")+
          (s.current===id?" active":""));
        card.tabIndex=0;card.setAttribute("role","button");
        const open=()=>view(candidate);
        card.setAttribute("aria-label",(byBall?"Розыгрыш по мячу":"Объединённый CPU-розыгрыш")+
          " · "+second(start+(Number(s.analysisOffset)||0)*fps())+
          "–"+second(end+(Number(s.analysisOffset)||0)*fps())+" с");
        card.addEventListener("click",event=>{
          if(!event.target.closest("button,label,input"))open();
        });
        card.addEventListener("keydown",event=>{
          if((event.key==="Enter"||event.key===" ")&&event.target===card){
            event.preventDefault();open();
          }
        });
        const top=make("div","card-top");
        top.append(make("span","id-label",byBall?"По мячу":"По касаниям"));
        const when=make("div","rally-card-time-range");
        when.append(make("strong","rally-card-time-value",
          second(start+(Number(s.analysisOffset)||0)*fps())+"–"+
          second(end+(Number(s.analysisOffset)||0)*fps())+" с"));
        top.append(when);
        const foot=make("div","card-stats");
        const duration=make("div","rally-card-duration");
        duration.append(make("span","","Длительность"),
          make("strong","",((end-start+1)/fps()).toFixed(1)+" с"));
        foot.append(duration);
        card.append(top,foot,make("div","rally-card-techniques",
          members.length>1?"Объединено интервалов: "+members.length:
          byBall?"Предварительный интервал · касания и игроки не подтверждены":
          "Интервал из CPU-индекса · касания и игроки не пересчитаны"));
        ui.list.append(card);
        continue;
      }
      const item=entry.item;
      const id=item.candidate_id;
      if(!valid(id))continue;
      const gold=s.gold.has(id);
      const card=make("article","rally-card"+(gold?" gold":"")+
        (s.current===id?" active":"")+(trainingList?" training-touch":""));
      card.tabIndex=0;card.setAttribute("role","button");
      card.setAttribute("aria-label","Просмотр розыгрыша "+clock(item.detected_start_frame+(Number(s.analysisOffset)||0)*fps()));
      card.addEventListener("click",e=>{
        if(!e.target.closest(".rally-card-progress,button,label,input"))view(item);
      });
      card.addEventListener("keydown",e=>{
        if((e.key==="Enter"||e.key===" ")&&e.target===card){
          e.preventDefault();view(item);
        }
      });
      card.addEventListener("contextmenu",e=>{e.preventDefault();toggleGold(id);});
      const top=make("div","card-top");
      const idLabel=make("span","id-label","ID "+id.slice(0,6));
      idLabel.title="Полный ID: "+id;
      if(trainingList)idLabel.append(make("span","rally-source-label","По касаниям"));
      const duration=make("div","rally-card-duration");
      duration.title="Длительность: "+length(item);
      duration.append(make("span","","Длительность"),make("strong","",length(item)));
      top.append(idLabel);
      const when=make("div","rally-card-time-range");
      when.append(make("strong","rally-card-time-value",
        second(item.detected_start_frame+(Number(s.analysisOffset)||0)*fps())+"–"+second(item.detected_end_frame+(Number(s.analysisOffset)||0)*fps())+" с"));
      when.title="На видео: "+second(item.detected_start_frame+(Number(s.analysisOffset)||0)*fps())+"–"+
        second(item.detected_end_frame+(Number(s.analysisOffset)||0)*fps())+" с";
      top.append(when);
      const stats=make("div","card-stats");
      const touches=make("div","rally-card-stat");
      const count=item.motion_event_count;
      touches.append(make("span","","Касаний"),
        make("strong","",Number.isSafeInteger(count)&&count>=0?count:"—"));
      const players=make("div","rally-card-stat");
      const countPlayers=rallyPlayerCount(item);
      players.title="Устойчивое число одновременно наблюдаемых Track ID в границах розыгрыша";
      players.append(make("span","","Игроков"),
        make("strong","",countPlayers===null?"—":countPlayers));
      stats.append(duration,touches,players);
      const bottom=make("div","card-bottom");
      const favorite=make("button","gold-toggle",gold?"★":"☆");
      favorite.type="button";
      favorite.title=gold?"Убрать из избранного":"Добавить в избранное";
      favorite.setAttribute("aria-pressed",String(gold));
      favorite.setAttribute("aria-label",favorite.title);
      favorite.disabled=!s.writable||s.busy;
      favorite.addEventListener("click",e=>{e.stopPropagation();toggleGold(id);});
      const predictions=(Array.isArray(item.motion_events)?item.motion_events:[])
        .map(event=>event.rally_context ? event.rally_context.estimated_kind : event.action_prediction?.kind).filter(kind=>typeof kind==="string");
      if(predictions.length)card.title="Автоклассификация (предварительная): "+
        [...new Set(predictions)].map(kind=>kind.replaceAll("_"," ")).join(", ");
      const foot=make("div","rally-card-foot");
      bottom.append(favorite);
      foot.append(stats,bottom);
      const types=make("div","rally-card-techniques",estimatedTypes(item));
      types.title="Автоматические оценки по всем кандидатам касаний в розыгрыше; не подтверждены человеком";
      const {start,end}=rallyClipFrames(item);
      const progress=make("div","rally-card-progress");
      progress.tabIndex=s.current===id?0:-1;
      progress.setAttribute("role","slider");
      progress.setAttribute("aria-orientation","horizontal");
      progress.setAttribute("aria-valuemin","0");
      progress.setAttribute("aria-valuemax",((end-start+1)/fps()).toFixed(1));
      progress.setAttribute("aria-valuenow","0");
      progress.setAttribute("aria-label","Переход к моменту розыгрыша");
      progress.title="Нажмите, чтобы воспроизвести розыгрыш с выбранного момента";
      const playFromFraction=fraction=>{
        const bounded=Math.max(0,Math.min(1,fraction));
        view(item,(Number(s.analysisOffset)||0)+(start+bounded*(end-start+1))/fps());
        // Clicking the timeline inside an already selected card retains focus.
      };
      progress.addEventListener("click",event=>{
        event.stopPropagation();
        const rect=progress.getBoundingClientRect();
        if(!rect.width)return;
        playFromFraction((event.clientX-rect.left)/rect.width);
      });
      progress.addEventListener("keydown",event=>{
        const key=event.key;
        if(!["ArrowLeft","ArrowRight","Home","End","PageUp","PageDown","Enter"," "].includes(key))return;
        event.preventDefault();event.stopPropagation();
        if(key==="Enter"||key===" "){
          view(item,Number.isFinite(ui.video.currentTime)?ui.video.currentTime:(Number(s.analysisOffset)||0)+start/fps());
          return;
        }
        const current=Number.isFinite(ui.video.currentTime)?
          ui.video.currentTime:(Number(s.analysisOffset)||0)+start/fps();
        const duration=(end-start+1)/fps();
        const fraction=key==="Home"?0:key==="End"?1:
          (current-(Number(s.analysisOffset)||0)-start/fps()+(
            key==="ArrowRight"||key==="PageUp"?1:-1
          )*(key==="PageUp"||key==="PageDown"||event.shiftKey?5:1))/duration;
        playFromFraction(fraction);
      });
      const filled=make("span","rally-card-progress-filled");
      const markers=make("span","rally-card-progress-markers");
      const head=make("span","rally-card-progress-head");
      for(const part of [filled,markers,head])part.setAttribute("aria-hidden","true");
      progress.append(filled,markers,head);
      rallyCardProgresses.set(id,{progress,markers,start,end});
      const result=make("div","rally-card-teams");
      const metadata=item.metadata;
      const winner=metadata?.winning_team;
      for(const [key,label,side] of [["team_1","Команда 1","near"],["team_2","Команда 2","far"]]){
        const team=make("div","rally-card-team rally-team-"+side);
        const ids=Array.isArray(metadata?.[key])?metadata[key]:[];
        const names=ids.filter(id=>Number.isSafeInteger(id)&&id>0).map(id=>
          (typeof overlayLabels!=="undefined"&&overlayLabels.loaded&&overlayLabels.key.startsWith([s.project,s.run,s.video].join(":")+":")?overlayLabels.names.get(String(id)):null)||"ID "+id);
        team.append(make("strong","rally-team-label",label),
          make("span","rally-team-members",names.length?names.join(", "):"Не определена"));
        if(winner===key)team.append(make("strong","rally-team-point","+1"));
        result.append(team);
      }
      const resultLabel=winner==="team_1"?"Очко: команда 1":winner==="team_2"?"Очко: команда 2":"Очко: не определено";
      result.append(make("div","rally-card-result",winner==="team_1"||winner==="team_2"?resultLabel+" (предп.)":resultLabel));
      card.append(top,foot,types,result,progress);ui.list.append(card);
    }
    ui.list.scrollTop=previousScroll;
    refreshRallyCardTouchTicks();updateRallyCardProgress();
  };
  // Fetch next rally pages when the right-hand list is scrolled to its end.
  const loadMore=async()=>{
    if(reference.active||s.loadingMore||!s.canLoadMore||!s.project||!s.run||
       (training.active&&!timelineRallyLayers.touches.checked))return;
    s.loadingMore=true;
    const token=s.generation,project=s.project,run=s.run;
    const offset=s.items.length;
    try{
      const raw=await previewCache.load(url("/api/cpu/native/rallies/preview",{
        project_id:project,rally_run_id:run,offset,limit:s.limit,
        ...rallyListQuery()
      }),900000,true);
      if(token!==s.generation||project!==s.project||run!==s.run)return;
      if(raw.schema!=="volley-native-rally-preview/1"||
        !validSurfaceContacts(raw)||raw.authoritative_game_events!==false||
        raw.automatic_montage_eligible!==false||
        !Array.isArray(raw.items))
        throw Error("Неподдерживаемая версия предварительного CPU-анализа");
      s.items.push(...raw.items);
      s.canLoadMore=raw.items.length>0&&s.items.length<s.filtered;
      cards();
      drawTimeline();
    }catch(error){
      if(token===s.generation)
        message("Не удалось догрузить розыгрыши: "+String(error?.message||error),"error");
    }finally{s.loadingMore=false;}
  };
  const uploadAudio=async()=>{
    const file=ui.music.files&&ui.music.files[0];
    if(!file)return null;
    if(file.size>90*1024*1024)throw Error("Файл музыки превышает 90 МиБ.");
    const form=new FormData();form.append("file",file,file.name);
    const data=await request("/api/montage/native/audio",{method:"POST",body:form});
    const name=data.audio_file||data.file||data.filename;
    if(typeof name!=="string"||!name)throw Error("Сервер не вернул имя аудиотрека.");
    if(document.getElementById("studio-montage-audio-fit")?.checked&&
      Number.isFinite(Number(data.duration_sec))&&Number(data.duration_sec)>0){
      const limit=document.getElementById("studio-montage-duration");
      if(limit)limit.value=String(Math.min(600,Math.max(8,Math.round(Number(data.duration_sec)))));
    }
    return name;
  };
  const updateJob=async()=>{
    if(!s.jobId)return;
    try{
      const result=await request("/api/montage/native/status");
      if(result.id&&result.id!==s.jobId){
        stopPoll();ui.job.textContent="Запущен другой монтаж.";return;
      }
      const percentage=Number(result.overall_percent);
      ui.job.textContent=(result.stage||result.state||"Монтаж")+
        " · "+(Number.isFinite(percentage)?percentage.toFixed(0)+"%":"—");
      if(["complete","completed","done"].includes(result.state)){
        stopPoll();ui.cancel.classList.add("hidden");ui.render.disabled=false;
        window.dispatchEvent(new Event("volley:montage-complete"));
        if(typeof result.media_url==="string"&&
          result.media_url.startsWith("/api/montage/native/media?")){
          ui.download.href=result.media_url;ui.download.classList.remove("hidden");
        }
        const archive=document.getElementById("studio-download-archive");
        if(archive&&typeof result.archive_url==="string"&&
          result.archive_url.startsWith("/api/montage/native/media?")&&
          result.archive_url.endsWith("&archive=1")){
          archive.href=result.archive_url;archive.classList.remove("hidden");
        }
        message("Монтаж завершён. Видео доступно для скачивания.","ok");
      }else if(["cancelled","failed","error"].includes(result.state)){
        stopPoll();ui.cancel.classList.add("hidden");ui.render.disabled=false;
        message(result.error||"Ошибка монтажа.","error");
      }
    }catch(e){failed(e);}
  };
  const startMontage=async()=>{
    const ids=window.VolleyballMontageDraft?.ids(s.project,s.run)||[];
    if(!wasmReady||s.busy||!s.run||!s.marksLoaded||!s.gold.size||
      s.gold.size>80||s.selected.size!==s.gold.size||!ids.length||
      ids.some(id=>!s.selected.has(id)))return;
    const project=s.project,run=s.run;
    s.busy=true;ui.render.disabled=true;ui.download.classList.add("hidden");
    document.getElementById("studio-download-archive")?.classList.add("hidden");
    try{
      ui.job.textContent="Загрузка музыки и проверка исходников…";
      const layout=ui.layout.value==="vertical"&&
        document.getElementById("studio-montage-camera")?.value==="on"?
        "vertical_follow":ui.layout.value;
      const camera_zoom=Number(document.getElementById("studio-montage-zoom")?.value)||1.25;
      const volume=Number(ui.volume.value);
      const encoder=ui.encoder.value;
      const music=await uploadAudio();
      const result=await post("/api/montage/native/start-from-rally-candidates",{
        project_id:project,rally_run_id:run,candidate_ids:ids,
        audio_file:music,audio_volume:volume,
        target_duration_sec:Number(document.getElementById("studio-montage-duration")?.value)||60,
        layout,camera_zoom,encoder,
        include_recognition_data:document.getElementById("studio-montage-recognition")?.checked===true
      });
      const id=result.id||result.job_id;
      if(typeof id!=="string"||!id)throw Error("Сервер не вернул ID монтажа.");
      s.jobId=id;ui.cancel.classList.remove("hidden");
      window.VolleyballTasks?.set("montage",{label:"Монтаж",stage:"В очереди",progress:0});
      window.VolleyballTasks?.refresh();
      ui.job.textContent="Монтаж запущен.";
      message("Сборка избранных розыгрышей из оригинального видео.","ok");
      stopPoll();await updateJob();
      if(s.jobId&&!ui.cancel.classList.contains("hidden")){
        s.timer=setInterval(updateJob,1500);
      }
    }catch(e){failed(e);ui.job.textContent="Монтаж не запущен.";ui.render.disabled=false;}
    finally{s.busy=false;}
  };
  // The project dialog accepts browser-owned drop/chooser uploads and athlete links.
  // Both use the existing Rust API and isolated storage; never infer local file paths.
  if(ui.projectNew){
    let creating=false,uploading=false,addingPlayer=false,activeUpload=null;
    let playerRevision="absent",dialogEpoch=0; const linkedFiles=new Map();
    const maxVideoBytes=window.VolleyballUpload.maxVideoBytes;
    const selectedVideos=()=>[...linkedFiles.keys()];
    const selectedPlayers=()=>[...ui.projectPlayers.querySelectorAll("input[type=checkbox]:checked")]
      .map(input=>input.value);
    const updateSubmit=()=>{
      const count=selectedVideos().length;
      ui.projectSubmit.disabled=creating||uploading||!ui.projectName.value.trim()||
        count<1||count>30;
      ui.projectSubmit.textContent=creating?"Создаём…":"Создать проект";
      ui.projectDrop.setAttribute("aria-busy",String(uploading));
      ui.projectDrop.classList.toggle("uploading",uploading);
    };
    const alertCreate=error=>{
      ui.projectCreateError.textContent=String(error?.message||error);
    };
    const formatVideoDuration=seconds=>{
      if(!Number.isFinite(seconds)||seconds<0)return "--:--";
      const total=Math.round(seconds);
      return String(Math.floor(total/60)).padStart(2,"0")+":"+
        String(total%60).padStart(2,"0");
    };
    const renderLinkedFiles=()=>{
      const scrollTop=ui.projectVideos.scrollTop;
      ui.projectVideos.replaceChildren();
      for(const [key,file] of linkedFiles){
        const row=make("div","studio-project-video-option studio-project-linked-video");
        row.title=String(file.display_name||file.name);
        const name=make("span","studio-project-video-name",file.display_name||file.name);
        const duration=make("time","studio-project-video-duration",
          formatVideoDuration(file.duration_sec));
        duration.title="Продолжительность видео";
        const remove=make("button","secondary studio-project-video-remove","×");
        remove.type="button";remove.title="Отвязать видео от нового проекта";
        remove.setAttribute("aria-label","Отвязать "+(file.display_name||file.name));
        remove.addEventListener("click",()=>{
          linkedFiles.delete(key);renderLinkedFiles();
        });
        row.append(name,duration,remove);
        ui.projectVideos.append(row);
      }
      if(!linkedFiles.size)ui.projectVideos.append(make("p","studio-project-video-empty",
        "Видео пока не выбраны. Добавьте исходные файлы выше."));
      ui.projectVideos.scrollTop=scrollTop;
      updateSubmit();
    };
    const renderPlayers=(players,selected)=>{
      ui.projectPlayers.replaceChildren();
      if(!players.length){
        ui.projectPlayers.append(make("p","panel-note","Добавьте первого игрока ниже."));
      }
      for(const player of players){
        if(typeof player.id!=="string"||typeof player.name!=="string")continue;
        const label=make("label","studio-project-video-option");
        label.title=player.name;
        const checkbox=document.createElement("input");
        checkbox.type="checkbox";checkbox.value=player.id;
        checkbox.checked=selected.has(player.id);
        label.append(checkbox,make("span","",player.name));
        ui.projectPlayers.append(label);
      }
    };
    const refreshPlayers=async(extraChecked=new Set(),token=dialogEpoch)=>{
      const selected=new Set([...selectedPlayers(),...extraChecked]);
      const raw=await post("/api/players/library/native",{action:"read"});
      if(token!==dialogEpoch||!ui.projectDialog.open)return;
      playerRevision=String(raw.revision||"absent");
      renderPlayers(Array.isArray(raw.players)?raw.players:[],selected);
    };
    const loadDialog=async()=>{
      if(creating||uploading)return;
      const token=++dialogEpoch;
      ui.createForm.reset();
      playerRevision="absent"; linkedFiles.clear();
      ui.projectCreateError.textContent="";
      ui.projectCreateStatus.textContent="";
      ui.projectUploadStatus.textContent="";
      renderLinkedFiles();
      ui.projectPlayers.replaceChildren(make("p","panel-note","Загрузка игроков…"));
      ui.projectSubmit.disabled=true;
      ui.projectDialog.showModal();
      ui.projectName.focus();
      const results=await Promise.allSettled([refreshPlayers(new Set(),token)]);
      if(token!==dialogEpoch)return;
      for(const result of results)if(result.status==="rejected")alertCreate(result.reason);
    };
    const closeDialog=()=>{
      if(creating)return;
      dialogEpoch++;
      // Closing an unrelated dialog must never cancel a project's background transfer.
      ui.projectDialog.close();
    };
    const nativePick=async()=>{
      if(creating||uploading||!ui.projectDialog.open)return;
      uploading=true;updateSubmit();
      const epoch=dialogEpoch;
      try{
        const result=await post("/api/project-source/native-pick",{});
        if(epoch!==dialogEpoch||!ui.projectDialog.open)return;
        const items=Array.isArray(result.videos)?result.videos:[];
        if(selectedVideos().length+items.length>30)throw Error("Максимум 30 видео в одном проекте");
        for(const item of items){
          if(typeof item.name!=="string"||typeof item.display_name!=="string")
            throw Error("Некорректный ответ выбора видео");
          linkedFiles.set(item.name,item);
        }
        ui.projectCreateError.textContent="";
        renderLinkedFiles();
      }catch(error){
        if(epoch===dialogEpoch)alertCreate(error);
      }finally{uploading=false;updateSubmit();}
    };
    ui.projectNew.addEventListener("click",()=>loadDialog().catch(failed));
    ui.projectClose.addEventListener("click",closeDialog);
    ui.projectCancel.addEventListener("click",closeDialog);
    ui.projectDialog.addEventListener("cancel",event=>{event.preventDefault();closeDialog();});
    ui.projectName.addEventListener("input",updateSubmit);
    ui.projectDrop.addEventListener("click",()=>{void nativePick();});
    ui.projectDrop.addEventListener("keydown",event=>{
      if(event.key==="Enter"||event.key===" "){
        event.preventDefault();void nativePick();
      }
    });
    ui.projectFiles.addEventListener("change",()=>{
      ui.projectFiles.value="";
      void nativePick();
    });
    ui.projectDrop.addEventListener("dragenter",event=>{
      event.preventDefault();ui.projectDrop.classList.add("drag-over");
    });
    ui.projectDrop.addEventListener("dragover",event=>{
      event.preventDefault();if(event.dataTransfer)event.dataTransfer.dropEffect="copy";
      ui.projectDrop.classList.add("drag-over");
    });
    ui.projectDrop.addEventListener("dragleave",()=>ui.projectDrop.classList.remove("drag-over"));
    ui.projectDrop.addEventListener("drop",event=>{
      event.preventDefault();ui.projectDrop.classList.remove("drag-over");
      // Browsers do not expose the real pathname of a dropped file.
      // The trusted native picker grants access to the original without transferring bytes.
      void nativePick();
    });
    ui.projectAddPlayer.addEventListener("click",async()=>{
      const name=ui.projectPlayerName.value.trim();
      if(!name||addingPlayer||creating)return;
      addingPlayer=true;ui.projectAddPlayer.disabled=true;
      ui.projectCreateError.textContent="";
      const epoch=dialogEpoch;
      try{
        const created=await post("/api/players/library/native",{
          action:"create",name,expected_revision:playerRevision
        });
        if(epoch!==dialogEpoch)return;
        playerRevision=String(created.revision);
        renderPlayers(Array.isArray(created.players)?created.players:[],
          new Set([...selectedPlayers(),created.player_id]));
        ui.projectPlayerName.value="";
        window.dispatchEvent(new Event("volley:player-library-updated"));
      }catch(error){
        if(epoch===dialogEpoch){
          alertCreate(error);
          await refreshPlayers(new Set(),epoch).catch(()=>{});
        }
      }finally{
        addingPlayer=false;ui.projectAddPlayer.disabled=false;
      }
    });
    ui.projectPlayerName.addEventListener("keydown",event=>{
      if(event.key==="Enter"){event.preventDefault();ui.projectAddPlayer.click();}
    });
    ui.createForm.addEventListener("submit",async event=>{
      event.preventDefault();
      if(creating||uploading)return;
      const name=ui.projectName.value.trim();
      const player_ids=selectedPlayers();
      if(!name||!selectedVideos().length||selectedVideos().length>30){updateSubmit();return;}
      creating=true;updateSubmit();
      ui.projectCancel.disabled=true;ui.projectClose.disabled=true;
      ui.projectCreateError.textContent="";
      ui.projectCreateStatus.textContent="Создаём проект…";
      try{
        const videos=selectedVideos();
        const created=await post("/api/project-create",{name,videos,player_ids});
        const id=created.project?.id;
        if(typeof id!=="string"||!id)
          throw Error("Проект создан, но его ID не получен. Обновите список.");
        ui.projectCreateStatus.textContent="Проект создан. Открываем…";
        ui.projectDialog.close();
        dialogEpoch++;
        invalidateCatalog();
        await projects(id);
        document.getElementById("studio-nav-analysis").click();
        message("Проект «"+name+"» создан и выбран.","ok");
      }catch(error){
        if(!ui.projectDialog.open){
          message("Проект создан, но не удалось выбрать его автоматически: "+
            String(error?.message||error),"error");
          return;
        }
        if(error.message==="Проект с таким названием уже существует"){
          try{
            invalidateCatalog();
            const catalog=await window.VolleyballStudioProjectCatalog();
            const existing=(Array.isArray(catalog.projects)?catalog.projects:[])
              .find(row=>typeof row.name==="string"&&
                row.name.toLocaleLowerCase()===name.toLocaleLowerCase());
            if(existing?.id){
              ui.projectDialog.close();
              dialogEpoch++;
              await projects(existing.id);
              document.getElementById("studio-nav-analysis").click();
              message("Проект «"+existing.name+"» уже существует — открыт существующий.","ok");
              return;
            }
          }catch(refreshError){
            console.warn("Не удалось открыть существующий проект:",refreshError);
          }
        }
        alertCreate(error);
      }finally{
        creating=false;uploading=false;
        ui.projectCancel.disabled=false;ui.projectClose.disabled=false;
        ui.projectCreateStatus.textContent="";
        updateSubmit();
      }
    });
  }
  ui.project.addEventListener("change",async()=>{
    s.catalogEpoch++;
    s.project=ui.project.value;s.analysis="";
    sourceSelection.save(s.project);
    const entry=s.projectCatalog.find(row=>row.id===s.project);
    s.projectVideos=Array.isArray(entry?.videos)?
      entry.videos.filter(video=>typeof video==="string"):[];
    const project=s.project;
    const loading=loadChain("","",false,true);
    window.dispatchEvent(new Event("volley:project-ready"));
    try{
      await loading;
      if(project===s.project&&reference.active)void refreshReferenceCatalog();
    }catch(e){if(project===s.project)failed(e);}
  });
  ui.analysis.addEventListener("change",()=>setAnalysis().catch(failed));
  ui.run.addEventListener("change",async()=>{
    const id=ui.run.value;reset();
    if(!(s.chain?.byAnalysis.get(s.analysis)||[]).some(row=>row.rally_run_id===id))return;
    s.run=id;
    sourceSelection.save(s.project,s.analysis,s.run);
    window.dispatchEvent(new CustomEvent("volley:postprocessing-change",{
      detail:{project_id:s.project,rally_run_id:s.run}
    }));
    void loadTimeline().catch(failed);
    try{await loadPage();}catch(e){failed(e);}
  });
  window.addEventListener("volley:refresh-projects",()=>{invalidateCatalog();projects().catch(failed);});
  window.addEventListener("volley:refresh-chain",()=>{invalidateCatalog();projects().catch(failed);});
  // One authoritative place to select the new GPU -> CPU lineage: header selects.
  window.addEventListener("volley:pipeline-complete",async event=>{
    const {project_id,video,analysis_id,postprocessing_id}=event.detail||{};
    if(project_id!==s.project||typeof analysis_id!=="string")return;
    try{
      invalidateCatalog();
      await loadChain();
      if(s.project!==project_id)return;
      const choice=s.chain?.choices.find(item=>item.kind==="fusion"&&
        item.id===analysis_id&&item.video===video);
      if(!choice)throw Error("Новая GPU-версия не найдена в индексе стадий");
      ui.analysis.value=choice.value;
      await setAnalysis(typeof postprocessing_id==="string"?postprocessing_id:"");
      if(typeof postprocessing_id==="string"&&s.run!==postprocessing_id)
        throw Error("Новая CPU-постобработка не привязана к GPU-версии");
    }catch(e){failed(e);}
  });
  window.VolleyballStudioApplyMontageAnalysis=async ({project_id,source_run_ids})=>{
    if(!/^[a-f0-9]{12}$/.test(project_id||"")||
        !Array.isArray(source_run_ids)||!source_run_ids.length)
      throw Error("У ролика не сохранена исходная версия анализа");
    await projects(project_id);
    if(s.project!==project_id)throw Error("Не удалось выбрать исходный проект");
    const source=new Set(source_run_ids.filter(id=>/^[a-f0-9]{32}$/.test(id)));
    const matched=[...(s.chain?.byAnalysis||new Map())].flatMap(([analysis,runs])=>
      runs.filter(row=>source.has(row.rally_run_id)).map(row=>({
        analysis,run:row.rally_run_id
      })));
    if(!matched.length)throw Error("Исходная CPU-версия недоступна в проекте");
    const selected=matched.find(row=>row.run===source_run_ids[0])||matched[0];
    ui.analysis.value=selected.analysis;
    await setAnalysis(selected.run);
    if(s.run!==selected.run)throw Error("Выбранная версия анализа не восстановлена");
    $("studio-nav-preview")?.click();
  };
  window.addEventListener("volley:open-legacy-project",async event=>{
    const id=event.detail?.project_id;
    if(typeof id!=="string"||!/^[a-f0-9]{12}$/.test(id))return;
    try{
      await projects();
      const has=[...ui.project.options].some(row=>row.value===id);
      if(!has)throw Error("Проект не найден в текущем реестре");
      ui.project.value=id;ui.project.dispatchEvent(new Event("change"));
      document.getElementById("studio-nav-preview").click();
    }catch(e){failed(e);}
  });
  ui.filter.addEventListener("change",()=>{
    ui.list.scrollTop=0;
    void loadPage().catch(failed);
  });
  ui.list.addEventListener("scroll",()=>{
    if(ui.list.scrollTop+ui.list.clientHeight>=ui.list.scrollHeight-240)
      void loadMore();
  },{passive:true});
  ui.render.addEventListener("click",startMontage);
  ui.cancel.addEventListener("click",async()=>{
    if(!s.jobId)return;
    try{
      await post("/api/montage/native/cancel",{job_id:s.jobId});
      message("Отмена отправлена.","ok");await updateJob();
    }catch(e){failed(e);}
  });
  ui.video.addEventListener("loadedmetadata",()=>{
    if(s.videoKey)try{
      const duration=ui.video.duration;
      ui.video.currentTime=Number.isFinite(duration)&&duration>0?
        Math.min(s.seekAt,Math.max(0,duration-0.05)):s.seekAt;
    }catch{}
    const duration=timelineDuration();
    if(duration){
      timeline.middle.textContent="00:00.0";
      timeline.end.textContent=timeLabel(duration);
    }
    drawTimeline();updateRallyCardProgress();
  });
  const nextRally=()=>{
    // The full CPU timeline is authoritative; filtered/paginated cards are only a fallback.
    const rows=timeline.data?.project_id===s.project&&
      timeline.data?.rally_run_id===s.run?timeline.data.rallies:s.items;
    const start=row=>Number(row.detected_start_frame??row.start_frame??
      row.manual_clip?.start_frame??Infinity);
    const ordered=[...rows].filter(row=>Number.isFinite(start(row)))
      .sort((a,b)=>start(a)-start(b));
    const index=ordered.findIndex(row=>row.candidate_id===s.current);
    const row=index>=0?ordered[index+1]:
      ordered.find(entry=>(Number(s.analysisOffset)||0)+start(entry)/fps()>=(s.clipEnd??ui.video.currentTime)-0.02);
    return row?(s.items.find(item=>item.candidate_id===row.candidate_id)||row):null;
  };
  const onRallyBoundary=()=>{
    if(s.playbackMode==="free"){s.stopAt=null;return;}
    if(s.playbackMode==="loop"&&s.clipStart!==null){
      s.seekAt=s.clipStart;
      s.stopAt=s.clipEnd;
      ui.video.currentTime=s.clipStart;
      if(ui.video.paused)ui.video.play().catch(error=>
        message("Воспроизведение недоступно: "+error.message,"error"));
      return;
    }
    if(s.playbackMode==="sequence"){
      const next=nextRally();
      if(next){view(next);return;}
    }
    s.stopAt=null;
    ui.video.pause();
  };
  ui.video.addEventListener("timeupdate",()=>{
    updateTimelineProgress();updateRallyCardProgress();
    if(s.stopAt!==null&&ui.video.currentTime>=s.stopAt)
      onRallyBoundary();
  });
  // Media can end before the rounded final-frame boundary; loop must still repeat.
  ui.video.addEventListener("ended",()=>{
    if(s.stopAt!==null&&s.playbackMode!=="free")onRallyBoundary();
  });
  let videoDecodeErrorShown=false;
  ui.video.addEventListener("loadeddata",()=>{
    if(videoDecodeErrorShown){videoDecodeErrorShown=false;overlayMessage("");}
  });
  ui.video.addEventListener("error",()=>{
    videoDecodeErrorShown=true;
    overlay.presentedTime=null;overlayContext();
    if(s.videoKey)overlayMessage("Видео не декодируется (код "+
      (ui.video.error?.code||"?")+"). Нажмите «Воспроизвести» для повторной загрузки.");
    if(s.videoKey)ui.viewDescription.textContent=
      "Видео недоступно для воспроизведения. Проверьте подготовку видео.";
  });
  // Native controls are intentionally disabled: the source-wide annotated
  // timeline is the ONE seek slider. Keep transport controls accessible.
  const playback={
    play:$("studio-video-play"),mute:$("studio-video-mute"),
    fullscreen:$("studio-video-fullscreen"),mode:$("studio-video-mode"),
    wrap:ui.video.closest(".video-wrap")
  };
  try{
    const mode=localStorage.getItem("volley:studio-playback-mode:v1");
    if(["sequence","loop","free","once"].includes(mode))playback.mode.value=mode;
  }catch{}
  s.playbackMode=playback.mode.value;
  const updatePlaybackControls=()=>{
    const playing=!ui.video.paused&&!ui.video.ended;
    playback.play.disabled=!s.videoKey;
    playback.play.textContent=playing?"❚❚ Пауза":"▶ Воспроизвести";
    playback.play.setAttribute("aria-label",playing?"Пауза":"Воспроизвести");
    playback.mute.disabled=!s.videoKey;
    playback.mute.textContent=ui.video.muted?"🔇 Без звука":"♫ Звук";
    playback.mute.setAttribute("aria-pressed",String(ui.video.muted));
    playback.fullscreen.disabled=!s.videoKey;
    playback.fullscreen.textContent=document.fullscreenElement?"⛶ Свернуть":"⛶ На весь экран";
  };
  const toggleVideoPlayback=()=>{
    if(!s.videoKey||!previewIsVisible())return;
    if(ui.video.paused){
      if(ui.video.error){
        // A failed byte-range request must not permanently poison this player.
        overlay.presentedTime=null;
        loadPreviewVideo(true);
        return;
      }
      // Replay a completed selection from its beginning, not a fraction past its end.
      if(s.playbackMode!=="free"&&s.clipStart!==null&&s.clipEnd!==null&&
         (ui.video.ended||ui.video.currentTime>=s.clipEnd-0.04)){
        s.seekAt=s.clipStart;s.stopAt=s.clipEnd;
        try{ui.video.currentTime=s.clipStart;}catch{}
      }
      ui.video.play().catch(error=>message("Воспроизведение недоступно: "+error.message,"error"));
    }else ui.video.pause();
  };
  playback.mode.addEventListener("change",()=>{
    s.playbackMode=playback.mode.value;
    s.stopAt=s.playbackMode==="free"?null:s.clipEnd;
    try{localStorage.setItem("volley:studio-playback-mode:v1",s.playbackMode);}catch{}
    if(s.stopAt!==null&&ui.video.currentTime>=s.stopAt&&s.clipStart!==null){
      s.seekAt=s.clipStart;
      try{ui.video.currentTime=s.seekAt;}catch{}
    }
  });
  playback.play.addEventListener("click",toggleVideoPlayback);
  playback.mute.addEventListener("click",()=>{
    ui.video.muted=!ui.video.muted;
    updatePlaybackControls();
  });
  playback.fullscreen.addEventListener("click",async()=>{
    try{
      if(document.fullscreenElement)await document.exitFullscreen();
      else await playback.wrap.requestFullscreen();
    }catch(error){message("Полноэкранный режим недоступен: "+error.message,"error");}
    updatePlaybackControls();
  });
  ui.video.addEventListener("click",toggleVideoPlayback);
  // Left/right arrows scrub the one source-wide timeline, including raw GPU.
  // Rust/WASM owns the precise 0.5 s tap and exponential hold acceleration.
  // JS only adapts keyboard events, requestAnimationFrame and media seeking.
  const arrowScrub={direction:0,startedAt:0,lastSeekAt:0,frame:0};
  const stopArrowScrub=()=>{
    arrowScrub.direction=0;
    if(arrowScrub.frame)cancelAnimationFrame(arrowScrub.frame);
    arrowScrub.frame=0;
  };
  const tickArrowScrub=timestamp=>{
    arrowScrub.frame=0;
    if(!arrowScrub.direction)return;
    if(!previewIsVisible()||s.videoKey!==s.project+":"+s.video||
       !timelineDuration()){stopArrowScrub();return;}
    if(timestamp-arrowScrub.lastSeekAt>=85){
      const delta=window.VolleyballUi.engine.volley_preview_seek_hold_delta(
        arrowScrub.direction,timestamp-arrowScrub.startedAt,
        timestamp-arrowScrub.lastSeekAt);
      if(delta)seekTimeline(s.seekAt+delta);
      arrowScrub.lastSeekAt=timestamp;
    }
    arrowScrub.frame=requestAnimationFrame(tickArrowScrub);
  };
  document.addEventListener("keydown",event=>{
    if(event.key!=="ArrowLeft"&&event.key!=="ArrowRight")return;
    if(event.altKey||event.ctrlKey||event.metaKey||event.isComposing||
       event.target?.closest?.(".rally-card-progress")||
       ["INPUT","TEXTAREA","SELECT"].includes(event.target?.tagName)||
       event.target?.isContentEditable||!s.videoKey||!previewIsVisible()||
       s.videoKey!==s.project+":"+s.video||!timelineDuration())return;
    event.preventDefault();
    event.stopPropagation(); // No duplicate seek from the focused timeline slider.
    const direction=event.key==="ArrowRight"?1:-1;
    if(arrowScrub.direction===direction)return;
    stopArrowScrub();
    arrowScrub.direction=direction;
    arrowScrub.startedAt=performance.now();
    arrowScrub.lastSeekAt=arrowScrub.startedAt;
    const current=Number.isFinite(ui.video.currentTime)?
      ui.video.currentTime:s.seekAt;
    seekTimeline(current+window.VolleyballUi.engine.volley_preview_seek_tap(direction));
    arrowScrub.frame=requestAnimationFrame(tickArrowScrub);
  },true);
  document.addEventListener("keyup",event=>{
    if((event.key==="ArrowLeft"&&arrowScrub.direction===-1)||
       (event.key==="ArrowRight"&&arrowScrub.direction===1))
      stopArrowScrub();
  },true);
  window.addEventListener("blur",stopArrowScrub);
  document.addEventListener("visibilitychange",()=>{
    if(document.hidden)stopArrowScrub();
  });
  // Only text editing owns Space. Range sliders, buttons and timeline cards
  // must not steal the preview's play/pause shortcut after they receive focus.
  const isTypingSpaceTarget=target=>{
    const tag=target?.tagName?.toUpperCase();
    return target?.isContentEditable||
      !!target?.closest?.('[contenteditable], [role="textbox"]')||
      tag==="TEXTAREA"||tag==="SELECT"||
      (tag==="INPUT"&&!["button","submit","reset","checkbox","radio",
        "range","color","file"].includes(target.type?.toLowerCase()||"text"));
  };
  document.addEventListener("keydown",event=>{
    if(event.code!=="Space"&&event.key!==" ")return;
    if(event.altKey||event.ctrlKey||event.metaKey||event.isComposing||
       !s.videoKey||!previewIsVisible()||
       isTypingSpaceTarget(event.target)||event.target?.closest?.("dialog[open]"))return;
    // Consume repeats as well: otherwise a held key scrolls the page or
    // activates a focused button after the first playback toggle.
    event.preventDefault();
    event.stopPropagation();
    if(!event.repeat)toggleVideoPlayback();
  },true);
  ui.video.addEventListener("keydown",event=>{
    if(event.key==="k"||event.key==="K"){
      event.preventDefault();toggleVideoPlayback();
    }
  });
  for(const type of ["play","pause","ended","loadedmetadata","emptied","volumechange"]){
    ui.video.addEventListener(type,updatePlaybackControls);
  }
  document.addEventListener("fullscreenchange",updatePlaybackControls);
  updatePlaybackControls();
  window.addEventListener("volley:open-native-rally",async event=>{
    const target=event.detail||{};
    if(typeof target.project!=="string"||typeof target.run!=="string"
      ||target.project!==s.project)return;
    try{
      if(!s.chain?.runToAnalysis.has(target.run))await projects();
      const analysis=s.chain?.runToAnalysis.get(target.run);
      if(analysis){
        ui.analysis.value=analysis;
        await setAnalysis(target.run);
      }
      if(s.run===target.run){
        document.getElementById("studio-nav-preview").click();
        message("Открыта выбранная CPU-версия розыгрышей.","ok");
      }else {
        message("Выбранная версия не найдена в списке. Обновите данные.","error");
      }
    }catch(e){failed(e);}
  });
  request("/api/studio/analysis-capabilities").then(value=>{
    if(value.analysis_edit){makeAnalysisPicker(ui.analysis,true);makeAnalysisPicker(ui.run,false);}
  }).catch(()=>{});
  makeAnalysisPicker(ui.project,"project");
  loadRustSelection().catch(e=>{
    wasmReady=false;
    failed(new Error("Модуль выбора фрагментов недоступен: "+String(e.message||e)));
    renderSelection();cards();
  });
  projects().catch(failed);
})();
