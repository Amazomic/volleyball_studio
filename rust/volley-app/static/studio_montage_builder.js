"use strict";
// Unified Rust-backed montage composer; never trust browser-supplied clip bounds.
(() => {
  const $=id=>document.getElementById(id);
  const ui={
    project:$("studio-project"),profile:$("studio-montage-profile"),
    settingsLabel:$("studio-montage-settings-label"),settingsHint:$("studio-montage-settings-hint"),
    player:$("studio-montage-player"),action:$("studio-montage-action"),
    run:$("studio-run"),statistics:$("studio-montage-statistics"),
    statisticsField:$("studio-montage-statistics-field"),
    context:$("studio-montage-context"),duration:$("studio-montage-duration"),
    playerField:$("studio-montage-player-field"),actionField:$("studio-montage-action-field"),
    contextField:$("studio-montage-context-field"),sources:$("studio-montage-sources"),
    sourceList:$("studio-montage-source-list"),favorites:$("studio-montage-favorites"),
    layout:$("studio-layout"),encoder:$("studio-encoder"),music:$("studio-music"),
    volume:$("studio-volume"),audioFit:$("studio-montage-audio-fit"),
    recognition:$("studio-montage-recognition"),archiveDownload:$("studio-download-archive"),
    audioLength:$("studio-montage-audio-length"),camera:$("studio-montage-camera"),
    cameraField:$("studio-montage-camera-field"),zoom:$("studio-montage-zoom"),
    zoomField:$("studio-montage-zoom-field"),previewList:$("studio-montage-preview-list"),
    refresh:$("studio-montage-refresh"),preview:$("studio-montage-preview"),
    start:$("studio-montage-start"),legacyStart:$("studio-render"),job:$("studio-job"),
    cancel:$("studio-cancel"),download:$("studio-download"),help:$("studio-montage-help"),
    assembly:$("studio-montage-assembly"),library:$("studio-montage-library"),
    sequence:$("studio-montage-sequence"),video:$("studio-montage-video"),
    play:$("studio-montage-play"),restart:$("studio-montage-restart"),
    playStatus:$("studio-montage-play-status"),playPosition:$("studio-montage-play-position"),
    draftCount:$("studio-montage-draft-count"),durationField:$("studio-montage-duration-field"),
    creative:$("studio-montage-creative"),creativeSource:$("studio-montage-creative-source"),
    creativeVideo:$("studio-montage-creative-video"),creativeStart:$("studio-montage-creative-start"),
    creativeEnd:$("studio-montage-creative-end"),creativeIn:$("studio-montage-creative-in"),
    creativeOut:$("studio-montage-creative-out"),creativeAdd:$("studio-montage-creative-add"),
    creativeList:$("studio-montage-creative-list"),creativeStatus:$("studio-montage-creative-status"),
    creativeKind:$("studio-montage-creative-kind"),creativeFilters:$("studio-montage-creative-filters"),
    creativeTouchKind:$("studio-montage-creative-touch-kind"),creativeTouchPlayer:$("studio-montage-creative-touch-player"),
    creativeCatalog:$("studio-montage-creative-catalog"),creativeCatalogStatus:$("studio-montage-creative-catalog-status"),
    creativeManual:$("studio-montage-creative-manual"),creativePlay:$("studio-montage-creative-play"),
    creativeDrop:$("studio-montage-creative-drop"),creativePlayStatus:$("studio-montage-creative-play-status")
  };
  if(!ui.profile)return;
  const st={project:"",token:0,busy:false,runs:new Map(),available:new Map(),fingerprint:"",key:"",
    jobId:"",timer:null,audioSeconds:null,audioURL:"",
    sourceKey:"",favoriteItems:[],order:[],previewIndex:-1,playEpoch:0,
    mediaVideo:"",advancing:false,finished:false,creativeClips:[],creativeVideos:[],creativeProject:"",creativeRows:[],creativeLoadEpoch:0,creativeDrag:null,
    creativePlayEpoch:0,creativePlayback:null,creativeMedia:"",creativeMediaPending:null};
  const option=(label,value)=>{const node=document.createElement("option");
    node.textContent=String(label);node.value=String(value);return node;};
  const validRun=value=>typeof value==="string"&&/^[a-f0-9]{32}$/.test(value);
  const selectedRuns=()=>[...st.runs.values()].filter(validRun).sort();
  const lengthSec=()=>Math.max(8,Math.min(600,Math.round(Number(ui.duration.value)||30)));
  const show=(message,error=false)=>{ui.previewList.textContent=String(message||"");
    ui.previewList.classList.toggle("track-error",error);};
  const api=async(path,opts={})=>{
    const response=await window.VolleyballStudioActivity.fetch(path,
      {credentials:"same-origin",cache:"no-store",...opts});
    const data=await response.json().catch(()=>({}));
    if(!response.ok)throw Error(data.message||data.error||"HTTP "+response.status);
    return data;
  };
  const post=(path,payload)=>api(path,{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const invalidate=()=>{st.fingerprint="";st.key="";ui.previewList.replaceChildren();
    ui.download.classList.add("hidden");ui.archiveDownload.classList.add("hidden");};
  const snapshot=()=>JSON.stringify({
    project:st.project,profile:ui.profile.value,mode:mode(),player:ui.player.value,action:ui.action.value,
    context:ui.context.value,duration:lengthSec(),runs:selectedRuns(),layout:ui.layout.value,
    camera:ui.camera.value,zoom:ui.zoom.value,encoder:ui.encoder.value,
    include_statistics:mode()==="report"&&ui.statistics.checked,
    include_recognition_data:ui.recognition.checked,
    music:ui.music.files?.[0]?.name||"",volume:ui.volume.value
  });
  const profiles={
    favorites:{mode:"favorites",label:"Избранные розыгрыши",
      hint:"Отмеченные в предпросмотре розыгрыши.",
      duration:60,layout:"original",action:"all",context:"1"},
    player:{mode:"player",label:"Рилс игрока",
      hint:"Выберите игрока и нужные типы касаний.",
      duration:30,layout:"vertical",action:"all",context:"1"},
    attacks:{mode:"player",label:"Лучшие атаки",
      hint:"Атаки выбранного игрока с контекстом до и после.",
      duration:30,layout:"vertical",action:"attack",context:"0.5"},
    creative:{mode:"creative",label:"Творчество",
      hint:"Выберите произвольные фрагменты исходных видео.",
      duration:30,layout:"original",action:"all",context:"1"},
    report:{mode:"report",label:"Лучшие и длинные розыгрыши",
      hint:"Автоматическая подборка из подтверждённых розыгрышей.",
      duration:30,layout:"original",action:"all",context:"1"}
  };
  const mode=()=>profiles[ui.profile.value]?.mode||"favorites";
  const sync=()=>{
    const person=mode()==="player",favorite=mode()==="favorites",creative=mode()==="creative";
    const selected=profiles[ui.profile.value]||profiles.favorites;
    ui.settingsLabel.textContent=selected.label;
    ui.settingsHint.textContent=selected.hint;
    ui.playerField.classList.toggle("hidden",!person);
    ui.actionField.classList.toggle("hidden",!person||ui.profile.value==="attacks");
    ui.contextField.classList.toggle("hidden",!person);
    ui.sources.classList.toggle("hidden",favorite||creative);
    ui.creative.classList.toggle("hidden",!creative);
    ui.durationField.classList.toggle("hidden",creative);
    if(!creative)cancelCreativePlayback();
    syncCreative();
    ui.statisticsField.classList.toggle("hidden",mode()!=="report");
    ui.favorites.classList.toggle("hidden",!favorite);
    ui.assembly.classList.toggle("hidden",!favorite);
    ui.play.disabled=st.busy||!favorite||!st.order.length;
    ui.restart.disabled=st.busy||!favorite||!st.order.length;
    ui.cameraField.classList.toggle("hidden",creative||ui.layout.value==="original");
    ui.zoomField.classList.toggle("hidden",creative||ui.layout.value==="original"||ui.camera.value!=="on");
    ui.start.disabled=st.busy;ui.preview.disabled=st.busy;ui.refresh.disabled=st.busy;
    ui.help.textContent=creative?"Монтаж сохраняет порядок добавленных фрагментов.":favorite?
      "Избранные розыгрыши собираются из оригинального видео.":
      person?"В рилс входят только фрагменты вокруг проверенных касаний выбранного игрока.":
      "Лучшие и длинные розыгрыши."+
        (ui.statistics.checked?" В конце — подтверждённая статистика по игрокам.":"");
  };
  const chooseProfile=profile=>{
    if(!profiles[profile])return;
    ui.profile.value=profile;
    const p=profiles[profile];
    ui.duration.value=String(p.duration);
    ui.layout.value=p.layout;ui.action.value=p.action;ui.context.value=p.context;
    ui.camera.value="off";
    if(ui.audioFit.checked&&st.audioSeconds)ui.duration.value=String(
      Math.max(8,Math.min(600,Math.round(st.audioSeconds))));
    invalidate();sync();
    if(mode()==="creative"){
      if(ui.creativeSource.value&&!st.creativeMedia)void openCreativeSource();
      void loadCreativeCatalog();
    }
  };
  const clipStart=clip=>(clip.analysis_offset_sec||0)+clip.start_frame/30;
  const clipEnd=clip=>(clip.analysis_offset_sec||0)+(clip.end_frame+1)/30;
  const creativeLabel=clip=>clip.video+" · "+clipStart(clip).toFixed(1)+" – "+clipEnd(clip).toFixed(1)+" сек.";
  const creativeButton=(label,action)=>{
    const node=document.createElement("button");node.type="button";node.className="secondary";node.textContent=label;
    node.addEventListener("click",action);return node;
  };
  const cancelCreativePlayback=()=>{
    st.creativePlayEpoch++;st.creativePlayback=null;ui.creativeVideo.pause();
    ui.creativePlayStatus.textContent=st.creativeClips.length?"Фрагментов: "+st.creativeClips.length:"Перетащите фрагменты сюда";
  };
  const ensureCreativeMedia=async(video,start=0)=>{
    if(st.creativeMedia===video){
      if(st.creativeMediaPending)await st.creativeMediaPending;
      return;
    }
    const media=ui.creativeVideo,project=st.project;
    st.creativeMedia=video;
    const pending=(async()=>{
      if(window.VolleyballBrowserVideo?.load)await window.VolleyballBrowserVideo.load(media,project,video,{seek:start,autoplay:false});
      else{media.src="/api/video-media?"+new URLSearchParams({project_id:project,video});media.load();}
    })();
    st.creativeMediaPending=pending;
    try{await pending;}catch(error){if(st.creativeMediaPending===pending)st.creativeMedia="";throw error;}
    finally{if(st.creativeMediaPending===pending)st.creativeMediaPending=null;}
  };
  const playCreative=async(clip,index=null)=>{
    const epoch=++st.creativePlayEpoch,project=st.project;
    st.creativePlayback=null;ui.creativeVideo.pause();
    const start=clipStart(clip),end=clipEnd(clip),media=ui.creativeVideo;
    if(!st.creativeVideos.includes(clip.video))return;
    ui.creativeSource.value=clip.video;
    ui.creativeStart.value=ui.creativeEnd.value="";
    try{
      await ensureCreativeMedia(clip.video,start);
      if(epoch!==st.creativePlayEpoch||project!==st.project||mode()!=="creative")return;
      if(media.readyState<1)await new Promise((resolve,reject)=>{
          const loaded=()=>{cleanup();resolve();},failed=()=>{cleanup();reject(Error("Не удалось открыть исходное видео."));};
          const cleanup=()=>{clearTimeout(timer);media.removeEventListener("loadedmetadata",loaded);media.removeEventListener("error",failed);};
          const timer=setTimeout(failed,15000);
          media.addEventListener("loadedmetadata",loaded);media.addEventListener("error",failed);
          if(media.readyState>=1)loaded();
      });
      if(epoch!==st.creativePlayEpoch||project!==st.project||mode()!=="creative")return;
      media.currentTime=start;
      st.creativePlayback={clip,index,end};
      ui.creativePlayStatus.textContent=index===null?"Предпросмотр фрагмента":(index+1)+" / "+st.creativeClips.length;
      await media.play();
    }catch(error){if(epoch===st.creativePlayEpoch){st.creativeMedia="";st.creativePlayback=null;show(error.message,true);}}
  };
  const addCreative=clip=>{
    if(st.busy||!clip||!st.creativeVideos.includes(clip.video))return;
    if(!creativeSummary(clip).canAdd){
      show("Монтаж поддерживает до 80 фрагментов и 600 секунд.",true);return;
    }
    cancelCreativePlayback();st.creativeClips.push({...clip});renderCreative();invalidate();
  };
  const renderCreativeCatalog=()=>{
    const kind=ui.creativeKind.value||"rallies",manual=kind==="manual";
    ui.creativeFilters.classList.toggle("hidden",kind!=="touches");
    ui.creativeManual.classList.toggle("hidden",!manual);
    ui.creativeCatalog.classList.toggle("hidden",manual);
    ui.creativeCatalogStatus.classList.toggle("hidden",manual);
    ui.creativeAdd.classList.toggle("hidden",!manual);
    ui.creativeStatus.classList.toggle("hidden",!manual);
    ui.creativeCatalog.replaceChildren();
    if(manual)return;
    const rows=[];
    try{
      for(let i=0;i<st.creativeRows.length;i+=512){
        const batch=st.creativeRows.slice(i,i+512);
        const indices=window.VolleyStudioChain.call(4,{rows:batch,kind,action:ui.creativeTouchKind.value,player:ui.creativeTouchPlayer.value});
        for(const index of indices)rows.push(batch[index]);
      }
    }catch(error){ui.creativeCatalogStatus.textContent=error.message;return;}
    for(const row of rows){
      const card=document.createElement("article");card.className="montage-creative-card";card.draggable=true;
      const image=document.createElement("img");image.loading="lazy";image.alt="";image.draggable=false;
      image.src="/api/video-preview?"+new URLSearchParams({video:row.clip.video,time:String(clipStart(row.clip)),width:"320"});
      const label=document.createElement("span");label.textContent=row.label+" · "+creativeLabel(row.clip);
      const actions=document.createElement("div");actions.className="montage-creative-card-actions";
      actions.append(creativeButton("Смотреть",()=>void playCreative(row.clip)),creativeButton("Добавить",()=>addCreative(row.clip)));
      card.append(image,label,actions);
      card.addEventListener("dblclick",()=>void playCreative(row.clip));
      card.addEventListener("dragstart",event=>{st.creativeDrag={row,project:st.project};event.dataTransfer?.setData("text/plain","creative-fragment");});
      card.addEventListener("dragend",()=>{st.creativeDrag=null;});
      ui.creativeCatalog.append(card);
    }
    ui.creativeCatalogStatus.textContent=rows.length?"Фрагментов: "+rows.length:"Нет подходящих фрагментов.";
  };
  const loadCreativeCatalog=async()=>{
    const epoch=++st.creativeLoadEpoch,project=st.project;
    const selection=window.VolleyballMontageSources?.();
    const sources=selection?.project_id===project?selection.sources:[...st.runs].map(([video,run])=>({video,run}));
    const valid=()=>epoch===st.creativeLoadEpoch&&project===st.project&&mode()==="creative";
    st.creativeRows=[];renderCreativeCatalog();
    if(!sources?.length)return;
    ui.creativeCatalogStatus.textContent="Загрузка фрагментов…";
    const rows=[],players=new Map();let failures=0;
    try{await window.VolleyballUi.ready;}catch(error){
      if(valid())ui.creativeCatalogStatus.textContent=error.message;
      return;
    }
    if(!valid())return;
    for(let i=0;i<sources.length;i+=4){
      const results=await Promise.allSettled(sources.slice(i,i+4).map(async source=>{
        const query=new URLSearchParams({project_id:project,rally_run_id:source.run});
        const [raw,marks]=await Promise.all([
          api("/api/cpu/native/rallies/timeline?"+query),api("/api/cpu/native/rallies/marks?"+query).catch(()=>null)
        ]);
        if(raw.schema!=="volley-native-rally-timeline/1"||raw.project_id!==project||raw.rally_run_id!==source.run||
          raw.video!==source.video||raw.fps!==30||!st.creativeVideos.includes(raw.video)||!Number.isSafeInteger(raw.total_frames))
          throw Error("Некорректный источник фрагментов");
        const base={video:raw.video,total_frames:raw.total_frames,analysis_offset_sec:Number(raw.analysis_offset_sec)||0};
        const sourceRows=[],sourcePlayers=new Map();
        const collect=input=>{
          const result=window.VolleyStudioChain.call(3,{...base,...input});
          sourceRows.push(...result.rows);
          for(const [id,name] of result.players)sourcePlayers.set(id,name);
        };
        const rallies=raw.rallies||[],touches=raw.touch_annotations||[];
        for(let i=0;i<rallies.length;i+=256)collect({rallies:rallies.slice(i,i+256),candidate_ids:marks?.candidate_ids||[]});
        for(let i=0;i<touches.length;i+=256)collect({touch_annotations:touches.slice(i,i+256)});
        for(const row of sourceRows)rows.push(row);
        for(const [id,name] of sourcePlayers)players.set(id,name);
        if(!marks)failures++;
      }));
      if(!valid())return;
      failures+=results.filter(result=>result.status==="rejected").length;
    }
    if(!valid())return;
    st.creativeRows=rows.sort((a,b)=>st.creativeVideos.indexOf(a.clip.video)-st.creativeVideos.indexOf(b.clip.video)||clipStart(a.clip)-clipStart(b.clip));
    const selected=ui.creativeTouchPlayer.value;
    ui.creativeTouchPlayer.replaceChildren(option("Любой",""),...[...players].sort((a,b)=>a[1].localeCompare(b[1])).map(([id,name])=>option(name,id)));
    ui.creativeTouchPlayer.value=players.has(selected)?selected:"";
    renderCreativeCatalog();
    if(failures)ui.creativeCatalogStatus.textContent+=" · Часть источников недоступна.";
  };
  const creativeBounds=()=>{
    if(!window.VolleyballUi?.isReady()||ui.creativeStart.value===""||ui.creativeEnd.value==="")return null;
    return window.VolleyStudioChain.call(8,{kind:"bounds",video:ui.creativeSource.value,
      videos:st.creativeVideos,duration:ui.creativeVideo.duration,
      start:Number(ui.creativeStart.value),end:Number(ui.creativeEnd.value)});
  };
  const creativeSummary=next=>window.VolleyballUi?.isReady()?
    window.VolleyStudioChain.call(8,{kind:"summary",videos:st.creativeVideos,clips:st.creativeClips,next}):
    {frames:0,canAdd:false,canExport:false};
  const creativeFrames=()=>creativeSummary().frames;
  const syncCreative=()=>{
    const clip=creativeBounds(),ready=Number.isFinite(ui.creativeVideo.duration)&&ui.creativeVideo.duration>0;
    ui.creativeIn.disabled=ui.creativeOut.disabled=st.busy||!ready;
    ui.creativePlay.disabled=st.busy||!st.creativeClips.length;
    ui.creativeAdd.disabled=st.busy||!clip||!creativeSummary(clip).canAdd;
    ui.creativeStatus.textContent=clip?
      "Длительность фрагмента: "+((clip.end_frame-clip.start_frame+1)/30).toFixed(1)+" сек.":
      "Отметьте начало и конец фрагмента.";
  };
  const openCreativeSource=async()=>{
    ui.creativeStart.value=ui.creativeEnd.value="";
    cancelCreativePlayback();
    const video=ui.creativeSource.value;
    if(!st.creativeVideos.includes(video))return;
    try{await ensureCreativeMedia(video);}catch(e){show(e.message,true);}
    syncCreative();
  };
  const renderCreative=()=>{
    ui.creativeList.replaceChildren();
    ui.creativePlayStatus.textContent=st.creativeClips.length?"Фрагментов: "+st.creativeClips.length:"Перетащите фрагменты сюда";
    st.creativeClips.forEach((clip,index)=>{
      const row=document.createElement("div");row.className="montage-creative-row";
      row.draggable=true;
      row.addEventListener("dragstart",event=>{st.creativeDrag={index,project:st.project};event.dataTransfer?.setData("text/plain","creative-order");});
      row.addEventListener("dragend",()=>{st.creativeDrag=null;});
      row.addEventListener("dragover",event=>event.preventDefault());
      row.addEventListener("drop",event=>{
        if(st.creativeDrag?.project!==st.project||!Number.isInteger(st.creativeDrag.index)||st.busy)return;
        event.preventDefault();event.stopPropagation();
        const from=st.creativeDrag.index;st.creativeDrag=null;
        if(from===index)return;
        cancelCreativePlayback();const [moved]=st.creativeClips.splice(from,1);if(moved)st.creativeClips.splice(index,0,moved);
        renderCreative();invalidate();
      });
      const label=document.createElement("span");label.textContent=(index+1)+". "+creativeLabel(clip);
      const remove=document.createElement("button");remove.type="button";remove.className="secondary";remove.textContent="Удалить";
      remove.addEventListener("click",()=>{if(st.busy)return;cancelCreativePlayback();st.creativeClips.splice(index,1);renderCreative();invalidate();});
      row.append(label,creativeButton("Смотреть",()=>void playCreative(clip)),remove);ui.creativeList.append(row);
    });
    syncCreative();
  };
  const favoriteId=id=>typeof id==="string"&&/^[a-f0-9]{24}$/.test(id);
  const draftStoreKey=()=>st.sourceKey?"volley:montage-draft:v1:"+st.sourceKey:"";
  const readDraft=()=>{
    try{
      const raw=sessionStorage.getItem(draftStoreKey());
      if(!raw||raw.length>2400)return [];
      const ids=JSON.parse(raw);
      return Array.isArray(ids)&&ids.length<=80&&ids.every(favoriteId)&&
        new Set(ids).size===ids.length?ids:[];
    }catch{return [];}
  };
  const saveDraft=()=>{
    try{if(st.sourceKey)sessionStorage.setItem(draftStoreKey(),JSON.stringify(st.order));}catch{}
  };
  const chosen=()=>st.order.map(id=>st.favoriteItems.find(item=>item.id===id)).filter(Boolean);
  window.VolleyballMontageDraft={
    ids:(project,run)=>st.sourceKey===JSON.stringify([project,run])?
      chosen().map(item=>item.id):[]
  };
  const setStatus=(message)=>{
    ui.playStatus.textContent=message;
    ui.playPosition.textContent=st.previewIndex<0?"0 / "+st.order.length:
      (st.previewIndex+1)+" / "+st.order.length;
  };
  const stopPreview=()=>{
    st.playEpoch++;st.previewIndex=-1;st.advancing=false;st.finished=false;
    st.mediaVideo="";
    if(window.VolleyballBrowserVideo?.cancel)window.VolleyballBrowserVideo.cancel(ui.video);
    else{
      ui.video.pause();
      ui.video.removeAttribute("src");
      ui.video.load();
    }
    setStatus(st.order.length?"Нажмите «Смотреть подборку» для просмотра всех фрагментов.":
      "Добавьте розыгрыши в монтаж, чтобы начать просмотр.");
  };
  const notifyDraft=()=>{
    saveDraft();stopPreview();renderAssembly();sync();
    if(typeof CustomEvent==="function")
      window.dispatchEvent(new CustomEvent("volley:montage-draft-change",
        {detail:{project_id:ui.project.value,ids:chosen().map(item=>item.id)}}));
  };
  const clipButton=(text,label,handler,disabled=false)=>{
    const button=document.createElement("button");button.type="button";
    button.className="secondary";button.textContent=text;button.title=label;
    button.setAttribute("aria-label",label);button.disabled=disabled;
    button.addEventListener("click",handler);return button;
  };
  const clipCard=(clip,index,isSequence)=>{
    const card=document.createElement("article");
    card.className="montage-clip-card"+(isSequence&&st.previewIndex===index?" active":"");
    if(isSequence){
      card.draggable=true;
      card.addEventListener("dragstart",event=>{
        event.dataTransfer?.setData("text/plain",clip.id);
        if(event.dataTransfer)event.dataTransfer.effectAllowed="move";
      });
      card.addEventListener("dragover",event=>{event.preventDefault();card.classList.add("drag-over");});
      card.addEventListener("dragleave",()=>card.classList.remove("drag-over"));
      card.addEventListener("drop",event=>{
        event.preventDefault();card.classList.remove("drag-over");
        const id=event.dataTransfer?.getData("text/plain");
        const from=st.order.indexOf(id);
        if(from<0||from===index)return;
        st.order.splice(from,1);st.order.splice(index,0,id);notifyDraft();
      });
    }
    const poster=document.createElement("div");poster.className="montage-clip-poster";
    const img=document.createElement("img");img.loading="lazy";
    img.alt="Кадр розыгрыша";
    img.src="/api/video-preview?"+new URLSearchParams({
      video:clip.video,time:String(Math.max(0,clip.start+0.2)),width:"320"
    }).toString();
    const number=document.createElement("span");number.className="montage-clip-index";
    number.textContent=isSequence?String(index+1):"★";
    poster.append(img,number);
    const body=document.createElement("div");body.className="montage-clip-body";
    const title=document.createElement("strong");title.className="montage-clip-title";
    title.textContent=clip.time;
    const detail=document.createElement("span");detail.className="montage-clip-detail";
    detail.textContent=clip.duration+" · "+clip.video;
    const controls=document.createElement("div");controls.className="montage-clip-controls";
    if(isSequence){
      controls.append(
        clipButton("←","Переместить раньше",()=>{
          if(index===0)return;
          [st.order[index-1],st.order[index]]=[st.order[index],st.order[index-1]];
          notifyDraft();
        },index===0),
        clipButton("→","Переместить позже",()=>{
          if(index===st.order.length-1)return;
          [st.order[index+1],st.order[index]]=[st.order[index],st.order[index+1]];
          notifyDraft();
        },index===st.order.length-1),
        clipButton("×","Убрать из монтажа",()=>{
          st.order.splice(index,1);notifyDraft();
        })
      );
    }else{
      const included=st.order.includes(clip.id);
      controls.append(clipButton(included?"✓ В монтаже":"+ Добавить в монтаж",
        included?"Розыгрыш уже в монтаже":"Добавить розыгрыш в монтаж",()=>{
          if(st.order.length>=80||st.order.includes(clip.id))return;
          st.order.push(clip.id);notifyDraft();
        },included||st.order.length>=80));
    }
    body.append(title,detail,controls);card.append(poster,body);
    return card;
  };
  const renderAssembly=()=>{
    ui.library.replaceChildren();ui.sequence.replaceChildren();
    for(const [index,clip] of chosen().entries())
      ui.sequence.append(clipCard(clip,index,true));
    if(!st.order.length){
      const empty=document.createElement("p");empty.className="montage-assembly-empty";
      empty.textContent="Монтажная лента пуста. Добавьте эпизоды из избранного ниже.";
      ui.sequence.append(empty);
    }
    for(const [index,clip] of st.favoriteItems.entries())
      ui.library.append(clipCard(clip,index,false));
    if(!st.favoriteItems.length){
      const empty=document.createElement("p");empty.className="montage-assembly-empty";
      empty.textContent="Нет размеченного избранного для выбранной CPU-версии. Отметьте розыгрыши в предпросмотре.";
      ui.library.append(empty);
    }
    ui.draftCount.textContent=st.order.length+" · "+chosen().reduce((total,clip)=>
      total+Math.max(0,clip.end-clip.start),0).toFixed(1)+" с";
    ui.play.disabled=st.busy||!st.order.length||mode()!=="favorites";
    ui.restart.disabled=ui.play.disabled;
  };
  const acceptFavorites=event=>{
    const data=event.detail;
    if(!data||data.project_id!==ui.project.value||!data.rally_run_id||
      (ui.run.value&&ui.run.value!==data.rally_run_id))return;
    const key=JSON.stringify([data.project_id,data.rally_run_id]);
    st.project=data.project_id;
    if(key!==st.sourceKey){
      stopPreview();st.sourceKey=key;st.order=readDraft();st.favoriteItems=[];
    }
    if(!data.ready)return;
    const items=Array.isArray(data.items)?data.items.filter(clip=>
      favoriteId(clip.id)&&typeof clip.video==="string"&&
      clip.video===data.video&&Number.isFinite(clip.start)&&
      Number.isFinite(clip.end)&&clip.start>=0&&clip.end>clip.start).slice(0,80):[];
    const ids=new Set(items.map(item=>item.id));
    const filtered=st.order.filter(id=>ids.has(id));
    if(JSON.stringify(items)===JSON.stringify(st.favoriteItems)&&
      filtered.length===st.order.length)return;
    const changed=filtered.length!==st.order.length;
    st.favoriteItems=items;st.order=filtered;
    if(changed){saveDraft();stopPreview();}
    renderAssembly();sync();
  };
  const showClip=async(index)=>{
    const clip=chosen()[index];
    if(!clip)return false;
    const epoch=++st.playEpoch;
    st.advancing=true;st.finished=false;st.previewIndex=index;
    setStatus("Загрузка розыгрыша "+(index+1)+" из "+st.order.length+"…");
    try{
      if(st.mediaVideo!==clip.video){
        st.mediaVideo=clip.video;
        if(window.VolleyballBrowserVideo?.load){
          await window.VolleyballBrowserVideo.load(ui.video,st.project,clip.video,
            {seek:clip.start,autoplay:false});
        }else{
          ui.video.src="/api/video-media?"+new URLSearchParams({
            project_id:st.project,video:clip.video
          }).toString();
          ui.video.load();
          if(ui.video.readyState<1)await new Promise((resolve,reject)=>{
            const loaded=()=>{cleanup();resolve();};
            const failed=()=>{cleanup();reject(Error("Не удалось открыть исходное видео"));};
            const cleanup=()=>{
              clearTimeout(timeout);
              ui.video.removeEventListener("loadedmetadata",loaded);
              ui.video.removeEventListener("error",failed);
            };
            const timeout=setTimeout(failed,15000);
            ui.video.addEventListener("loadedmetadata",loaded);
            ui.video.addEventListener("error",failed);
            if(ui.video.readyState>=1)loaded();
          });
        }
      }
      if(epoch!==st.playEpoch)return false;
      if(ui.video.readyState<1)await new Promise((resolve,reject)=>{
        const ready=()=>{cleanup();resolve();};
        const failed=()=>{cleanup();reject(Error("Не удалось загрузить видеокадры"));};
        const cleanup=()=>{
          clearTimeout(timeout);
          ui.video.removeEventListener("loadedmetadata",ready);
          ui.video.removeEventListener("error",failed);
        };
        const timeout=setTimeout(failed,15000);
        ui.video.addEventListener("loadedmetadata",ready);
        ui.video.addEventListener("error",failed);
        if(ui.video.readyState>=1)ready();
      });
      if(epoch!==st.playEpoch)return false;
      ui.video.currentTime=clip.start;
      ui.playPosition.textContent=(index+1)+" / "+st.order.length;
      await ui.video.play();
      if(epoch!==st.playEpoch)return false;
      setStatus("Просмотр: розыгрыш "+(index+1)+" из "+st.order.length);
      return true;
    }catch(error){
      if(epoch===st.playEpoch)setStatus("Ошибка предпросмотра: "+String(error.message||error));
      return false;
    }finally{
      if(epoch===st.playEpoch){st.advancing=false;renderAssembly();}
    }
  };
  ui.video.addEventListener("timeupdate",()=>{
    if(st.advancing||st.finished||st.previewIndex<0)return;
    const clip=chosen()[st.previewIndex];
    if(!clip)return;
    if(ui.video.currentTime+0.06>=clip.end){
      if(st.previewIndex+1<st.order.length)void showClip(st.previewIndex+1);
      else{
        st.finished=true;ui.video.pause();
        setStatus("Просмотр подборки завершён. Можно изменить порядок или запустить монтаж.");
      }
    }
  });
  ui.video.addEventListener("ended",()=>{
    if(!st.advancing&&!st.finished&&st.previewIndex>=0){
      if(st.previewIndex+1<st.order.length)void showClip(st.previewIndex+1);
      else{
        st.finished=true;setStatus("Просмотр подборки завершён.");
      }
    }
  });
  ui.play.addEventListener("click",()=>{void showClip(0);});
  ui.restart.addEventListener("click",()=>{void showClip(0);});
  window.addEventListener("volley:montage-favorites",acceptFavorites);
  renderAssembly();
  const load=async()=>{
    const token=++st.token;st.project=ui.project.value;st.runs.clear();
    st.available.clear();
    if(st.creativeProject!==st.project){
      cancelCreativePlayback();st.creativeLoadEpoch++;st.creativeMedia="";st.creativeMediaPending=null;st.creativeRows=[];st.creativeDrag=null;
      st.creativeProject=st.project;st.creativeClips=[];st.creativeVideos=[];renderCreativeCatalog();
      ui.creativeSource.replaceChildren();ui.creativeStart.value=ui.creativeEnd.value="";
      if(window.VolleyballBrowserVideo?.cancel)window.VolleyballBrowserVideo.cancel(ui.creativeVideo);
      else{ui.creativeVideo.pause();ui.creativeVideo.removeAttribute("src");ui.creativeVideo.load();}
      renderCreative();
    }
    invalidate();ui.sourceList.replaceChildren();ui.player.replaceChildren();
    if(st.sourceKey&&!st.sourceKey.startsWith(JSON.stringify([st.project]).slice(0,-1))){
      stopPreview();st.sourceKey="";st.order=[];st.favoriteItems=[];
      renderAssembly();
    }
    ui.player.append(option("Выберите игрока",""));
    if(!st.project){show("Выберите проект.");sync();return;}
    st.busy=true;sync();show("Получаем версии CPU и состав игроков...");
    try{
      const [projects,registry]=await Promise.all([
        api("/api/projects"),post("/api/players/library/native",{action:"read"})
      ]);
      if(token!==st.token)return;
      const project=(Array.isArray(projects.projects)?projects.projects:[])
        .find(item=>item.id===st.project);
      if(!project)throw Error("Проект не найден");
      const roster=Array.isArray(project.player_ids)?new Set(project.player_ids):null;
      const players=(Array.isArray(registry.players)?registry.players:[])
        .filter(row=>typeof row.id==="string"&&typeof row.name==="string"&&
          (roster===null||roster.has(row.id)));
      players.forEach(row=>ui.player.append(option(row.name,row.id)));
      if(players.length===1)ui.player.value=players[0].id;
      const videos=Array.isArray(project.videos)?project.videos:[];
      const previousSource=ui.creativeSource.value;
      st.creativeVideos=videos;ui.creativeSource.replaceChildren(...videos.map(video=>option(video,video)));
      ui.creativeSource.value=videos.includes(previousSource)?previousSource:videos[0]||"";
      if(mode()==="creative"&&ui.creativeSource.value)void openCreativeSource();
      if(videos.length>40)throw Error("Монтаж поддерживает не более 40 видео");
      const overviews=await window.VolleyballOverview.loadMany(st.project,videos);
      if(token!==st.token)return;
      for(const [video,overview] of overviews){
        const stage=(Array.isArray(overview.stages)?overview.stages:[])
          .find(row=>row.key==="rallies");
        if(stage?.truncated)throw Error("Индекс CPU-версий "+video+" неполный");
        const versions=(Array.isArray(stage?.items)?stage.items:[]).filter(row=>validRun(row.id));
        const ids=new Set(versions.map(row=>row.id));
        st.available.set(video,ids);
        const selected=ids.has(ui.run.value)?ui.run.value:null;
        if(selected)st.runs.set(video,selected);
        const row=document.createElement("div");row.className="montage-builder-source-row";
        const caption=document.createElement("span");caption.textContent=video;
        row.append(caption);ui.sourceList.append(row);
      }
      show("Найдено CPU-версий: "+selectedRuns().length+
        "; игроков: "+players.length+". Можно создавать варианты монтажа.");
    }catch(e){if(token===st.token)show(e.message,true);}
    finally{if(token===st.token){st.busy=false;sync();if(mode()==="creative")void loadCreativeCatalog();}}
  };
  const payload=(operation)=>{
    const value={
      project_id:st.project,operation,rally_run_ids:selectedRuns(),
      preferred_rally_run_id:selectedRuns().includes(ui.run.value)?ui.run.value:null,
      target_duration_sec:lengthSec(),
      layout:ui.layout.value==="vertical"&&ui.camera.value==="on"?
        "vertical_follow":ui.layout.value,
      camera_zoom:Number(ui.zoom.value),
      encoder:ui.encoder.value,audio_volume:Number(ui.volume.value),
      include_recognition_data:ui.recognition.checked
    };
    if(mode()==="report")value.include_statistics=ui.statistics.checked;
    if(mode()==="player")Object.assign(value,{
      player_id:ui.player.value,action_kind:ui.action.value,
      clip_context_sec:Number(ui.context.value),selection_mode:"auto",max_clips:80
    });
    if(operation==="start")value.expected_source_fingerprint=st.fingerprint;
    return value;
  };
  const endpoint=()=>mode()==="player"?
    "/api/montage/native/player-highlights":"/api/montage/native/game-report";
  const drawPlan=result=>{
    ui.previewList.replaceChildren();
    const person=mode()==="player";
    const clips=Number(person?result.selected_clips:result.selected_rallies);
    if(!Number.isInteger(clips)||clips<0)throw Error("Подборка недостоверна");
    const head=document.createElement("strong");
    head.textContent=person?"Коротких фрагментов: "+clips:"Розыгрышей: "+clips;
    ui.previewList.append(head);
    if(person){
      const row=document.createElement("p");
      row.textContent="Подходящих событий: "+
        Number(result.selected_matching_player_events||0)+
        " (подтверждённых касаний: "+
        result.selected_verified_player_contacts+
        "; начальных подач-кандидатов: "+
        Number(result.selected_initial_serve_candidates||0)+
        "; ручных уточнений: "+
        result.selected_manually_reviewed_actions+
        "). В рилс входят только короткие фрагменты.";
      ui.previewList.append(row);
    }else{
      const intro=document.createElement("p");
      intro.textContent="Игра: около "+Number(result.selected_content_sec||0).toFixed(1)+
        " сек."+(ui.statistics.checked?" + финальная статистика 5 сек.":"");
      ui.previewList.append(intro);
      if(ui.statistics.checked&&Array.isArray(result.statistics))for(const line of result.statistics.slice(0,15)){
        const node=document.createElement("div");node.className="montage-builder-item";
        node.textContent=String(line);ui.previewList.append(node);
      }
    }
    return clips>0;
  };
  const preview=async()=>{
    if(mode()==="creative"){
      if(!creativeSummary().canExport){show("Добавьте не менее одной секунды видео.",true);return false;}
      show("Фрагментов: "+st.creativeClips.length+" · "+(creativeFrames()/30).toFixed(1)+" сек.");return true;
    }
    if(mode()==="favorites"){
      if(!st.order.length){show("Добавьте розыгрыши в монтажную ленту.",true);return false;}
      show("В подборке "+st.order.length+" розыгрышей. Предпросмотр из оригинального видео.");
      return showClip(0);
    }
    if(!selectedRuns().length)throw Error("Нет выбранных CPU-версий");
    if(mode()==="player"&&!ui.player.value)throw Error("Выберите игрока");
    const token=st.token,key=snapshot();
    st.busy=true;sync();show("Проверяем SHA, касания и привязки...");
    try{
      const result=await post(endpoint(),payload("preview"));
      if(token!==st.token||key!==snapshot())return false;
      if(!/^[a-f0-9]{64}$/.test(result.source_fingerprint||"")||
        result.project_id!==st.project)throw Error("Неверный источник подборки");
      const hasClips=drawPlan(result);
      st.fingerprint=hasClips?result.source_fingerprint:"";st.key=hasClips?key:"";
      return hasClips;
    }finally{st.busy=false;sync();}
  };
  const uploadAudio=async()=>{
    const file=ui.music.files?.[0];if(!file)return null;
    if(file.size>90*1024*1024)throw Error("Файл музыки превышает 90 МиБ");
    const form=new FormData();form.append("file",file,file.name);
    const result=await api("/api/montage/native/audio",{method:"POST",body:form});
    if(!/^track_[a-f0-9]{32}\.(mp3|m4a|wav|flac|aac|ogg|opus)$/.test(result.audio_file||""))
      throw Error("Аудиодорожка не подтверждена сервером");
    const seconds=Number(result.duration_sec);
    if(ui.audioFit.checked&&Number.isFinite(seconds)&&seconds>0){
      st.audioSeconds=seconds;
      ui.duration.value=String(Math.min(600,Math.max(8,Math.round(seconds))));
    }
    return result.audio_file;
  };
  const stopPoll=()=>{if(st.timer!==null)clearInterval(st.timer);st.timer=null;};
  const poll=async()=>{
    if(!st.jobId)return;
    try{
      const status=await api("/api/montage/native/status");
      if(status.id!==st.jobId){
        stopPoll();ui.cancel.classList.add("hidden");
        ui.job.textContent=status.state==="idle"?
          "Монтаж не найден на сервере. Запустите его повторно.":"Запущен другой монтаж.";return;
      }
      ui.job.textContent=(status.stage||status.state||"Монтаж")+" · "+
        Math.round(Number(status.overall_percent)||0)+"%";
      if(status.state==="complete"){
        stopPoll();ui.cancel.classList.add("hidden");
        if(typeof status.media_url==="string"&&
          status.media_url.startsWith("/api/montage/native/media?")){
          ui.download.href=status.media_url;ui.download.classList.remove("hidden");
          window.dispatchEvent(new CustomEvent("volley:montage-complete",{detail:status}));
        }else{
          ui.job.textContent="Монтаж завершён, но сервер не вернул готовое видео.";
        }
        if(typeof status.archive_url==="string"&&
          status.archive_url.startsWith("/api/montage/native/media?")&&
          status.archive_url.endsWith("&archive=1")){
          ui.archiveDownload.href=status.archive_url;
          ui.archiveDownload.classList.remove("hidden");
        }
      }else if(status.state==="cancelled"||status.state==="failed"){
        stopPoll();ui.cancel.classList.add("hidden");
        ui.job.textContent=status.error||"Монтаж отменён";
      }
    }catch(e){ui.job.textContent="Монтаж: "+e.message;}
  };
  const start=async()=>{
    if(st.busy)return;
    if(mode()==="favorites"){
      if(ui.legacyStart.disabled){
        ui.job.textContent="Монтаж не запущен. Добавьте избранные розыгрыши в монтажную ленту.";
        show(ui.job.textContent,true);return;
      }
      ui.legacyStart.click();return;
    }
    if(mode()==="creative"){
      if(!await preview())return;
      const project=st.project,token=st.token,clips=st.creativeClips.map(c=>({...c}));
      const settings={audio_volume:Number(ui.volume.value),encoder:ui.encoder.value,layout:ui.layout.value,
        include_recognition_data:ui.recognition.checked};
      const fit=ui.audioFit.checked;
      st.busy=true;sync();ui.job.textContent="Загрузка музыки...";
      try{
        const audio_file=await uploadAudio();
        if(token!==st.token||project!==st.project||mode()!=="creative")return;
        const result=await post("/api/montage/native/start",{project_id:project,clips,audio_file,...settings,
          target_duration_sec:audio_file&&fit?Math.min(600,Math.max(1,st.audioSeconds||creativeFrames()/30)):null});
        if(!/^montage_[a-f0-9]{32}$/.test(result.job_id||"")||result.status!=="queued")throw Error("Монтаж не подтверждён");
        st.jobId=result.job_id;ui.cancel.classList.remove("hidden");
        window.VolleyballTasks?.set("montage",{label:"Монтаж",stage:"В очереди",progress:0});
        window.VolleyballTasks?.refresh();stopPoll();st.timer=setInterval(poll,1300);await poll();
      }catch(e){ui.job.textContent=e.message;show(e.message,true);}
      finally{if(token===st.token){st.busy=false;sync();}}
      return;
    }
    if(!st.fingerprint||st.key!==snapshot()){
      try{if(!await preview()){
        ui.job.textContent="Монтаж не запущен: нет подходящих фрагментов.";
        return;
      }}catch(e){ui.job.textContent=e.message;show(e.message,true);return;}
    }
    if(!st.fingerprint||st.key!==snapshot())return;
    st.busy=true;sync();ui.download.classList.add("hidden");
    ui.archiveDownload.classList.add("hidden");
    ui.job.textContent="Загрузка музыки...";
    try{
      const before=lengthSec();
      const audioFile=await uploadAudio();
      if(lengthSec()!==before){
        // FFprobe is authoritative when browser audio metadata was imprecise.
        const refreshed=await post(endpoint(),payload("preview"));
        if(!/^[a-f0-9]{64}$/.test(refreshed.source_fingerprint||"")||
          refreshed.project_id!==st.project)throw Error("Сервер не подтвердил обновлённую подборку");
        if(!drawPlan(refreshed))throw Error("Для длины музыки не удалось подобрать эпизоды");
        st.fingerprint=refreshed.source_fingerprint;
        st.key=snapshot();
      }
      const req=payload("start");
      req.audio_file=audioFile;
      const result=await post(endpoint(),req);
      if(!/^montage_[a-f0-9]{32}$/.test(result.job_id||"")||result.status!=="queued")
        throw Error("Монтаж не подтверждён");
      st.jobId=result.job_id;ui.cancel.classList.remove("hidden");
      window.VolleyballTasks?.set("montage",{label:"Монтаж",stage:"В очереди",progress:0});
      window.VolleyballTasks?.refresh();
      stopPoll();st.timer=setInterval(poll,1300);await poll();
    }catch(e){ui.job.textContent=e.message;}
    finally{st.busy=false;sync();}
  };
  const audioChanged=()=>{
    if(st.audioURL)URL.revokeObjectURL(st.audioURL);
    st.audioSeconds=null;st.audioURL="";invalidate();
    const file=ui.music.files?.[0];
    if(!file){ui.audioLength.textContent="Аудиодорожка не добавлена.";return;}
    const audio=document.createElement("audio");
    const url=URL.createObjectURL(file);st.audioURL=url;
    audio.preload="metadata";audio.src=url;
    audio.onloadedmetadata=()=>{
      if(st.audioURL!==url)return;
      if(!Number.isFinite(audio.duration)||audio.duration<=0)return;
      st.audioSeconds=audio.duration;
      ui.audioLength.textContent="Аудиодорожка: "+audio.duration.toFixed(1)+" сек.";
      if(ui.audioFit.checked){
        ui.duration.value=String(Math.min(600,Math.max(8,Math.round(audio.duration))));
        invalidate();
      }
    };
    audio.onerror=()=>{ui.audioLength.textContent="Не удалось прочитать длительность музыки.";};
  };
  ui.creativeKind.value="rallies";
  ui.creativeKind.addEventListener("change",()=>{renderCreativeCatalog();if(ui.creativeKind.value==="favorites")void loadCreativeCatalog();});
  ui.creativeTouchKind.addEventListener("change",renderCreativeCatalog);
  ui.creativeTouchPlayer.addEventListener("change",renderCreativeCatalog);
  ui.creativeDrop.addEventListener("dragover",event=>{if(st.creativeDrag?.project===st.project){event.preventDefault();ui.creativeList.classList.add("drag-over");}});
  ui.creativeDrop.addEventListener("dragleave",()=>ui.creativeList.classList.remove("drag-over"));
  ui.creativeDrop.addEventListener("drop",event=>{
    event.preventDefault();ui.creativeList.classList.remove("drag-over");
    const drag=st.creativeDrag;st.creativeDrag=null;
    if(drag?.project===st.project&&drag.row)addCreative(drag.row.clip);
  });
  ui.creativePlay.addEventListener("click",()=>{if(st.creativeClips.length)void playCreative(st.creativeClips[0],0);});
  const advanceCreative=()=>{
    const current=st.creativePlayback;if(!current)return;
    st.creativePlayback=null;
    if(current.index!==null&&current.index+1<st.creativeClips.length)void playCreative(st.creativeClips[current.index+1],current.index+1);
    else{ui.creativeVideo.pause();ui.creativeVideo.currentTime=Math.min(current.end,ui.creativeVideo.duration);
      ui.creativePlayStatus.textContent=current.index===null?"Фрагмент завершён":"Просмотр подборки завершён";}
  };
  ui.creativeVideo.addEventListener("timeupdate",()=>{if(st.creativePlayback&&ui.creativeVideo.currentTime>=st.creativePlayback.end-0.02)advanceCreative();});
  ui.creativeVideo.addEventListener("ended",advanceCreative);
  ui.creativeSource.addEventListener("change",openCreativeSource);
  ui.creativeVideo.addEventListener("loadedmetadata",syncCreative);
  ui.creativeVideo.addEventListener("emptied",syncCreative);
  ui.creativeVideo.addEventListener("error",()=>{st.creativeMedia="";cancelCreativePlayback();show("Не удалось открыть исходное видео.",true);});
  for(const field of [ui.creativeStart,ui.creativeEnd])field.addEventListener("input",syncCreative);
  ui.creativeIn.addEventListener("click",()=>{ui.creativeStart.value=ui.creativeVideo.currentTime.toFixed(3);syncCreative();});
  ui.creativeOut.addEventListener("click",()=>{ui.creativeEnd.value=ui.creativeVideo.currentTime.toFixed(3);syncCreative();});
  ui.creativeAdd.addEventListener("click",()=>{
    const clip=creativeBounds();if(ui.creativeAdd.disabled||!clip)return;
    addCreative(clip);
  });
  ui.profile.addEventListener("change",()=>chooseProfile(ui.profile.value));
  for(const input of [ui.player,ui.action,ui.context,ui.duration,ui.layout,ui.encoder,
    ui.volume,ui.camera,ui.zoom,ui.statistics,ui.recognition])input.addEventListener("change",()=>{invalidate();sync();});
  ui.music.addEventListener("change",audioChanged);
  ui.audioFit.addEventListener("change",()=>{
    if(ui.audioFit.checked&&st.audioSeconds){
      ui.duration.value=String(Math.min(600,Math.max(8,Math.round(st.audioSeconds))));
      invalidate();
    }
  });
  ui.refresh.addEventListener("click",load);
  ui.preview.addEventListener("click",()=>preview().catch(e=>show(e.message,true)));
  ui.start.addEventListener("click",start);
  ui.cancel.addEventListener("click",async()=>{
    if(!st.jobId)return;
    try{await post("/api/montage/native/cancel",{job_id:st.jobId});
      ui.job.textContent="Отмена отправлена...";await poll();
    }catch(e){ui.job.textContent=e.message;}
  });
  ui.project.addEventListener("change",load);
  window.addEventListener("volley:postprocessing-change",event=>{
    if(event.detail?.project_id!==st.project)return;
    if(mode()==="creative")void loadCreativeCatalog();
    const run=event.detail.rally_run_id;
    if(!validRun(run)){
      st.runs.clear();invalidate();show("Выберите CPU-версию в верхней панели.");return;
    }
    const video=[...st.available].find(([,ids])=>ids.has(run))?.[0];
    if(!video){load();return;}
    if(selectedRuns().length!==1||st.runs.get(video)!==run){
      st.runs.clear();st.runs.set(video,run);
      invalidate();show("CPU-версия изменена, проверьте подборку.");
    }
  });
  window.addEventListener("volley:montage-marks-change",event=>{
    if(event.detail?.project_id===st.project&&mode()==="creative")void loadCreativeCatalog();
  });
  window.addEventListener("volley:project-ready",()=>{if(ui.project.value!==st.project)load();});
  window.addEventListener("pagehide",()=>{
    stopPoll();if(st.audioURL)URL.revokeObjectURL(st.audioURL);
  });
  chooseProfile("favorites");
  if(ui.project.value)load();
})();
