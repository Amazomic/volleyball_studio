"use strict";
// Unified Rust-backed montage composer; never trust browser-supplied clip bounds.
(() => {
  const $=id=>document.getElementById(id);
  const ui={
    project:$("studio-project"),presets:$("studio-montage-presets"),
    mode:$("studio-montage-mode"),player:$("studio-montage-player"),action:$("studio-montage-action"),
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
    cancel:$("studio-cancel"),download:$("studio-download"),help:$("studio-montage-help")
  };
  if(!ui.mode)return;
  const st={project:"",token:0,busy:false,runs:new Map(),available:new Map(),fingerprint:"",key:"",
    jobId:"",timer:null,audioSeconds:null,audioURL:"",profile:"favorites"};
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
    project:st.project,mode:ui.mode.value,player:ui.player.value,action:ui.action.value,
    context:ui.context.value,duration:lengthSec(),runs:selectedRuns(),layout:ui.layout.value,
    camera:ui.camera.value,zoom:ui.zoom.value,encoder:ui.encoder.value,
    include_statistics:ui.mode.value==="report"&&ui.statistics.checked,
    include_recognition_data:ui.recognition.checked,
    music:ui.music.files?.[0]?.name||"",volume:ui.volume.value
  });
  const sync=()=>{
    const person=ui.mode.value==="player",favorite=ui.mode.value==="favorites";
    ui.playerField.classList.toggle("hidden",!person);
    ui.actionField.classList.toggle("hidden",!person);
    ui.contextField.classList.toggle("hidden",!person);
    ui.sources.classList.toggle("hidden",favorite);
    ui.statisticsField.classList.toggle("hidden",ui.mode.value!=="report");
    ui.favorites.classList.toggle("hidden",!favorite);
    ui.cameraField.classList.toggle("hidden",ui.layout.value==="original");
    ui.zoomField.classList.toggle("hidden",ui.layout.value==="original"||ui.camera.value!=="on");
    ui.start.disabled=st.busy;ui.preview.disabled=st.busy;ui.refresh.disabled=st.busy;
    ui.presets.querySelectorAll("[data-profile]").forEach(btn=>{
      btn.classList.toggle("active",btn.dataset.profile===st.profile);
      btn.setAttribute("aria-pressed",String(btn.dataset.profile===st.profile));
    });
    ui.help.textContent=favorite?
      "Избранные розыгрыши собираются из оригинального видео.":
      person?"В рилс входят только фрагменты вокруг проверенных касаний выбранного игрока.":
      "Лучшие и длинные розыгрыши."+
        (ui.statistics.checked?" В конце — подтверждённая статистика по игрокам.":"");
  };
  const profiles={
    favorites:{mode:"favorites",duration:60,layout:"original",action:"all",context:"1"},
    player:{mode:"player",duration:30,layout:"vertical",action:"all",context:"1"},
    attacks:{mode:"player",duration:30,layout:"vertical",action:"attack",context:"0.5"}
  };
  const chooseProfile=profile=>{
    if(!profiles[profile])return;
    st.profile=profile;
    const p=profiles[profile];
    ui.mode.value=p.mode;ui.duration.value=String(p.duration);
    ui.layout.value=p.layout;ui.action.value=p.action;ui.context.value=p.context;
    ui.camera.value="off";
    if(ui.audioFit.checked&&st.audioSeconds)ui.duration.value=String(
      Math.max(8,Math.min(600,Math.round(st.audioSeconds))));
    invalidate();sync();
  };
  const load=async()=>{
    const token=++st.token;st.project=ui.project.value;st.runs.clear();
    st.available.clear();
    invalidate();ui.sourceList.replaceChildren();ui.player.replaceChildren();
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
    finally{if(token===st.token){st.busy=false;sync();}}
  };
  const payload=(operation)=>{
    const value={
      project_id:st.project,operation,rally_run_ids:selectedRuns(),
      target_duration_sec:lengthSec(),
      layout:ui.layout.value==="vertical"&&ui.camera.value==="on"?
        "vertical_follow":ui.layout.value,
      camera_zoom:Number(ui.zoom.value),
      encoder:ui.encoder.value,audio_volume:Number(ui.volume.value),
      include_recognition_data:ui.recognition.checked
    };
    if(ui.mode.value==="report")value.include_statistics=ui.statistics.checked;
    if(ui.mode.value==="player")Object.assign(value,{
      player_id:ui.player.value,action_kind:ui.action.value,
      clip_context_sec:Number(ui.context.value),selection_mode:"auto",max_clips:80
    });
    if(operation==="start")value.expected_source_fingerprint=st.fingerprint;
    return value;
  };
  const endpoint=()=>ui.mode.value==="player"?
    "/api/montage/native/player-highlights":"/api/montage/native/game-report";
  const drawPlan=result=>{
    ui.previewList.replaceChildren();
    const person=ui.mode.value==="player";
    const clips=Number(person?result.selected_clips:result.selected_rallies);
    if(!Number.isInteger(clips)||clips<0)throw Error("Подборка недостоверна");
    const head=document.createElement("strong");
    head.textContent=person?"Коротких фрагментов: "+clips:"Розыгрышей: "+clips;
    ui.previewList.append(head);
    if(person){
      const row=document.createElement("p");
      row.textContent="Касаний выбранного типа: "+
        result.selected_verified_player_contacts+
        " (тип определён анализом; ручных уточнений: "+
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
    if(ui.mode.value==="favorites"){
      const count=Number($("studio-selected-count")?.textContent)||0;
      show(count?"Избранных розыгрышей: "+count:"Нет избранных розыгрышей.");
      return count>0;
    }
    if(!selectedRuns().length)throw Error("Нет выбранных CPU-версий");
    if(ui.mode.value==="player"&&!ui.player.value)throw Error("Выберите игрока");
    const token=st.token,key=snapshot();
    st.busy=true;sync();show("Проверяем SHA, касания и привязки...");
    try{
      const result=await post(endpoint(),payload("preview"));
      if(token!==st.token||key!==snapshot())return false;
      if(!/^[a-f0-9]{64}$/.test(result.source_fingerprint||"")||
        result.project_id!==st.project)throw Error("Неверный источник подборки");
      const hasClips=drawPlan(result);
      st.fingerprint=result.source_fingerprint;st.key=key;
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
      if(status.id!==st.jobId){stopPoll();ui.job.textContent="Запущен другой монтаж.";return;}
      ui.job.textContent=(status.stage||status.state||"Монтаж")+" · "+
        Math.round(Number(status.overall_percent)||0)+"%";
      if(status.state==="complete"){
        stopPoll();ui.cancel.classList.add("hidden");
        if(typeof status.media_url==="string"&&
          status.media_url.startsWith("/api/montage/native/media?")){
          ui.download.href=status.media_url;ui.download.classList.remove("hidden");
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
    if(ui.mode.value==="favorites"){
      if(ui.legacyStart.disabled){show("Отметьте розыгрыши в предпросмотре.",true);return;}
      ui.legacyStart.click();return;
    }
    if(!st.fingerprint||st.key!==snapshot()){
      try{if(!await preview())return;}catch(e){show(e.message,true);return;}
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
  ui.presets.addEventListener("click",e=>{
    const button=e.target.closest("[data-profile]");
    if(button)chooseProfile(button.dataset.profile);
  });
  ui.mode.addEventListener("change",()=>{
    st.profile=ui.mode.value==="favorites"?"favorites":ui.mode.value==="report"?"report":"player";
    invalidate();sync();
  });
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
  window.addEventListener("volley:project-ready",()=>{if(ui.project.value!==st.project)load();});
  window.addEventListener("pagehide",()=>{
    stopPoll();if(st.audioURL)URL.revokeObjectURL(st.audioURL);
  });
  chooseProfile("favorites");
  if(ui.project.value)load();
})();
