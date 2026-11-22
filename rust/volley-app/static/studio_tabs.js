"use strict";
// Browser-only DOM and same-origin network adapters. Rust/WASM validates
// navigation, language, settings and async load epochs; native Rust owns
// evidence checksums, stage isolation and revision-safe mutations.
(() => {
  const $ = id => document.getElementById(id);
  const ui = {
    preview:$("studio-preview-view"),montage:$("studio-montage-view"),
    analysis:$("studio-analysis-view"),
    players:$("studio-players-view"),report:$("studio-report-view"),quick:$("studio-quick-view"),
    project:$("studio-project"),video:$("studio-analysis-video"),
    note:$("studio-analysis-note"),start:$("studio-analysis-start"),
    gpuOnly:$("studio-analysis-gpu-only"),postOnly:$("studio-analysis-post-only"),
    reset:$("studio-analysis-reset"),
    progress:$("studio-analysis-progress"),profileDelta:$("studio-analysis-profile-delta"),
    logPanel:$("studio-analysis-log-panel"),logOutput:$("studio-analysis-log-output"),
    logCurrent:$("studio-analysis-log-current"),logCount:$("studio-analysis-log-count"),
    historyList:$("studio-history-list"),historyRefresh:$("studio-history-refresh"),historyMore:$("studio-history-more"),historyClear:$("studio-history-clear"),
    playerForm:$("studio-player-create"),playerName:$("studio-player-name"),
    playerList:$("studio-player-list"),playerCount:$("studio-player-count"),
    playerRefresh:$("studio-player-refresh")
  };
  const runtime=window.VolleyballUi;
  if(!runtime?.ready)throw Error("Studio UI initialization missing");
  const state={
    tab:"preview",analysisGeneration:0,playerGeneration:0,formProject:"",
    pipelineId:"",pipelineProject:"",pipelineTimer:null,
    starting:false,stopRequested:false,polling:false,pollErrors:0,completedId:"",
    logKey:"",logOffset:0,logBusy:false,logLastCall:"",logPending:null,
    historyRows:[],historyFilter:"all",historyEpoch:0,historyOffset:0,latestPipeline:null,
    revision:"absent",players:[],mutating:false,playersLoaded:false,
    projectMemberIds:new Set(),trackCoverScope:"",trackCovers:new Map(),playerTrackCounts:null
  };
  // Per-project form snapshot: store only deviations from the versioned gold
  // defaults. A changed native default never gets silently overwritten.
  const analysisFormCache=(()=>{
    const prefix="volley:studio-analysis-form:v1:";
    const read=project=>{
      if(!/^[a-f0-9]{12}$/.test(project||""))return null;
      try{
        const raw=localStorage.getItem(prefix+project);
        if(!raw||raw.length>8192)return null;
        const data=JSON.parse(raw);
        if(!data||![1,2,3].includes(data.version)||!data.fields||typeof data.fields!=="object")return null;
        if(data.version!==3){
          const fields={...data.fields};
          delete fields["studio-post-require-pose"];
          const migrated={version:3,video:data.video,fields};
          localStorage.setItem(prefix+project,JSON.stringify(migrated));
          return migrated;
        }
        return data;
      }catch{return null;}
    };
    const write=(project,video,fields)=>{
      if(!/^[a-f0-9]{12}$/.test(project||""))return;
      try{localStorage.setItem(prefix+project,JSON.stringify({
        version:3,video,fields
      }));}catch{}
    };
    return {read,write};
  })();
  const make=(tag,klass,text)=>{
    const node=document.createElement(tag);
    if(klass)node.className=klass;
    if(text!==undefined)node.textContent=String(text);
    return node;
  };
  const clear=node=>{while(node.firstChild)node.firstChild.remove();};
  const api=async(path,opts={})=>{
    const res=await window.VolleyballStudioActivity.fetch(path,{cache:"no-store",credentials:"same-origin",...opts});
    const data=await res.json().catch(()=>({}));
    if(!res.ok){
      const msg=typeof data.message==="string"?data.message:
        typeof data.error==="string"?data.error:"HTTP "+res.status;
      throw Error(msg);
    }
    return data;
  };
  const get=(path,params)=>api(path+"?"+new URLSearchParams(params));
  const post=(path,body)=>api(path,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)});
  const requestId=()=>{
    const id=window.crypto?.randomUUID?.().replace(/-/g,"");
    if(!/^[a-f0-9]{32}$/.test(id||""))
      throw Error("Для безопасного запуска нужен криптографический идентификатор запроса");
    return id;
  };
  // A lost POST response is ambiguous: the Rust pipeline may already be
  // queued. Check the durable server status before reporting an error.
  const startWithRecovery=async payload=>{
    try{return await post("/api/studio/pipeline/start",payload);}
    catch(error){
      if(!/failed to fetch|networkerror|network request failed|load failed/i.test(
        String(error?.message||error)))throw error;
      saveMessage("Связь прервана. Проверяем, был ли запуск принят сервером…");
      for(let attempt=0;attempt<5;attempt++){
        try{
          const status=await get("/api/studio/pipeline/status",{
            project_id:payload.project_id
          });
          if(status.schema==="volley-studio-pipeline/1"&&
              status.project_id===payload.project_id&&
              status.client_request_id===payload.client_request_id&&
              status.video===payload.video&&status.mode===payload.mode&&
              /^[a-f0-9]{32}$/.test(String(status.id||""))){
            return {schema:status.schema,id:status.id,status:status.status,recovered:true};
          }
        }catch{/* WSL port forwarding may briefly disconnect. */}
        if(attempt<4)await new Promise(resolve=>setTimeout(resolve,500));
      }
      // Same client token: the Rust endpoint is idempotent even if the
      // original POST was accepted but its response disappeared.
      try{return await post("/api/studio/pipeline/start",payload);}
      catch{
        throw Error("Связь с сервером потеряна. Запуск мог продолжиться. "+
          "Обновите страницу, чтобы восстановить задачу, прежде чем запускать её снова.");
      }
    }
  };
  const announce=(text,error=false)=>{
    const banner=$("studio-banner");
    banner.textContent=String(text);
    banner.className=error?"banner error":"banner ok";
  };
  const failure=e=>announce(e instanceof Error?e.message:String(e),true);
  const stopPipelinePolling=()=>{
    if(state.pipelineTimer!==null)clearInterval(state.pipelineTimer);
    state.pipelineTimer=null;
  };
  // Keep every media element scoped to its visible workspace. Hidden views
  // stay mounted for fast navigation but must never keep playing.
  const views={
    preview:ui.preview,training:ui.preview,montage:ui.montage,quick:ui.quick,
    analysis:ui.analysis,players:ui.players,report:ui.report
  };
  const pauseInactiveMedia=(activeTab)=>{
    const active=views[activeTab];
    for(const media of document.querySelectorAll("video, audio")){
      if(!media.paused&&(document.hidden||!active?.contains(media)))media.pause();
    }
  };
  const switchTab=(requested)=>{
    const previous=state.tab;
    const next=runtime.navigate(requested);
    if(previous!==next)runtime.beginLoad(previous);
    state.tab=next;
    ui.preview.classList.toggle("hidden",next!=="preview"&&next!=="training");
    ui.preview.classList.toggle("training-workspace",next==="training");
    ui.montage.classList.toggle("hidden",next!=="montage");
    ui.quick.classList.toggle("hidden",next!=="quick");
    ui.analysis.classList.toggle("hidden",next!=="analysis");
    ui.players.classList.toggle("hidden",next!=="players");
    ui.report.classList.toggle("hidden",next!=="report");
    pauseInactiveMedia(next);
    window.dispatchEvent(new CustomEvent("volley:tab-change",{detail:{tab:next}}));
    for(const nav of document.querySelectorAll("[data-studio-tab]")){
      const active=nav.dataset.studioTab===next;
      nav.classList.toggle("active",active);
      if(active)nav.setAttribute("aria-current","page");
      else nav.removeAttribute("aria-current");
    }
    if(next==="analysis"&&state.formProject!==projectId())loadVideos().catch(failure);
    if(next==="players"&&!state.playersLoaded)loadPlayers().catch(failure);
  };
  // Browser backgrounding and navigation never preserve active playback.
  // A capture listener also guards delayed play() calls from hidden panels.
  document.addEventListener("visibilitychange",()=>{
    if(document.hidden)pauseInactiveMedia(null);
  });
  window.addEventListener("pagehide",()=>pauseInactiveMedia(null));
  document.addEventListener("play",event=>{
    const media=event.target;
    if(media?.tagName!=="VIDEO"&&media?.tagName!=="AUDIO")return;
    if(document.hidden||!views[state.tab]?.contains(media))media.pause();
  },true);
  const projectId=()=>ui.project.value;
  const inputValue=(id)=>{
    const input=$(id),raw=input.value.trim(),value=Number(raw);
    const low=Number(input.min),high=Number(input.max);
    if(!raw||!Number.isFinite(value)||value<low||value>high||
        (input.step==="1"&&!Number.isSafeInteger(value))||!input.checkValidity()){
      throw Error(input.labels?.[0]?.textContent?.trim()+": недопустимое значение");
    }
    return value;
  };
  const refreshPersonSamplingRate=()=>{
    const step=Number($("studio-gpu-person-stride").value);
    const rate=$("studio-gpu-person-rate");
    if(!Number.isSafeInteger(step)||step<1||step>30){
      rate.textContent="Видео: 30 кадров в секунду (30 FPS). Укажите шаг от 1 до 30.";
      return;
    }
    const frequency=30/step;
    const count=frequency.toLocaleString("ru-RU",{maximumFractionDigits:2});
    const form=Number.isInteger(frequency)&&frequency%100>=11&&frequency%100<=14?"раз":
      Number.isInteger(frequency)&&frequency%10===1?"раз":
      Number.isInteger(frequency)&&frequency%10>=2&&frequency%10<=4?"раза":
      Number.isInteger(frequency)?"раз":"раза";
    rate.textContent="Видео: 30 кадров в секунду (30 FPS). Каждый "+step+
      "-й кадр → "+count+" "+form+"/с.";
  };
  const gpuProfile=()=>({
      stride:inputValue("studio-gpu-stride"),
      gpu_memory_limit_mb:inputValue("studio-gpu-memory"),
      cpu_threads:inputValue("studio-gpu-threads"),
      pose_enabled:$("studio-gpu-pose-enabled").value==="1",
      pose_mode:"photo",
      pose_stride:inputValue("studio-gpu-person-stride"),
      primary_rallies:{
        min_trajectory_seconds:inputValue("studio-gpu-rally-min-seconds"),
        merge_gap_seconds:inputValue("studio-gpu-rally-merge-seconds"),
        stationary_ball_ms:inputValue("studio-gpu-rally-stationary-ms")
      }
  });
  // Mirrors volley-core's validated native defaults (no JS-only processing).
  const postProfile=()=>({
    interpolate_players:$("studio-gpu-interpolate-players").checked,
    interpolate_pose:$("studio-gpu-interpolate-pose").checked,
    pose_classification:true,
    auto_assign_threshold_percent:inputValue("studio-post-auto-assign-threshold"),
    tracking:{
      min_player_score:inputValue("studio-post-player-score"),
      nms_iou:inputValue("studio-post-nms"),
      max_gap_frames:inputValue("studio-post-max-gap"),
      lost_low_confidence_max_gap_frames:inputValue("studio-post-lost-low-gap"),
      lost_low_confidence_min_iou:inputValue("studio-post-lost-low-iou"),
      min_track_observations:inputValue("studio-post-min-track-frames"),
      min_assignment_score:inputValue("studio-post-assignment"),
      ambiguity_margin:inputValue("studio-post-margin")
    },
    touches:{
      algorithm:$("studio-post-tcn-experimental").checked?"trajectory_tcn":"pose_motion", // PoseMotion default; explicit experimental opt-in.
      threshold:inputValue("studio-post-touch-threshold"),
      min_ball_score:inputValue("studio-post-ball-score"),
      smooth:inputValue("studio-post-smooth"),
      min_gap:inputValue("studio-post-touch-gap"),
      max_reach_ratio:inputValue("studio-post-reach"),
      min_interval_sec:inputValue("studio-post-interval"),
      require_pose:true,
      pose_min_confidence:inputValue("studio-post-pose-confidence"),
      pose_max_distance:inputValue("studio-post-pose-distance"),
      occluded_min_turn_deg:inputValue("studio-post-occluded-turn")
    },
    rallies:{
      min_ball_score:inputValue("studio-post-ball-score"),
      max_ball_gap_frames:inputValue("studio-post-ball-gap"),
      max_idle_frames:inputValue("studio-post-idle"),
      stationary_px_per_frame:inputValue("studio-post-stationary"),
      min_rally_frames:inputValue("studio-post-rally-min"),
      min_ball_samples:inputValue("studio-post-samples"),
      min_ball_coverage:inputValue("studio-post-coverage"),
      min_touch_candidates:inputValue("studio-post-min-touches"),
      max_without_touch_frames:inputValue("studio-post-no-touch"),
      suggested_context_frames:inputValue("studio-post-context")
    }
  });
  const allProfileFields=()=>[...ui.analysis.querySelectorAll("input[data-default],select[data-default]")];
  const fieldValue=input=>input.type==="checkbox"?(input.checked?"1":"0"):input.value;
  const applyFieldValue=(input,value)=>{
    if(input.type==="checkbox")input.checked=value==="1";
    else input.value=value;
  };
  const saveAnalysisSettings=()=>{
    const fields={};
    for(const input of allProfileFields()){
      if(fieldValue(input)!==input.dataset.default)
        fields[input.id]=[input.dataset.default,fieldValue(input)];
    }
    analysisFormCache.write(projectId(),ui.video.value,fields);
  };
  const restoreAnalysisSettings=project=>{
    const saved=analysisFormCache.read(project);
    for(const input of allProfileFields()){
      applyFieldValue(input,input.dataset.default);
      const row=saved?.fields?.[input.id];
      if(Array.isArray(row)&&row.length===2&&
          (row[0]===input.dataset.default||
           (input.id==="studio-gpu-threads"&&row[0]==="8"&&input.dataset.default==="0"))&&typeof row[1]==="string"&&
          row[1].length<=64){
        applyFieldValue(input,row[1]);
        if(!input.checkValidity())applyFieldValue(input,input.dataset.default);
      }
    }
    return typeof saved?.video==="string"?saved.video:"";
  };
  const selectedGpuSource=()=>{
    const raw=$("studio-inference")?.value;
    if(!raw)return "";
    try{
      const tuple=JSON.parse(raw);
      return Array.isArray(tuple)&&tuple.length===3&&tuple[0]===ui.video.value&&
        tuple[1]==="fusion"&&/^[a-f0-9]{32}$/.test(tuple[2])?tuple[2]:"";
    }catch{return "";}
  };
  const updateProfile=()=>{
    refreshPersonSamplingRate();
    let gpuValid=true,postValid=true,invalid="";
    try{gpuProfile();}catch(e){gpuValid=false;invalid=String(e.message||e);}
    try{postProfile();}catch(e){postValid=false;invalid=String(e.message||e);}
    const valid=gpuValid&&postValid;
    if(!valid)ui.profileDelta.textContent=invalid;
    if(valid){
      const changed=allProfileFields().filter(item=>fieldValue(item)!==item.dataset.default);
      ui.profileDelta.textContent=changed.length?
        "Изменено параметров: "+changed.length+
        ". При изменении GPU создаётся новая 1 фаза; при изменении только CPU — новая 2 фаза.":
        "Эталонный профиль активен. Совпадающие готовые версии используются повторно.";
      if($("studio-post-tcn-experimental").checked){
        ui.profileDelta.textContent+=" Экспериментальная TCN: требуется обученная модель models/volley3d_touch_tcn.onnx.";
      }
    }
    const running=Boolean(state.pipelineId);
    ui.start.textContent=running?
      (state.stopRequested?"Остановка анализа…":"■ Остановить анализ"):
      "▶ Полный анализ";
    ui.start.classList.toggle("is-stop",running);
    ui.start.disabled=state.starting||(running?state.stopRequested:
      (!valid||!projectId()||!ui.video.value));
    const blocked=running||state.starting||!projectId()||!ui.video.value;
    // Primary analysis starts above the history panel. Show live progress on
    // that very button so large FFmpeg preparations do not look like a no-op.
    const live=state.latestPipeline;
    const primaryRunning=running&&state.pipelineProject===projectId()&&
      live?.id===state.pipelineId&&live.mode==="gpu"&&
      ["queued","running","cancelling"].includes(live.status);
    if(state.starting&&!running){
      ui.gpuOnly.textContent="… Создание задачи";
    }else if(primaryRunning){
      const total=Number(live.progress_percent);
      const measured=Number.isFinite(total)?Math.max(0,Math.min(100,total)):null;
      const preparing=live.stage==="Подготовка исходного видео";
      const phase=preparing?"Подготовка":
        /V[Bb]allNet/.test(live.stage||"")?"Мяч":
        /RF-DETR|RTMPose|RTMDet/.test(live.stage||"")?"Игроки":"1 фаза";
      const shown=measured===null?"…":
        (preparing?Math.min(100,measured/8*100):measured).toFixed(1)+"%";
      ui.gpuOnly.textContent="◌ "+phase+" · "+shown;
      ui.gpuOnly.title=String(live.stage||"Выполняется")+
        (measured===null?"":" · "+measured.toFixed(1)+"% всего");
    }else{
      ui.gpuOnly.textContent="▶ Начать анализ";
      ui.gpuOnly.title="Запустить 1 фазу (GPU) и создать отдельную версию";
    }
    ui.gpuOnly.disabled=blocked||!gpuValid;
    ui.postOnly.disabled=blocked||!postValid||!selectedGpuSource();
    for(const [id,button] of [["analysis-flow-gpu-start",ui.gpuOnly],
      ["analysis-flow-cpu-start",ui.postOnly],["analysis-flow-full-start",ui.start]]){
      const node=$(id);
      node.disabled=button.disabled;
      if(id==="analysis-flow-full-start"){
        node.textContent=running?ui.start.textContent:"▶ Полный анализ";
        node.classList.toggle("is-stop",running);
      }
    }
    ui.postOnly.title=selectedGpuSource()?
      "Создать новый CPU-вариант выбранного GPU-анализа без повторного GPU":
      "Сначала выберите объединённый GPU-анализ в шапке";
  };
  const saveMessage=(message,error=false)=>{
    ui.note.textContent=message;
    ui.note.classList.toggle("analysis-error",error);
  };
  const resetLog=()=>{
    state.logKey="";state.logOffset=0;state.logLastCall="";state.logPending=null;
    ui.logCount.textContent="0 записей";
    ui.logCurrent.textContent="Ожидание запуска";
    ui.logOutput.textContent="Пока нет записей.";
  };
  const formatLogRow=entry=>{
    const date=new Date(Number(entry.timestamp_ms));
    const time=Number.isFinite(date.getTime())?date.toLocaleTimeString("ru-RU",{hour12:false}):"--:--:--";
    const label=String(entry.level||"info").toUpperCase();
    const stage=pipelineNames.get(pipelineId(String(entry.stage||"")))||
      String(entry.stage||"Обработка");
    const message=String(entry.message||"");
    const heading="["+time+"] "+label+" ["+stage+"] "+message;
    const call=entry.call;
    if(!call||typeof call!=="object")return heading;
    if(call.type==="process"){
      const argv=Array.isArray(call.argv)?call.argv.map(String):[];
      return heading+"\n  EXEC: "+String(call.name||argv[0]||"process")+
        "\n  argv: "+JSON.stringify(argv,null,2);
    }
    return heading+"\n  RUST: "+String(call.name||"unknown")+
      "\n  args: "+JSON.stringify(call.arguments??{},null,2);
  };
  const loadLog=async(status)=>{
    const project=String(status.project_id||""),id=String(status.id||"");
    if(!project||!id||project!==projectId())return;
    if(state.logBusy){state.logPending=status;return;}
    const key=project+":"+id;
    if(state.logKey!==key){
      state.logKey=key;state.logOffset=0;state.logLastCall="";
      ui.logCount.textContent="0 записей";
      ui.logOutput.textContent="Журнал загружается…";
    }
    const after=state.logOffset;
    let more=false;
    state.logBusy=true;
    try{
      const data=await get("/api/studio/pipeline/log",{project_id:project,id,after});
      if(state.logKey!==key||projectId()!==project)return;
      if(data.schema!=="volley-studio-pipeline-log/1"||data.id!==id||
        data.project_id!==project||!Array.isArray(data.entries)||
        !Number.isSafeInteger(data.next_offset)||data.next_offset<after)
        throw Error("Некорректный ответ журнала");
      const atBottom=ui.logOutput.scrollTop+ui.logOutput.clientHeight>=
        ui.logOutput.scrollHeight-55;
      const additions=[];
      for(const entry of data.entries){
        additions.push(formatLogRow(entry));
        if(entry.call&&typeof entry.call==="object")
          state.logLastCall=entry.call.type==="process"?
            "FFmpeg (внешний процесс)":"Rust: "+String(entry.call.name||"unknown");
      }
      if(additions.length){
        if(after===0)ui.logOutput.textContent="";
        ui.logOutput.textContent+=(ui.logOutput.textContent?"\n\n":"")+additions.join("\n\n");
        if(atBottom)ui.logOutput.scrollTop=ui.logOutput.scrollHeight;
      }else if(after===0){
        ui.logOutput.textContent="Ожидание первых событий сервера…";
      }
      state.logOffset=data.next_offset;
      more=data.has_more===true&&data.entries.length>0;
      ui.logCount.textContent=state.logOffset+" записей"+(data.has_more?" · загрузка…":"");
      ui.logCurrent.textContent="Стадия: "+String(status.stage||"—")+
        " · "+(state.logLastCall||"ожидание вызова");
    }catch(error){
      if(state.logKey===key){
        ui.logCurrent.textContent="Журнал: "+String(error.message||error);
      }
    }finally{state.logBusy=false;}
    const pending=state.logPending;state.logPending=null;
    if(pending&&pending.project_id===projectId())await loadLog(pending);
    else if(more&&state.logKey===key)await loadLog(status);
  };

  const pipelineStages=[
    ["video","Подготовка видео","input",1,1],
    ["ball","Детекция мяча","gpu",1,2],
    ["rallies","Розыгрыши","gpu",1,3],
    ["players","Поиск игроков","gpu",1,4],
    ["fusion","Данные GPU","gpu",1,5],
    ["pose","Позы игроков","gpu",1,6],
    ["tracking","Трекинг ID","cpu",2,1],
    ["touches","Касания","cpu",2,2],
    ["touch_types","Типы касаний","cpu",2,3],
    ["publish","Публикация","cpu",2,4],
    ["identity","Привязка игроков","cpu",2,5],
    ["material","Результат","output",2,6]
  ];
  const pipelineNames=new Map(pipelineStages.map(item=>[item[0],item[1]]));
  pipelineNames.set("cache","Кэш GPU");
  const pipelineModelMeta=(id,run)=>{
    const tcn=run?.profile?.post?.touches?.algorithm==="trajectory_tcn"||
      (!run&&$("studio-post-tcn-experimental")?.checked);
    const known={
      ball:["VballNetV4c","model"],
      players:["RF-DETR Medium","model"],
      pose:["RTMPose-M Body7","model"],
      tracking:["ByteTrack","algorithm"],
      touch_types:tcn?["Volley3D TCN","model"]:["PoseMotion","algorithm"]
    };
    return known[id]||null;
  };
  const pipelineGlyphs={
    video:[["rect",{x:2,y:5,width:14,height:14,rx:3}],["path",{d:"m16 10 6-4v12l-6-4"}]],
    ball:[["circle",{cx:12,cy:12,r:10}],["path",{d:"M2.8 7.4c5.2-1 10.7 2.8 11 7.5m5.4-11.1C14 8 12 15 14.6 21.6M4.1 18.2c4.1-4.4 9.2-5.5 17.4-4.5"}]],
    rallies:[["path",{d:"m12 2 10 6-10 6L2 8l10-6Zm-10 11 10 6 10-6M2 18l10 6 10-6"}]],
    players:[["circle",{cx:9,cy:7,r:3}],["path",{d:"M2 21v-2a7 7 0 0 1 14 0v2M17 4a4 4 0 0 1 0 7m4 10v-2a6 6 0 0 0-4-5.7"}]],
    pose:[["circle",{cx:12,cy:3,r:2}],["path",{d:"M12 5v7m0 0-7 9m7-9 7 9M5 8l7 4 7-4M5 8l-2 6m16-6 2 6"}]],
    fusion:[["rect",{x:5,y:5,width:14,height:14,rx:2}],["path",{d:"M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"}]],
    tracking:[["rect",{x:2,y:3,width:20,height:18,rx:3}],["circle",{cx:8,cy:9,r:2}],["path",{d:"M4 17c1-3 7-3 8 0m4-9h4m-4 5h4m-4 5h3"}]],
    touches:[["path",{d:"M8 22V11a2 2 0 0 1 4 0V5a2 2 0 0 1 4 0v6a2 2 0 0 1 4 0v5a7 7 0 0 1-7 7h-2a8 8 0 0 1-6-3l-3-4a2 2 0 0 1 3-3l3 3"}]],
    touch_types:[["circle",{cx:11,cy:13,r:8}],["path",{d:"m5 8 6 2 4-4m-7 9 3-5 8 3m-8 2-4 6m4-11 1 11M19 1v4m-2-2h4"}]],
    publish:[["path",{d:"M12 16V2m-5 5 5-5 5 5M3 15v6h18v-6"}]],
    identity:[["path",{d:"M10 13a5 5 0 0 0 7 .5l3-3a5 5 0 0 0-7-7l-2 2m3 5a5 5 0 0 0-7-.5l-3 3a5 5 0 0 0 7 7l2-2"}]],
    material:[["path",{d:"M4 20V12m5 8V6m5 14v-9m5 9V3M2 22h20"}]]
  };
  const pipelineIcon=id=>{
    const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
    svg.setAttribute("class","analysis-flow-node-icon");
    svg.setAttribute("viewBox","0 0 24 24");
    svg.setAttribute("fill","none");
    svg.setAttribute("stroke","currentColor");
    svg.setAttribute("stroke-width","1.7");
    svg.setAttribute("stroke-linecap","round");
    svg.setAttribute("stroke-linejoin","round");
    svg.setAttribute("aria-hidden","true");
    for(const [tag,attributes] of pipelineGlyphs[id]||[]){
      const part=document.createElementNS("http://www.w3.org/2000/svg",tag);
      for(const [key,value] of Object.entries(attributes))part.setAttribute(key,String(value));
      svg.append(part);
    }
    return svg;
  };
  const pipelineId=entry=>{
    if(entry?.id&&pipelineNames.has(entry.id))return entry.id;
    const name=String(typeof entry==="string"?entry:entry?.name||"");
    if(/Проверка исходного видео|Подготовка исходного видео/.test(name))return "video";
    if(/Поиск розыгрышей/.test(name))return "rallies";
    if(/GPU.*V[Bb]allNet/.test(name))return "ball";
    if(/RTMPose/.test(name)&&!/RF-DETR/.test(name))return "pose";
    if(/Объединение/.test(name))return "fusion";
    if(/RF-DETR|RTMDet|игровые интервалы|общий поток/.test(name))return "players";
    if(/GPU-анализ завершён/.test(name))return "fusion";
    if(/Track ID/.test(name))return "tracking";
    if(/касания в первичных|физические касания/.test(name))return "touches";
    if(/Публикация розыгрышей/.test(name))return "publish";
    if(/автоматическая привязка/.test(name))return "identity";
    if(/Монтажная временная разметка/.test(name))return "material";
    if(/сохранённые версии|Сохранённый GPU-анализ|сохранённый GPU-анализ/.test(name))return "cache";
    return "";
  };
  const pipelineLabel=entry=>window.VolleyballUi.translate(pipelineNames.get(pipelineId(entry))||String(entry?.name||"Стадия"));
  const stageTime=ms=>{
    if(!Number.isFinite(ms)||ms<0)return "—";
    const seconds=Math.floor(ms/1000);
    return String(Math.floor(seconds/3600)).padStart(2,"0")+":"+
      String(Math.floor(seconds/60)%60).padStart(2,"0")+":"+
      String(seconds%60).padStart(2,"0");
  };
  const stageResultText=(id,entry,run,now,done,running)=>{
    if(id==="pose"&&!entry&&(run?.gpu_jobs?.pose||running)){
      const shared=run?.shared_pose;
      const player=run?.stages?.find(stage=>stage.id==="players");
      const joint=shared?.source_stage_id==="players"||/RF-DETR.*RTMPose/.test(String(player?.name||""));
      if(!joint)return "Позы сохранены · время не измерено";
      const duration=Number.isFinite(shared?.duration_ms)?shared.duration_ms:
        running&&Number.isFinite(player?.started_ms)?Math.max(0,now-player.started_ms):
        player?.duration_ms;
      const count=shared?.sampled_frames,poses=shared?.player_poses;
      if(Number.isSafeInteger(count)&&count>=0&&Number.isSafeInteger(poses)&&poses>=0)
        return count.toLocaleString("ru-RU")+" кадр. · "+poses.toLocaleString("ru-RU")+
          " поз · "+stageTime(duration)+" (общ.)";
      return "С RF-DETR · "+stageTime(duration);
    }
    if(id==="touch_types"&&entry&&run?.profile?.post?.pose_classification!==false){
      const duration=running&&Number.isFinite(entry.started_ms)&&entry.finished_ms==null?
        Math.max(0,now-entry.started_ms):entry.duration_ms;
      return "В составе касаний · "+stageTime(duration);
    }
    let result="";
    if(id==="rallies"&&entry&&Array.isArray(run?.gpu_rally_mask?.merged)){
      const count=run.gpu_rally_mask.merged.length;
      const noun=count%10===1&&count%100!==11?"розыгрыш":
        count%10>=2&&count%10<=4&&(count%100<12||count%100>14)?"розыгрыша":"розыгрышей";
      result=count.toLocaleString("ru-RU")+" "+noun;
    }else if(entry&&Number.isSafeInteger(entry.frames_processed)&&entry.frames_processed>=0&&
        Number.isSafeInteger(entry.frames_total)&&entry.frames_total>0){
      result=entry.frames_processed.toLocaleString("ru-RU")+" / "+
        entry.frames_total.toLocaleString("ru-RU")+" кадров";
    }else if(done)result="Готово";
    else if(running)result="Выполняется";
    if(!result)return "—";
    const duration=running&&Number.isFinite(entry?.started_ms)&&entry?.finished_ms==null?
      Math.max(0,now-entry.started_ms):entry?.duration_ms;
    return result+" · "+stageTime(duration);
  };
  const stageProgressPercent=(entry,running,done)=>{
    if(done)return 100;
    if(!running)return 0;
    const reported=entry?.progress_percent;
    if(typeof reported==="number"&&Number.isFinite(reported))
      return Math.max(0,Math.min(100,reported));
    const processed=entry?.frames_processed,total=entry?.frames_total;
    if(typeof processed==="number"&&Number.isFinite(processed)&&
      typeof total==="number"&&Number.isFinite(total)&&total>0)
      return Math.max(0,Math.min(100,processed/total*100));
    return null;
  };
  const pipelineEdges=[
    ["video","ball"],["ball","rallies"],["rallies","players"],["players","fusion"],
    ["fusion","pose"],["pose","tracking"],["tracking","touches"],
    ["touches","touch_types"],["touch_types","publish"],
    ["publish","identity"],["identity","material"],
    ["video","fusion","cache"],["fusion","tracking","no-pose"],["touches","publish","no-types"]
  ];
  const graphSettings={
    video:[".analysis-profile-source","#studio-analysis-editor"],
    ball:["#studio-gpu-stride"],
    rallies:["#studio-gpu-rally-min-seconds","#studio-gpu-rally-merge-seconds","#studio-gpu-rally-stationary-ms"],
    players:["#studio-gpu-person-stride"],
    pose:["#studio-gpu-pose-enabled","#studio-gpu-interpolate-pose"],
    fusion:["#studio-gpu-memory","#studio-gpu-threads"],
    tracking:["#studio-gpu-interpolate-players","#studio-post-player-score","#studio-post-max-gap",
      "#studio-post-lost-low-gap","#studio-post-lost-low-iou","#studio-post-min-track-frames","#studio-post-nms","#studio-post-assignment","#studio-post-margin"],
    touches:["#studio-post-touch-threshold","#studio-post-pose-confidence",
      "#studio-post-pose-distance","#studio-post-occluded-turn","#studio-post-touch-gap",
      "#studio-post-reach","#studio-post-smooth","#studio-post-interval",
      "#studio-post-ball-score","#studio-post-ball-gap",
      "#studio-post-idle","#studio-post-stationary","#studio-post-rally-min",
      "#studio-post-coverage","#studio-post-samples","#studio-post-min-touches",
      "#studio-post-no-touch","#studio-post-context"],
    touch_types:["#studio-post-tcn-experimental"],
    identity:["#studio-post-auto-assign-threshold"]
  };
  const graphDescriptions={
    video:"Выбор видео, интервалов анализа и разметки площадки.",
    ball:"VballNetV4c выполняет нейросетевую детекцию мяча на GPU.",
    rallies:"Рабочие настройки первичного анализа: длительность траектории, остановка мяча в миллисекундах и объединение без перекрытия удержаний или пропусков детекции от 3 секунд. Эксперименты в «Дебаге» имеют собственные значения и не меняют этот GPU-профиль.",
    players:"RF-DETR Medium ищет игроков только внутри найденных игровых интервалов.",
    pose:"RTMPose-M Body7 получает рамки игроков и исходные пиксели видео.",
    fusion:"Сохранение общей версии GPU-наблюдений и повторное использование кэша.",
    tracking:"ByteTrack сопровождает Track ID на CPU. Это алгоритм трекинга, а не нейросеть.",
    touches:"Физические касания и определение их типов выполняются в одном CPU-проходе.",
    touch_types:"По умолчанию используется алгоритм PoseMotion. Экспериментальная нейросеть Volley3D TCN используется только при включённой опции и наличии модели; отдельного GPU-прохода здесь нет.",
    publish:"Первичные розыгрыши публикуются без повторного определения границ.",
    identity:"Автоматическая привязка Track ID к игрокам из библиотеки.",
    material:"Разметка временной шкалы для предпросмотра и монтажа.",
    cache:"Готовый GPU-анализ переиспользуется при совместимых настройках."
  };
  let graphMoved=[],graphFocus=null;
  const closeGraphSettings=()=>{
    const dialog=$("analysis-flow-dialog");
    if(dialog.classList.contains("hidden"))return;
    for(const [node,placeholder,wasOpen] of graphMoved){
      placeholder.replaceWith(node);
      if(wasOpen!==null)node.open=wasOpen;
    }
    graphMoved=[];dialog.classList.add("hidden");
    dialog.querySelector(".analysis-flow-dialog-panel").classList.remove("is-editor","is-compact","is-single","is-medium");
    if(graphFocus?.isConnected)graphFocus.focus({preventScroll:true});
  };
  const openGraphSettings=id=>{
    closeGraphSettings();
    graphFocus=document.activeElement;
    $("analysis-flow-dialog-title").textContent=pipelineNames.get(id)||"Стадия";
    $("analysis-flow-dialog-description").textContent=graphDescriptions[id]||"";
    const body=$("analysis-flow-dialog-body");clear(body);
    for(const selector of graphSettings[id]||[]){
      const control=ui.analysis.querySelector(selector);
      if(!control)continue;
      const item=control.matches("input,select")?
        control.closest(".analysis-profile-toggle")||control.parentElement:control;
      if(!item||graphMoved.some(([node])=>node===item))continue;
      const mark=document.createComment("stage-settings");
      item.before(mark);
      graphMoved.push([item,mark,item.tagName==="DETAILS"?item.open:null]);
      body.append(item);
      if(item.tagName==="DETAILS")item.open=true;
    }
    if(!body.children.length)body.append(make("p","panel-note","Дополнительных параметров нет."));
    const panel=$("analysis-flow-dialog").querySelector(".analysis-flow-dialog-panel");
    panel.classList.toggle("is-editor",id==="video");
    panel.classList.toggle("is-single",id!=="video"&&body.children.length<=1);
    panel.classList.toggle("is-compact",id!=="video"&&body.children.length>=2&&body.children.length<=4);
    panel.classList.toggle("is-medium",id!=="video"&&body.children.length>=5&&body.children.length<=10);
    $("analysis-flow-dialog").classList.remove("hidden");
    $("analysis-flow-dialog-close").focus();
  };
  $("analysis-flow-dialog-close").addEventListener("click",closeGraphSettings);
  $("analysis-flow-dialog-ok").addEventListener("click",closeGraphSettings);
  $("analysis-flow-dialog").addEventListener("click",event=>{
    if(event.target.id==="analysis-flow-dialog")closeGraphSettings();
  });
  document.addEventListener("keydown",event=>{
    if(event.key==="Escape"&&!$("analysis-flow-dialog").classList.contains("hidden"))
      closeGraphSettings();
  });
  let graphSequence=0;
  const graphBounds=(node,root)=>{
    const a=node.getBoundingClientRect(),b=root.getBoundingClientRect();
    return {l:a.left-b.left,r:a.right-b.left,t:a.top-b.top,b:a.bottom-b.top,
      cx:(a.left+a.right)/2-b.left,cy:(a.top+a.bottom)/2-b.top};
  };
  const graphDraw=target=>{
    const svg=target.querySelector(".analysis-flow-lines");
    if(!svg||!target.isConnected||!target.offsetWidth)return;
    const byId=new Map([...target.querySelectorAll(".analysis-flow-node")].map(n=>
      [n.dataset.stageId,graphBounds(n,target)]));
    svg.replaceChildren();
    svg.setAttribute("viewBox","0 0 "+target.clientWidth+" "+target.clientHeight);
    if(!target.dataset.graphArrow)target.dataset.graphArrow="flow-arrow-"+(++graphSequence);
    const markerId=target.dataset.graphArrow;
    const element=(tag,attrs)=>{
      const el=document.createElementNS("http://www.w3.org/2000/svg",tag);
      for(const [key,value] of Object.entries(attrs))el.setAttribute(key,value);
      return el;
    };
    const defs=element("defs",{}),marker=element("marker",{
      id:markerId,markerWidth:"7",markerHeight:"7",refX:"6",refY:"3.5",
      orient:"auto",markerUnits:"strokeWidth"});
    marker.append(element("path",{d:"M 0 0 L 7 3.5 L 0 7 Z",fill:"#7ba0c9"}));
    defs.append(marker);svg.append(defs);
    for(const [from,to,kind] of pipelineEdges){
      const a=byId.get(from),b=byId.get(to);if(!a||!b)continue;
      let d="",label=null;
      if(kind==="no-types"||kind==="cache"){
        const y=Math.min(a.t,b.t)-14;
        d="M "+a.cx+" "+a.t+" L "+a.cx+" "+y+" L "+b.cx+" "+y+" L "+b.cx+" "+b.t;
        label=[kind==="cache"?"Кэш GPU":"Без типов",(a.cx+b.cx)/2,y-3];
      }else if(kind==="no-pose"){
        const y=(a.b+b.t)/2;
        d="M "+(a.r-12)+" "+a.b+" L "+(a.r-12)+" "+y+" L "+b.cx+" "+y+" L "+b.cx+" "+b.t;
        label=["Без поз",(a.cx+b.cx)/2,y-3];
      }else if(Math.abs(a.cy-b.cy)<8){
        d=a.cx<b.cx?"M "+a.r+" "+a.cy+" L "+b.l+" "+b.cy:
          "M "+a.l+" "+a.cy+" L "+b.r+" "+b.cy;
      }else if(Math.abs(a.cx-b.cx)<8){
        d="M "+a.cx+" "+a.b+" L "+b.cx+" "+b.t;
      }else{
        const y=(a.b+b.t)/2;
        d="M "+a.cx+" "+a.b+" L "+a.cx+" "+y+" L "+b.cx+" "+y+" L "+b.cx+" "+b.t;
      }
      const path=element("path",{d,"class":"analysis-flow-edge"+(kind?" is-branch":""),
        "data-edge":from+"-"+to,"marker-end":"url(#"+markerId+")"});
      svg.append(path);
      const dot=element("circle",{r:"3.5","class":"analysis-flow-progress-dot",
        "data-edge":from+"-"+to,"aria-hidden":"true"});
      dot.append(element("animateMotion",{
        path:d,dur:"1.4s",repeatCount:"indefinite",calcMode:"linear"
      }));
      svg.append(dot);
      if(label){
        const t=element("text",{x:String(label[1]),y:String(label[2]),
          "class":"analysis-flow-edge-label"});
        t.textContent=label[0];svg.append(t);
      }
    }
    paintGraphEdges(target,target._graphRun);
  };
  const createPipelineGraph=(target,run)=>{
    clear(target);
    const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
    svg.setAttribute("class","analysis-flow-lines");svg.setAttribute("aria-hidden","true");
    const grid=make("div","analysis-flow-grid");
    for(const [phase,label,rows] of [
      ["gpu","1 фаза · GPU · подготовка и детекция","1"],
      ["cpu","2 фаза · CPU · обработка результатов","2"]
    ]){
      const panel=make("div","analysis-flow-phase");
      panel.dataset.phase=phase;
      panel.style.gridRow=rows;
      panel.style.gridColumn="1 / -1";
      panel.setAttribute("aria-hidden","true");
      panel.append(make("span","analysis-flow-phase-label",label));
      grid.append(panel);
    }
    for(const [id,name,kind,row,col] of pipelineStages){
      const cell=make("div","analysis-flow-cell");
      cell.style.gridRow=row;cell.style.gridColumn=row===2?7-col:col;
      const node=make("button","analysis-flow-node");
      node.type="button";node.dataset.stageId=id;node.dataset.kind=kind;
      node.dataset.flow=row===2?"rtl":"ltr";
      const heading=make("span","analysis-flow-node-title",window.VolleyballUi.translate(name));
      const technical=make("span","analysis-flow-node-model");
      const meta=pipelineModelMeta(id,run);
      if(meta){
        technical.textContent=meta[0];
        technical.dataset.modelKind=meta[1];
      }else technical.classList.add("hidden");
      const headingRow=make("span","analysis-flow-node-heading-row");
      headingRow.append(pipelineIcon(id),heading);
      node.append(headingRow,technical);
      node.setAttribute("aria-label",window.VolleyballUi.translate(name)+
        (meta?" — "+(meta[1]==="model"?"нейросеть ":"алгоритм ")+meta[0]:""));
      node.title=graphDescriptions[id];
      node.addEventListener("click",()=>{
        openGraphSettings(id);
      });
      const stats=make("small","analysis-flow-node-stats","—");
      cell.append(node,stats);
      grid.append(cell);
    }
    target.append(svg,grid);
    requestAnimationFrame(()=>graphDraw(target));
  };
  const updatePipelineGraph=(target,run)=>{
    target._graphRun=run;
    if(!target.querySelector(".analysis-flow-grid"))createPipelineGraph(target,run);
    const stages=Array.isArray(run?.stages)?run.stages:[];
    const seen=new Map();
    for(const item of stages){const id=pipelineId(item);if(id)seen.set(id,item);}
    const active=run&&["queued","running","cancelling"].includes(run.status);
    const currentId=pipelineNames.has(run?.stage_id)?run.stage_id:pipelineId(run?.stage);
    const cached=run?.gpu_reused===true;
    const complete=run?.status==="complete";
    for(const node of target.querySelectorAll(".analysis-flow-node")){
      const id=node.dataset.stageId,entry=seen.get(id==="touch_types"?"touches":id);
      const classification=run?.profile?.post?.pose_classification!==false;
      const combinedPose=id==="pose"&&currentId==="players"&&
        /RTMPose/.test(String(run?.stage||""));
      const concurrentTypes=id==="touch_types"&&currentId==="touches"&&classification;
      const running=active&&(id===currentId||combinedPose||concurrentTypes||
        !!entry&&entry.finished_ms==null&&entry===stages.at(-1));
      const done=!!entry&&(entry.finished_ms!=null||complete&&!running)&&
        (id!=="touch_types"||classification)||
        (id==="pose"&&!!run?.gpu_jobs?.pose&&!running);
      const reused=cached&&id==="fusion"&&!done;
      node.classList.toggle("is-active",!!running);
      node.classList.toggle("is-complete",done);
      node.classList.toggle("is-reused",reused);
      node.classList.toggle("is-skipped",
        cached&&["ball","rallies","players","pose"].includes(id)&&!entry&&
          !(id==="pose"&&run?.gpu_jobs?.pose)||
        id==="pose"&&run?.profile?.gpu?.pose_enabled===false||
        id==="touch_types"&&!classification);
      node.classList.toggle("is-failed",run?.status==="failed"&&id===currentId);
      const pct=stageProgressPercent(entry,!!running,done);
      node.style.setProperty("--stage-progress",(pct??0)+"%");
      node.classList.toggle("is-progress-unknown",!!running&&pct===null);
      const meta=pipelineModelMeta(id,run);
      const technical=node.querySelector(".analysis-flow-node-model");
      if(technical){
        technical.textContent=meta?.[0]||"";
        technical.classList.toggle("hidden",!meta);
        if(meta)technical.dataset.modelKind=meta[1];
      }
      const stats=node.nextElementSibling;
      if(stats?.classList.contains("analysis-flow-node-stats")){
        const statusText=stageResultText(id,entry,run,Date.now(),done,running);
        stats.textContent=running&&pct!==null?Math.round(pct)+"% · "+statusText:statusText;
        const sharedPose=id==="pose"&&!entry&&
          (run?.shared_pose?.source_stage_id==="players"||
            /RF-DETR.*RTMPose/.test(String(seen.get("players")?.name||"")));
        stats.title=stats.textContent==="—"?"Нет данных об этапе":
          sharedPose?"RTMPose и RF-DETR использовали общий GPU-проход; время указано совместное. "+stats.textContent:stats.textContent;
      }
      const name=pipelineNames.get(id)||id;
      node.title=name+(meta?" · "+(meta[1]==="model"?"Нейросеть":"Алгоритм")+": "+meta[0]:"")+
        (entry?.frames_total?" · "+(entry.frames_processed||0)+"/"+entry.frames_total+" кадров":"")+
        (running?" · выполняется":done?" · завершено":reused?" · из кэша":"");
    }
    paintGraphEdges(target,run);
  };
  const paintGraphEdges=(target,run)=>{
    const active=run&&["queued","running","cancelling"].includes(run.status);
    const currentId=pipelineNames.has(run?.stage_id)?run.stage_id:pipelineId(run?.stage);
    const cached=run?.gpu_reused===true,complete=run?.status==="complete";
    for(const edge of target.querySelectorAll(".analysis-flow-edge")){
      const [from,to]=edge.dataset.edge.split("-");
      const a=target.querySelector('[data-stage-id="'+from+'"]');
      const b=target.querySelector('[data-stage-id="'+to+'"]');
      const branchCache=from==="video"&&to==="fusion";
      const branchSkip=from==="fusion"&&to==="tracking";
      const branchTypes=from==="touches"&&to==="publish";
      const classification=run?.profile?.post?.pose_classification!==false;
      const selected=branchCache?cached:branchSkip?run?.profile?.gpu?.pose_enabled===false:
        branchTypes?!classification:!cached;
      const done=branchCache?cached&&complete:selected&&a?.classList.contains("is-complete")&&b?.classList.contains("is-complete");
      const running=selected&&active&&(branchCache?
        currentId==="cache"||(currentId==="tracking"&&cached):
        b?.classList.contains("is-active")&&
        (a?.classList.contains("is-complete")||a?.classList.contains("is-active")));
      edge.classList.toggle("is-complete",!!done);
      edge.classList.toggle("is-active",!!running);
      edge.style.opacity=selected?"1":".35";
      const dot=target.querySelector('.analysis-flow-progress-dot[data-edge="'+from+"-"+to+'"]');
      dot?.classList.toggle("is-visible",!!running);
    }
  };
  $("analysis-flow-feedback").append(ui.note,ui.profileDelta);
  const mainGraph=$("analysis-flow-diagram");
  if(mainGraph&&typeof mainGraph.querySelector==="function"){
    updatePipelineGraph(mainGraph,null);
    window.addEventListener("resize",()=>requestAnimationFrame(()=>graphDraw(mainGraph)));
  }
  window.addEventListener("volley:ui-language",()=>{
    for(const node of ui.analysis.querySelectorAll(".analysis-flow-node")){
      const name=pipelineNames.get(node.dataset.stageId);
      const heading=node.querySelector(".analysis-flow-node-title");
      if(name&&heading)heading.textContent=window.VolleyballUi.translate(name);
    }
    requestAnimationFrame(()=>graphDraw(mainGraph));
  });
  if(typeof ResizeObserver!=="undefined"){
    const observer=new ResizeObserver(()=>requestAnimationFrame(()=>graphDraw(mainGraph)));
    observer.observe(mainGraph);
  }
  for(const [id,button] of [["analysis-flow-gpu-start",ui.gpuOnly],
    ["analysis-flow-cpu-start",ui.postOnly],["analysis-flow-full-start",ui.start]]){
    $(id).addEventListener("click",()=>button.click());
  }
  if(mainGraph&&typeof mainGraph.querySelector==="function")ui.analysis.classList.add("analysis-flow-ready");
  const historyDuration=ms=>{
    if(!Number.isFinite(ms)||ms<0)return "—";
    const seconds=Math.floor(ms/1000),hours=Math.floor(seconds/3600);
    return hours?hours+" ч "+Math.floor(seconds%3600/60)+" мин":
      seconds>=60?Math.floor(seconds/60)+" мин "+seconds%60+" с":seconds+" с";
  };
  const historyLog=async(project,id,output)=>{
    let after=0,more=true,rows=[];
    try{
      while(more){
        const data=await get("/api/studio/pipeline/log",{project_id:project,id,after});
        if(projectId()!==project)return;
        if(data.schema!=="volley-studio-pipeline-log/1"||data.id!==id||
          data.project_id!==project||!Array.isArray(data.entries)||
          !Number.isSafeInteger(data.next_offset)||data.next_offset<after)
          throw Error("Некорректный ответ журнала");
        rows.push(...data.entries.map(formatLogRow));
        more=data.has_more===true&&data.next_offset>after;
        after=data.next_offset;
      }
      output.textContent=rows.length?rows.join("\n\n"):"Записей нет.";
      return true;
    }catch(error){output.textContent="Не удалось загрузить журнал: "+String(error.message||error);return false;}
  };
  const showGraphResult=source=>{
    const target=$("analysis-flow-result");
    if(!target)return;
    clear(target);
    if(!source){
      target.append(make("span","analysis-flow-result-muted",
        "После анализа здесь будет виден последний результат."));
      return;
    }
    const mode=source.mode==="gpu"?"1 фаза":source.mode==="post"?"2 фаза":"Полный анализ";
    const statuses={queued:"В очереди",running:"В процессе",cancelling:"Останавливается",
      complete:"Завершён",failed:"Ошибка",cancelled:"Отменён",interrupted:"Прерван"};
    const elapsed=source.elapsed_ms??source.duration_ms;
    const started=Number(source.started_ms);
    const timestamp=Number.isFinite(started)&&started>0?
      " · "+new Date(started).toLocaleString("ru-RU"):"";
    target.append(
      make("strong","analysis-flow-result-title","Последний запуск · "+mode),
      make("span","analysis-flow-result-status",
        (statuses[source.status]||"Ожидание данных")+
        (elapsed==null?"":" · "+historyDuration(Number(elapsed)))+timestamp)
    );
  };
  const liveHistory=()=>{
    const status=state.latestPipeline;
    return status&&status.project_id===projectId()&&status.id&&status.status!=="idle"?status:null;
  };
  const historyVisibleByProject=new Map();
  const visibleHistoryRuns=project=>{
    if(!historyVisibleByProject.has(project)){
      let stored=[];
      try{
        const raw=window.sessionStorage?.getItem("volley:analysis-session-runs:v1:"+project);
        const parsed=JSON.parse(raw||"[]");
        if(Array.isArray(parsed))stored=parsed.filter(id=>/^[a-f0-9]{32}$/.test(id)).slice(-40);
      }catch{}
      historyVisibleByProject.set(project,new Set(stored));
    }
    return historyVisibleByProject.get(project);
  };
  const recordHistoryRun=(project,id)=>{
    if(!/^[a-f0-9]{12}$/.test(project||"")||!/^[a-f0-9]{32}$/.test(id||""))return;
    const ids=visibleHistoryRuns(project);
    ids.add(id);
    if(ids.size>40)ids.delete(ids.values().next().value);
    try{window.sessionStorage?.setItem("volley:analysis-session-runs:v1:"+project,
      JSON.stringify([...ids]));}catch{}
  };
  const forgetHistoryRuns=project=>{
    historyVisibleByProject.delete(project);
    try{window.sessionStorage?.removeItem("volley:analysis-session-runs:v1:"+project);}catch{}
  };
  const liveHistoryLine=status=>{
    const pct=Number(status.progress_percent);
    return (pipelineNames.get(status.stage_id)||pipelineNames.get(pipelineId(status.stage))||String(status.stage||"Обработка"))+" · "+
      (Number.isFinite(pct)?Math.max(0,Math.min(pct,100)).toFixed(1)+"%":"—");
  };
  const historyClock=status=>{
    const moving=["queued","running","cancelling"].includes(status.status);
    const since=Math.max(0,Date.now()-(state.historyStatusAt||Date.now()));
    const base=status.elapsed_ms==null?NaN:Number(status.elapsed_ms);
    const elapsed=Number.isFinite(base)?base+(moving?since:0):null;
    const remaining=Number(status.remaining_ms);
    return {elapsed,remaining:moving&&status.remaining_ms!=null&&
      Number.isFinite(remaining)?Math.max(0,remaining-since):null};
  };
  const historyTimeLine=(row,current)=>{
    if(!current)return Number.isFinite(row.duration_ms)?
      "Всего · "+historyDuration(row.duration_ms):"";
    const clock=historyClock(current);
    if(current.status==="queued")return "Ожидает запуска";
    const prefix=current.status==="complete"?"Завершён":
      current.status==="failed"?"Ошибка":
      current.status==="cancelled"?"Отменён":
      current.status==="interrupted"?"Прерван":"Выполняется";
    return clock.elapsed==null?"Время выполнения уточняется":
      prefix+" · "+historyDuration(clock.elapsed);
  };
  const historyEtaLine=current=>{
    if(!current||!["running","queued","cancelling"].includes(current.status))return "";
    if(current.status==="queued")return "Ожидание очереди";
    const eta=historyClock(current).remaining;
    return eta==null?"Осталось: рассчитывается":"Осталось ≈ "+historyDuration(eta);
  };
  const renderStageTimeline=(target,row,current)=>{
    clear(target);
    const stages=(Array.isArray(current?.stages)&&current.stages.length)?
      current.stages:(Array.isArray(row?.stages)?row.stages:[]);
    if(!stages.length){
      target.append(make("span","analysis-history-no-stages",
        current?"Стадии появятся после запуска.":"Время стадий не записано."));
      return;
    }
    const now=Date.now();
    for(const entry of stages){
      const live=!!current&&["running","cancelling"].includes(current.status)&&
        entry.started_ms!=null&&entry.finished_ms==null;
      const started=entry.started_ms==null?NaN:Number(entry.started_ms);
      const elapsed=live&&Number.isFinite(started)?Math.max(0,now-started):
        entry.duration_ms==null?null:Number(entry.duration_ms);
      const line=make("div","analysis-history-stage"+(live?" is-running":""));
      line.append(
        make("span","analysis-history-stage-name",pipelineLabel(entry)),
        make("span","analysis-history-stage-duration",historyDuration(elapsed))
      );
      target.append(line);
    }
  };
  const renderHistory=()=>{
    // Preserve the existing streaming log/progress nodes when history is refreshed.
    $("studio-analysis-live-host").append(ui.progress,ui.logPanel);
    clear(ui.historyList);
    const latest=liveHistory();
    if(typeof updatePipelineGraph==="function"&&typeof mainGraph!=="undefined"&&
      mainGraph&&typeof mainGraph.querySelector==="function"){
      const source=latest||state.historyRows[0]||null;
      updatePipelineGraph(mainGraph,source);
      showGraphResult(source);
      requestAnimationFrame(()=>graphDraw(mainGraph));
    }
    const active=latest&&["queued","running","cancelling"].includes(latest.status);
    const visibleIds=visibleHistoryRuns(projectId());
    const rows=state.historyRows.filter(row=>visibleIds.has(row.id)&&
      (state.historyFilter==="all"||row.mode===state.historyFilter||
        !row.mode||row.mode==="full"||(active&&row.id===latest.id)));
    // The status endpoint can observe a new job before the history endpoint lists it.
    if(active&&visibleIds.has(latest.id)&&!rows.some(row=>row.id===latest.id)){
      rows.unshift({id:latest.id,mode:latest.mode,video:latest.video,
        status:latest.status,stage:latest.stage,stages:[],started_ms:null});
    }
    if(!rows.length){
      ui.historyList.textContent=visibleIds.size?
        "Запуски текущей сессии не найдены в выбранном фильтре.":
        "История заполнится после нового запуска анализа.";
      return;
    }
    const labels={gpu:"1 фаза",post:"2 фаза",full:"Полный анализ"};
    const statuses={queued:"В очереди",running:"Выполняется",cancelling:"Останавливается",
      complete:"Завершён",failed:"Ошибка",cancelled:"Отменён",interrupted:"Прерван"};
    for(const row of rows){
      const current=latest&&latest.id===row.id?latest:null;
      const currentStatus=current?.status||row.status;
      const details=make("details","analysis-history-run"),summary=make("summary");
      details.dataset.runId=row.id;
      summary.append(make("strong","",(labels[row.mode]||labels.full)+" · "+String(row.video||"—")));
      const date=Number.isFinite(row.started_ms)?new Date(row.started_ms).toLocaleString():"Дата недоступна";
      summary.append(make("span","analysis-history-date",date),
        make("span","analysis-history-status",statuses[currentStatus]||String(currentStatus||"—")),
        make("span","analysis-history-runtime",historyTimeLine(row,current)));
      if(row.result_reused===true)summary.append(make("span","","Готовый результат"));
      else if(row.gpu_reused===true)summary.append(make("span","","GPU использован повторно"));
      if(current)summary.append(make("span","analysis-history-live-stage",liveHistoryLine(current)));
      if(current)summary.append(make("span","analysis-history-eta",historyEtaLine(current)));
      details.append(summary);
      const body=make("div","analysis-history-detail"),stages=make("div","analysis-history-stages");
      renderStageTimeline(stages,row,current);
      body.append(stages);
      if(current){
        body.append(ui.progress,ui.logPanel);
        if(["queued","running","cancelling"].includes(current.status))details.open=true;
      }else{
        const log=make("details"),title=make("summary","","Журнал выполнения"),
          output=make("pre","analysis-run-log-output","Загрузка…");
        log.append(title,output);body.append(log);
        let loaded=false;
        const logProject=projectId();
        log.addEventListener("toggle",()=>{
          if(log.open&&!loaded){loaded=true;
            historyLog(logProject,row.id,output).then(ok=>{if(!ok)loaded=false;});
          }
        });
      }
      if(row.error)body.append(make("p","",String(row.error)));
      details.append(body);ui.historyList.append(details);
    }
  };
  const updateLiveHistory=status=>{
    if(status.project_id!==projectId()||!status.id||status.status==="idle")return;
    state.latestPipeline=status;
    if(typeof updatePipelineGraph==="function"&&typeof mainGraph!=="undefined"&&
      mainGraph&&typeof mainGraph.querySelector==="function"){
      updatePipelineGraph(mainGraph,status);
      showGraphResult(status);
      requestAnimationFrame(()=>graphDraw(mainGraph));
    }
    if(!visibleHistoryRuns(status.project_id).has(status.id))return;
    let item=[...ui.historyList.children].find(node=>node.dataset?.runId===status.id);
    if(!item){renderHistory();item=[...ui.historyList.children].find(node=>node.dataset?.runId===status.id);}
    if(!item)return;
    const statuses={queued:"В очереди",running:"Выполняется",cancelling:"Останавливается",
      complete:"Завершён",failed:"Ошибка",cancelled:"Отменён",interrupted:"Прерван"};
    const label=item.querySelector(".analysis-history-status");
    if(label)label.textContent=statuses[status.status]||String(status.status);
    const stage=item.querySelector(".analysis-history-live-stage");
    if(stage)stage.textContent=liveHistoryLine(status);
    const runtime=item.querySelector(".analysis-history-runtime");
    if(runtime)runtime.textContent=historyTimeLine(null,status);
    const eta=item.querySelector(".analysis-history-eta");
    if(eta)eta.textContent=historyEtaLine(status);
    const stages=item.querySelector(".analysis-history-stages");
    if(stages)renderStageTimeline(stages,null,status);
    if(["queued","running","cancelling"].includes(status.status))item.open=true;
  };
  const loadHistory=async(append=false)=>{
    const project=projectId(),epoch=++state.historyEpoch,offset=append?state.historyOffset:0;
    if(!append)state.historyOffset=0;
    ui.historyMore.classList.add("hidden");
    if(!project){ui.historyList.textContent="Выберите проект.";return;}
    // Keep the visible live run and its log mounted while refreshing history.
    if(!append&&!ui.historyList.children.length&&!liveHistory())
      ui.historyList.textContent="Загрузка истории…";
    try{
      const data=await get("/api/studio/pipeline/history",{project_id:project,offset});
      if(epoch!==state.historyEpoch||project!==projectId())return;
      if(data.schema!=="volley-studio-pipeline-history/1"||data.project_id!==project||!Array.isArray(data.runs)||
        !Number.isSafeInteger(data.next_offset)||data.next_offset<offset)
        throw Error("Некорректный ответ истории");
      state.historyRows=append?state.historyRows.concat(data.runs):data.runs;
      state.historyOffset=data.next_offset;
      ui.historyMore.classList.add("hidden");
      renderHistory();
    }catch(error){if(epoch===state.historyEpoch){
      if(append)ui.historyMore.classList.remove("hidden");
      else if(liveHistory())renderHistory();
      else ui.historyList.textContent="Не удалось загрузить историю: "+String(error.message||error);
    }}
  };
  ui.historyRefresh.addEventListener("click",()=>loadHistory());
  ui.historyClear.addEventListener("click",async()=>{
    const project=projectId();
    if(!project||ui.historyClear.disabled)return;
    ui.historyClear.disabled=true;
    state.historyEpoch++;
    try{
      await post("/api/studio/pipeline/history",{project_id:project});
      forgetHistoryRuns(project);
      if(project===projectId())await loadHistory();
    }catch(error){if(project===projectId())ui.historyList.textContent="Не удалось очистить историю: "+String(error.message||error);}
    finally{ui.historyClear.disabled=false;}
  });
  ui.historyMore.addEventListener("click",()=>loadHistory(true));
  // Only update the existing current row: do not rebuild it or remount the log.
  setInterval(()=>{
    const current=liveHistory();
    if(current&&["queued","running","cancelling"].includes(current.status))
      updateLiveHistory(current);
  },1000);
  for(const button of document.querySelectorAll("[data-history-filter]")){
    button.addEventListener("click",()=>{
      state.historyFilter=button.dataset.historyFilter;
      for(const tab of document.querySelectorAll("[data-history-filter]"))
        tab.setAttribute("aria-selected",String(tab===button));
      renderHistory();
    });
  }
  const displayPipeline=(status,notifyCompletion=false)=>{
    if(!status||status.schema!=="volley-studio-pipeline/1")return;
    const percent=Number(status.progress_percent);
    ui.progress.value=Number.isFinite(percent)?Math.max(0,Math.min(percent,100)):0;
    if(typeof updatePipelineGraph==="function"&&typeof mainGraph!=="undefined"&&
      mainGraph&&typeof mainGraph.querySelector==="function"){
      updatePipelineGraph(mainGraph,status);
      showGraphResult(status);
      requestAnimationFrame(()=>graphDraw(mainGraph));
    }
    const active=["queued","running","cancelling"].includes(status.status);
    if(status.status==="cancelling")state.stopRequested=true;
    const stage=String(status.stage||"Обработка");
    const project=String(status.project_id||"");
    const mode=["full","gpu","post"].includes(status.mode)?status.mode:"full";
    const runLabel=mode==="gpu"?"1 фаза · GPU":
      mode==="post"?"2 фаза · CPU":"Полный анализ · GPU/CPU";
    if(active){
      window.VolleyballTasks?.set("analysis",{label:runLabel,
        projectId:project,stage,progress:percent});
      saveMessage(status.status==="cancelling"?
        "Отмена запрошена · ожидаем остановки текущего GPU/CPU-вызова":
        stage+" · "+(Number.isFinite(percent)?percent.toFixed(1)+"%":"—"));
    }else{
      window.VolleyballTasks?.clear("analysis");
      saveMessage(status.status==="complete"?
        (mode==="gpu"?"Новый GPU-анализ сохранён. Версия доступна в шапке.":
         mode==="post"?"Создан новый вариант CPU-постобработки для выбранного GPU.":
         status.result_reused===true?
          "Настройки совпали: выбраны готовые версии GPU и CPU.":
          status.gpu_reused===true?
          "GPU использован повторно. Новая 2 фаза доступна в шапке.":
          "GPU и CPU завершены. Новые версии доступны в шапке страницы."):
        status.status==="cancelled"?"Анализ отменён. Частичные версии не выбраны.":
        status.status==="failed"||status.status==="interrupted"?
          "Ошибка анализа: "+String(status.error||stage):
          "Готово к новому запуску.",
        status.status==="failed"||status.status==="interrupted");
    }
    if(notifyCompletion&&status.status==="complete"&&status.id!==state.completedId&&
        status.project_id===projectId()){
      state.completedId=status.id;
      window.dispatchEvent(new CustomEvent("volley:pipeline-complete",{
        detail:{project_id:status.project_id,video:status.video,
          analysis_id:status.analysis_id,postprocessing_id:status.postprocessing_id}
      }));
    }
    state.historyStatusAt=Date.now();
    updateLiveHistory(status);
    if(!active&&status.id===state.pipelineId){
      stopPipelinePolling();
      state.pipelineId="";state.pipelineProject="";state.stopRequested=false;
      loadHistory();
      if(status.status==="failed"||status.status==="interrupted")
        announce(runLabel+": "+String(status.error||stage),true);
    }
    updateProfile();
    loadLog(status).catch(failure);
  };
  const pollPipeline=async()=>{
    if(!state.pipelineId||state.polling)return;
    state.polling=true;
    const id=state.pipelineId,project=state.pipelineProject;
    try{
      const status=await get("/api/studio/pipeline/status",{project_id:project,id});
      if(state.pipelineId===id){
        state.pollErrors=0;
        displayPipeline(status,true);
      }
    }catch(e){
      if(state.pipelineId===id&&++state.pollErrors>=3){
        window.VolleyballTasks?.clear("analysis");
        saveMessage("Не удалось получить состояние анализа: "+String(e.message||e),true);
        stopPipelinePolling();
      }
    }finally{state.polling=false;}
  };
  const restorePipeline=async()=>{
    const project=projectId();
    if(!project||state.pipelineId)return;
    const epoch=state.analysisGeneration;
    const status=await get("/api/studio/pipeline/status",{project_id:project});
    if(project!==projectId()||epoch!==state.analysisGeneration)return;
    if(["queued","running","cancelling"].includes(status.status)){
      recordHistoryRun(project,status.id);
      state.pipelineId=status.id;
      state.pipelineProject=project;
      state.stopRequested=status.status==="cancelling";
      ui.logPanel.open=true;
      saveMessage("Подключение к уже запущенному анализу…");
      stopPipelinePolling();
      await pollPipeline();
      if(state.pipelineId)state.pipelineTimer=setInterval(pollPipeline,1800);
    }else if(status.status!=="idle"){
      // Restoring an old completed job must not invalidate GPU/CPU snapshots.
      if(status.status==="complete")state.completedId=status.id;
      displayPipeline(status);
    }
    updateProfile();
  };
  const loadVideos=async()=>{
    const project=projectId(),old=ui.video.value;
    const token=runtime.beginLoad("analysis");
    state.analysisGeneration++;
    loadHistory();
    if(!project){
      resetLog();
      clear(ui.video);ui.video.append(make("option","","Сначала выберите проект"));
      window.VolleyballEditor?.load("","");
      updateProfile();
      return;
    }
    const catalog=window.VolleyballStudioProjectCatalog?
      await window.VolleyballStudioProjectCatalog():await api("/api/projects");
    const rows=Array.isArray(catalog.projects)?catalog.projects:[];
    if(state.tab!=="analysis"||project!==projectId()||
        !runtime.currentLoad("analysis",token))return;
    const current=rows.find(p=>p.id===project);
    const videos=Array.isArray(current?.videos)?
      current.videos.filter(name=>typeof name==="string"):[];
    clear(ui.video);
    if(!videos.length){
      const option=make("option","","Нет видео проекта");option.value="";ui.video.append(option);
    }
    for(const name of videos){
      // Linked originals have opaque internal IDs; show the actual filename.
      const recorded=current?.source_paths?.[name];
      const label=name.startsWith("linked_")&&typeof recorded==="string"?
        recorded.replace(/\\/g,"/").split("/").pop()||name:name;
      const item=make("option","",label);item.value=name;ui.video.append(item);
    }
    const savedVideo=restoreAnalysisSettings(project);
    const selected=$("studio-inference")?.value||"";
    let chainVideo="";
    try{
      const scope=JSON.parse(selected);
      if(Array.isArray(scope)&&scope[1]==="fusion"&&videos.includes(scope[0]))
        chainVideo=scope[0];
    }catch{}
    ui.video.value=chainVideo||(
      state.formProject===project&&videos.includes(old)?old:
      videos.includes(savedVideo)?savedVideo:
      videos[0]||"");
    state.formProject=project;
    window.VolleyballEditor?.load(project,ui.video.value,true);
    updateProfile();
    await restorePipeline();
  };
  const launchPipeline=async(mode="full")=>{
    if(state.pipelineId||state.starting)return;
    const project=projectId(),video=ui.video.value;
    if(!project||!video){saveMessage("Выберите проект и исходное видео.",true);return;}
    const source=mode==="post"?selectedGpuSource():"";
    if(mode!=="post"&&window.VolleyballEditor?.hasUnsavedChanges?.(project,video)){
      saveMessage("Сохраните начало и конец отрезка перед запуском анализа.",true);
      return;
    }
    if(mode!=="post"&&window.VolleyballEditor?.isLoading?.(project,video)){
      saveMessage("Настройки отрезка этого видео ещё загружаются.",true);
      return;
    }
    if(mode==="post"&&!source){
      saveMessage("Для постобработки выберите объединённый GPU-анализ в шапке.",true);
      return;
    }
    state.starting=true;updateProfile();
    try{
      const payload={project_id:project,video,mode,client_request_id:requestId()};
      if(mode!=="post")payload.gpu=gpuProfile();
      if(mode!=="gpu")payload.post=postProfile();
      if(mode==="post")payload.source_analysis_id=source;
      const response=await startWithRecovery(payload);
      if(response.schema!=="volley-studio-pipeline/1"||
          !/^[a-f0-9]{32}$/.test(String(response.id||"")))
        throw Error("Сервер не подтвердил создание анализа");
      recordHistoryRun(project,response.id);
      state.pipelineId=response.id;state.pipelineProject=project;
      resetLog();ui.logPanel.open=true;
      state.latestPipeline=response;
      renderHistory();
      loadHistory();
      ui.progress.value=0;state.stopRequested=false;
      saveMessage(mode==="gpu"?"Запущен отдельный анализ: VballNet → "+(window.VolleyballPersonDetector?.label||"детектор игроков + RTMPose")+" → объединение.":
        mode==="post"?"Запущена 2 фаза для выбранного результата 1 фазы.":
        "Запущен полный анализ: VBallNet → "+(window.VolleyballPersonDetector?.label||"детектор игроков + RTMPose")+" → CPU.");
      stopPipelinePolling();
      await pollPipeline();
      if(state.pipelineId)state.pipelineTimer=setInterval(pollPipeline,1800);
    }catch(e){saveMessage(String(e.message||e),true);failure(e);}
    finally{state.starting=false;updateProfile();}
  };
  const cancelPipeline=async()=>{
    if(!state.pipelineId)return;
    if(state.stopRequested)return;
    state.stopRequested=true;updateProfile();
    try{
      await post("/api/studio/pipeline/cancel",{
        project_id:state.pipelineProject,id:state.pipelineId
      });
      saveMessage("Запрошена отмена. Ожидаем остановки текущего GPU/CPU-вызова.");
    }catch(e){state.stopRequested=false;updateProfile();saveMessage(String(e.message||e),true);}
  };
  const resetProfile=()=>{
    for(const input of allProfileFields())applyFieldValue(input,input.dataset.default);
    saveAnalysisSettings();
    updateProfile();
  };
  const renderSidebarPlayers=()=>{
    const sidebar=$("studio-players-sidebar-list");
    if(!sidebar)return;
    clear(sidebar);
    const rows=[...state.players].filter(p=>
      typeof p.id==="string"&&typeof p.name==="string"&&state.projectMemberIds.has(p.id)
    ).sort((a,b)=>String(a.name).localeCompare(String(b.name),"ru"));
    if(!rows.length){
      sidebar.append(make("div","players-sidebar-empty","В проект пока не добавлены игроки."));
      return;
    }
    for(const player of rows){
      const card=make("article","players-side-card");
      const avatar=make("div","players-side-avatar");
      const count=Number.isSafeInteger(player.saved_photo_count)&&
        player.saved_photo_count>=0?player.saved_photo_count:0;
      const refId=player.cover_reference_id;
      const cover=state.trackCovers.get(player.id);
      const saved=typeof refId==="string"&&/^[a-f0-9]{24}$/.test(refId);
      const url=saved?"/api/players/library/photo?"+new URLSearchParams({
        player_id:player.id,reference_id:refId
      }):cover?"/api/players/native/track-photo?"+new URLSearchParams(cover):null;
      if(url){
        const img=make("img");img.loading="lazy";img.decoding="async";
        img.alt=(saved?"Сохранённое фото ":"Кадр привязанного Track ID ")+player.name;
        img.title=saved?"Сохранённый фотоэталон":
          "Кадр подтверждённого Track ID, ещё не сохранённый фотоэталон";
        img.src=url;
        img.addEventListener("error",()=>{
          img.remove();avatar.textContent=player.name.slice(0,1).toUpperCase();
        });
        avatar.append(img);
      }else avatar.textContent=player.name.slice(0,1).toUpperCase();
      const meta=make("div","players-side-meta");
      meta.append(make("strong","",player.name),
        make("small","",count+" сохранённых фото"+(state.playerTrackCounts?
          " · "+(state.playerTrackCounts[player.id]||0)+" привязанных ID":"")),
        make("span","players-side-membership"+
          (state.projectMemberIds.has(player.id)?" is-linked":""),
          state.projectMemberIds.has(player.id)?"В проекте":"В библиотеке"));
      const menu=make("details","players-side-menu");
      const selectPlayer=make("button","players-side-select");
      selectPlayer.type="button";
      selectPlayer.ariaLabel="Показать привязанные Track ID игрока "+player.name;
      selectPlayer.addEventListener("click",()=>window.dispatchEvent(
        new CustomEvent("volley:player-gallery-select",{
          detail:{project_id:projectId(),player_id:player.id,name:player.name}
        })));
      const toggle=make("summary","","⋮");
      toggle.ariaLabel="Действия с фотографиями "+player.name;
      const rename=make("button","","Переименовать");rename.type="button";
      rename.addEventListener("click",async()=>{
        menu.open=false;
        const name=window.prompt("Новое имя игрока",player.name);
        if(name===null||!name.trim()||state.mutating)return;
        state.mutating=true;
        try{
          const result=await post("/api/players/library/native",{
            action:"rename",player_id:player.id,name:name.trim(),expected_revision:state.revision
          });
          state.revision=String(result.revision);
          state.players=Array.isArray(result.players)?result.players:[];
          renderPlayers();window.dispatchEvent(new Event("volley:player-library-updated"));
        }catch(error){failure(error);await loadPlayers().catch(failure);}
        finally{state.mutating=false;}
      });
      const remove=make("button","","Удалить из проекта");remove.type="button";
      remove.addEventListener("click",()=>{
        menu.open=false;
        if(window.confirm("Удалить игрока «"+player.name+"» из проекта?"))
          window.dispatchEvent(new CustomEvent("volley:remove-project-player",{
            detail:{project_id:projectId(),player_id:player.id}
          }));
      });
      menu.append(toggle,rename,remove);
      card.append(avatar,meta,menu,selectPlayer);
      sidebar.append(card);
    }
  };
  const renderPlayers=()=>{
    clear(ui.playerList);
    renderSidebarPlayers();
    ui.playerCount.textContent=String(state.players.length)+" записей";
    if(!state.players.length){
      ui.playerList.append(make("div","empty-state","В библиотеке пока нет игроков."));
      return;
    }
    const rows=[...state.players].sort((a,b)=>
      String(a.name||"").localeCompare(String(b.name||""),"ru"));
    for(const player of rows){
      if(typeof player.id!=="string"||typeof player.name!=="string")continue;
      const row=make("article","player-row");
      row.append(make("div","player-avatar",player.name.slice(0,1).toUpperCase()));
      const meta=make("div","player-meta");
      meta.append(make("strong","",player.name),
        make("small","","ID "+player.id+" · "+
          (Number.isSafeInteger(player.saved_photo_count)?
            player.saved_photo_count:0)+" сохранённых фото · "+
          (state.projectMemberIds.has(player.id)?"в проекте":"не в проекте")));
      const edit=make("button","secondary","Переименовать");edit.type="button";
      edit.addEventListener("click",()=>{
        edit.disabled=true;
        const form=make("form","player-edit");
        const input=make("input");input.type="text";
        input.maxLength=80;input.required=true;input.value=player.name;
        input.setAttribute("aria-label","Новое имя игрока");
        const save=make("button","primary","Сохранить");save.type="submit";
        const cancel=make("button","secondary","Отмена");cancel.type="button";
        cancel.addEventListener("click",()=>form.remove());
        form.addEventListener("submit",async event=>{
          event.preventDefault();
          if(state.mutating)return;
          state.mutating=true;save.disabled=true;
          try{
            const result=await post("/api/players/library/native",{
              action:"rename",player_id:player.id,
              name:input.value,expected_revision:state.revision
            });
            state.revision=String(result.revision);
            state.players=Array.isArray(result.players)?result.players:[];
            renderPlayers();announce("Имя игрока сохранено.");
            window.dispatchEvent(new Event("volley:player-library-updated"));
          }catch(e){
            failure(e);
            await loadPlayers().catch(failure);
          }finally{state.mutating=false;}
        });
        form.append(input,save,cancel);row.append(form);input.focus();
      });
      const remove=make("button","secondary player-delete","Удалить");
      remove.type="button";
      remove.title="Исключить из активной библиотеки; исторические фото и привязки сохранятся";
      remove.addEventListener("click",async()=>{
        if(state.mutating)return;
        if(!window.confirm("Удалить игрока «"+player.name+
          "» из активной библиотеки? Исторические фото и привязки сохранятся."))return;
        state.mutating=true;remove.disabled=true;edit.disabled=true;
        try{
          const result=await post("/api/players/library/native",{
            action:"delete",player_id:player.id,
            expected_revision:state.revision
          });
          state.revision=String(result.revision);
          state.players=Array.isArray(result.players)?result.players:[];
          renderPlayers();announce("Игрок исключён из активной библиотеки.");
          window.dispatchEvent(new Event("volley:player-library-updated"));
        }catch(e){
          failure(e);
          await loadPlayers().catch(failure);
        }finally{state.mutating=false;}
      });
      const actions=make("div","player-library-actions");
      actions.append(edit,remove);
      row.append(meta,actions);
      ui.playerList.append(row);
    }
  };
  const loadPlayers=async()=>{
    const token=++state.playerGeneration;
    const wasmToken=runtime.beginLoad("players");
    ui.playerList.replaceChildren(make("div","empty-state","Загрузка библиотеки игроков…"));
    const result=await post("/api/players/library/native",{action:"read"});
    if(token!==state.playerGeneration||!runtime.currentLoad("players",wasmToken))return;
    state.players=Array.isArray(result.players)?result.players:[];
    state.revision=String(result.revision||"absent");
    state.playersLoaded=true;
    renderPlayers();
  };
  ui.playerForm.addEventListener("submit",async event=>{
    event.preventDefault();
    if(state.mutating)return;
    const name=ui.playerName.value.trim();
    if(!name)return;
    state.mutating=true;
    try{
      const result=await post("/api/players/library/native",{
        action:"create",name,expected_revision:state.revision
      });
      state.players=Array.isArray(result.players)?result.players:[];
      state.revision=String(result.revision);
      ui.playerName.value="";
      renderPlayers();announce("Игрок добавлен в библиотеку студии.");
      window.dispatchEvent(new Event("volley:player-library-updated"));
    }catch(e){
      failure(e);
      await loadPlayers().catch(failure);
    }finally{state.mutating=false;}
  });
  ui.playerRefresh.addEventListener("click",()=>loadPlayers().catch(failure));
  for(const field of allProfileFields())field.addEventListener(field.tagName==="SELECT"||field.type==="checkbox"?"change":"input",()=>{
    updateProfile();saveAnalysisSettings();
    if(mainGraph?.isConnected)updatePipelineGraph(mainGraph,state.latestPipeline||null);
  });
  ui.reset.addEventListener("click",event=>{event.preventDefault();resetProfile();});
  ui.start.addEventListener("click",()=>state.pipelineId?cancelPipeline():launchPipeline());
  ui.gpuOnly.addEventListener("click",()=>launchPipeline("gpu"));
  ui.postOnly.addEventListener("click",()=>launchPipeline("post"));
  ui.video.addEventListener("change",()=>{
    saveAnalysisSettings();
    window.VolleyballEditor?.load(projectId(),ui.video.value);
    updateProfile();
  });
  updateProfile();
  window.addEventListener("volley:chain-change",event=>{
    const detail=event.detail||{};
    if(detail.project_id!==projectId()||state.tab!=="analysis")return;
    // Clearing the last GPU analysis must immediately disable CPU postprocessing.
    if([...ui.video.options].some(option=>option.value===detail.video)&&
        ui.video.value!==detail.video){
      ui.video.value=detail.video;
      window.VolleyballEditor?.load(projectId(),ui.video.value);
    }
    updateProfile();
  });
  ui.project.addEventListener("change",()=>{
    $("studio-analysis-live-host").append(ui.progress,ui.logPanel);
    state.latestPipeline=null;
    if(mainGraph&&typeof mainGraph.querySelector==="function")updatePipelineGraph(mainGraph,null);
    resetLog();
    state.historyEpoch++;state.historyRows=[];state.historyOffset=0;ui.historyMore.classList.add("hidden");ui.historyList.textContent="Выберите проект.";
    state.analysisGeneration++;
    // project-ready is the canonical data refresh event (no double fetch).
  });
  window.addEventListener("volley:track-photo-covers",event=>{
    const detail=event.detail||{};
    if(detail.project_id!==projectId())return;
    const scope=[detail.project_id,detail.video,detail.replay_id].join("/");
    state.playerTrackCounts=detail.player_track_counts&&typeof detail.player_track_counts==="object"?
      detail.player_track_counts:null;
    if(state.trackCoverScope!==scope){
      state.trackCoverScope=scope;
      state.trackCovers.clear();
    }
    for(const row of Array.isArray(detail.tracks)?detail.tracks:[]){
      if(typeof row.player_id==="string"&&Number.isSafeInteger(row.track_id)&&
        row.track_id>0&&!state.trackCovers.has(row.player_id)){
        state.trackCovers.set(row.player_id,{
          project_id:detail.project_id,video:detail.video,
          replay_id:detail.replay_id,track_id:String(row.track_id)
        });
      }
    }
    renderSidebarPlayers();
  });
  window.addEventListener("volley:roster-updated",event=>{
    const detail=event.detail||{};
    if(detail.project_id!==projectId())return;
    state.projectMemberIds=new Set(Array.isArray(detail.player_ids)?
      detail.player_ids:[]);
    renderSidebarPlayers();
    if(state.playersLoaded)renderPlayers();
  });
  window.addEventListener("volley:player-photo-saved",()=>{
    if(state.tab==="players")loadPlayers().catch(failure);
  });
  window.addEventListener("volley:project-ready",()=>{
    state.playersLoaded=false;
    state.projectMemberIds=new Set();
    state.trackCovers.clear();state.trackCoverScope="";state.playerTrackCounts=null;
    resetLog();
    state.analysisGeneration++;
    if(state.tab==="analysis")loadVideos().catch(failure);
  });
  for(const nav of document.querySelectorAll("[data-studio-tab]")){
    nav.addEventListener("click",event=>{
      event.preventDefault();
      const requested=nav.dataset.studioTab;
      runtime.ready.then(()=>{
        switchTab(requested);
        history.replaceState(null,"","#"+runtime.tabs[runtime.get("tab")]);
      }).catch(failure);
    });
  }
  window.addEventListener("hashchange",()=>{
    runtime.ready.then(()=>switchTab(location.hash.slice(1))).catch(failure);
  });
  runtime.ready.then(()=>switchTab(location.hash.slice(1))).catch(failure);
})();
