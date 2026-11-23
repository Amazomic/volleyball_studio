"use strict";
// Source-scoped project statistics UI. Server owns all counts and source
// verification. Browser never derives identity, Touch or technique labels.
(() => {
  const get=id=>document.getElementById(id);
  const ui={project:get("studio-project"),status:get("studio-stats-status"),
    summary:get("studio-stats-summary"),total:get("studio-stats-total"),
    rallies:get("studio-stats-rallies"),attacks:get("studio-stats-attacks"),
    blocks:get("studio-stats-blocks"),serves:get("studio-stats-serves"),players:get("studio-stats-players"),
    legend:get("studio-activity-legend"),recalculate:get("studio-stats-recalculate")};
  const state={token:0,project:"",selections:new Map(),savedRevision:"absent",photos:new Map()};
  const el=(tag,klass,value)=>{
    const n=document.createElement(tag);if(klass)n.className=klass;
    if(value!==undefined)n.textContent=String(value);
    return n;
  };
  const clear=node=>node.replaceChildren();
  const note=(message,error=false)=>{
    ui.status.textContent=String(message);
    ui.status.classList.toggle("track-error",error);
  };
  const hash=value=>typeof value==="string"&&/^[a-f0-9]{64}$/.test(value);
  const finiteCount=value=>Number.isSafeInteger(value)&&value>=0;
  const selection=()=>[...state.selections.entries()].sort(([a],[b])=>a.localeCompare(b))
    .map(([,rally_run_id])=>({rally_run_id}));
  const invalidate=()=>{
    clear(ui.players);ui.summary.classList.add("hidden");
    ui.recalculate.disabled=true;state.savedRevision="absent";
  };
  const query=async url=>{
    const response=await window.VolleyballStudioActivity.fetch(url,{credentials:"same-origin",cache:"no-store"});
    const data=await response.json().catch(()=>({}));
    if(!response.ok)throw Error(data.message||data.error||"HTTP "+response.status);
    return data;
  };
  const post=async body=>{
    const response=await window.VolleyballStudioActivity.fetch("/api/players/native/statistics",{
      method:"POST",credentials:"same-origin",cache:"no-store",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(body)
    });
    const data=await response.json().catch(()=>({}));
    if(!response.ok)throw Error(data.message||data.error||"HTTP "+response.status);
    return data;
  };
  const loadPhotos=async()=>{
    const response=await window.VolleyballStudioActivity.fetch("/api/players/library/native",{
      method:"POST",credentials:"same-origin",cache:"no-store",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({action:"read"})
    });
    if(!response.ok)throw Error("Не удалось загрузить фотографии игроков");
    const library=await response.json();
    return new Map((Array.isArray(library.players)?library.players:[]).map(player=>
      [player.id,{cover:player.cover_reference_id,count:player.saved_photo_count}]));
  };
  const activityFigures=[
    {level:"very-low",range:"< 0,5×",head:[7,6],path:"M7 9 L8 14 L14 14 M8 10 L4 13 M9 10 L12 12 M14 14 L17 19 M8 14 L6 19"},
    {level:"low",range:"0,5–<0,9×",head:[10,5],path:"M10 8 L11 14 M10 9 L7 12 M11 9 L14 12 M11 14 L8 20 M11 14 L15 19"},
    {level:"medium",range:"0,9–1,1×",head:[12,4],path:"M12 7 L12 15 M12 9 L7 12 M12 9 L17 12 M12 15 L9 21 M12 15 L15 21"},
    {level:"high",range:">1,1–1,5×",head:[15,4],path:"M14 7 L11 13 M13 8 L18 10 M13 8 L9 7 M11 13 L7 18 M11 13 L16 15 L18 20"},
    {level:"very-high",range:">1,5×",head:[12,4],path:"M12 7 L12 14 M12 9 L6 5 M12 9 L18 5 M12 14 L7 20 M12 14 L17 20"}
  ];
  const activityFigure=figure=>{
    const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
    svg.setAttribute("class","report-activity-"+figure.level);
    svg.setAttribute("viewBox","0 0 24 24");svg.setAttribute("aria-hidden","true");
    const head=document.createElementNS("http://www.w3.org/2000/svg","circle");
    head.setAttribute("cx",figure.head[0]);head.setAttribute("cy",figure.head[1]);head.setAttribute("r","2");
    const body=document.createElementNS("http://www.w3.org/2000/svg","path");body.setAttribute("d",figure.path);
    svg.append(head,body);return svg;
  };
  const renderActivityLegend=()=>{
    ui.legend.replaceChildren(el("span","report-legend-label","Активность относительно среднего"));
    for(const figure of activityFigures){
      const item=el("span","report-legend-item");
      item.append(activityFigure(figure),el("span","",figure.range));ui.legend.append(item);
    }
  };
  renderActivityLegend();
  const playerCard=(player,classification)=>{
    const card=el("article","report-player-card");
    const header=el("div","report-player-header");
    const photo=state.photos.get(player.player_id)||{};
    const multiple=Number.isSafeInteger(photo.count)&&photo.count>1;
    const avatar=el(multiple?"button":"div","report-player-avatar");
    let photoLoadFailed=false;
    let current=typeof photo.cover==="string"&&/^[a-f0-9]{24}$/.test(photo.cover)?photo.cover:null;
    const showPhoto=()=>{
      if(!current){avatar.textContent=player.name.slice(0,1).toUpperCase();return;}
      const shown=current;
      const img=el("img");img.alt="";img.loading="lazy";
      img.src="/api/players/library/photo?"+new URLSearchParams({player_id:player.player_id,reference_id:current});
      img.addEventListener("error",()=>{if(current===shown){avatar.replaceChildren();avatar.textContent=player.name.slice(0,1).toUpperCase();}});
      avatar.replaceChildren(img);
    };
    showPhoto();
    if(multiple){
      avatar.type="button";
      avatar.title="Следующее фото игрока";
      avatar.setAttribute("aria-label","Следующее фото игрока "+player.name);
      avatar.addEventListener("click",async()=>{
        const project=state.project,token=state.token;
        avatar.disabled=true;
        try{
          const response=await window.VolleyballStudioActivity.fetch("/api/players/native/references",{
            method:"POST",credentials:"same-origin",cache:"no-store",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({project_id:project,player_id:player.player_id,action:"read"})
          });
          if(!response.ok)throw Error("Не удалось загрузить фотоальбом");
          const album=await response.json();
          if(token!==state.token||project!==state.project)return;
          const ids=[...new Set((Array.isArray(album.photos)?album.photos:[])
            .map(item=>item.id).filter(id=>typeof id==="string"&&/^[a-f0-9]{24}$/.test(id)))];
          if(ids.length){current=ids[(ids.indexOf(current)+1)%ids.length];showPhoto();}
          if(photoLoadFailed){note("Фотография игрока обновлена.");photoLoadFailed=false;}
        }catch(error){
          if(token===state.token&&project===state.project){photoLoadFailed=true;note("Не удалось загрузить фотографии игрока. Попробуйте ещё раз.",true);}
        }finally{avatar.disabled=false;}
      });
    }
    const identity=el("div","report-player-identity");
    identity.append(el("h4","report-player-name",player.name));
    const touches=el("div","report-player-touch-count");
    touches.append(el("strong","",player.touches),el("span","","Касания"));
    header.append(avatar,identity,touches);
    const metrics=el("div","report-player-metrics");
    const actions=player.estimated_actions||{};
    for(const [key,label] of [["attack","Атаки"],["serve","≈ Подачи"],["block","Блоки"],
      ["dig","Приёмы"],["overhead_pass","Передачи"]]){
      const metric=el("div","report-player-metric report-action-"+key);
      metric.append(el("span","",label),el("strong","",
        key==="serve"?player.initial_serve_candidates:(classification?(actions[key]??0):"—")));
      metrics.append(metric);
    }
    const footer=el("div","report-player-footer");
    footer.append(el("span","report-unclassified","Без типа: "+player.unclassified_touches));
    const activity=el("div","report-player-activity"+(player.activity===null?" is-unavailable":""));
    if(player.activity!==null){
      activity.style.setProperty("--report-activity",String(player.activity));
      const icon=el("span","report-activity-icon");icon.setAttribute("aria-hidden","true");
      icon.append(...activityFigures.map(activityFigure));
      activity.append(icon);
    }
    const ratio=player.activity===null?"—":Number(player.activity).toLocaleString("ru-RU",{maximumFractionDigits:4})+"×";
    activity.append(el("span","report-activity-caption","Активность"),el("span","report-activity-value",ratio));
    activity.setAttribute("aria-label","Активность "+ratio+" от среднего по составу");
    footer.append(activity);
    card.append(header,metrics,footer);
    return card;
  };
  const show=(data,mode)=>{
    if(data.schema!=="volley-native-player-statistics/1"||
      !hash(data.source_fingerprint)||!Array.isArray(data.sources)||
      !Array.isArray(data.players)||!Array.isArray(data.raw_tracks)||
      !Array.isArray(data.roster_player_ids)||
      typeof data.auto_player_identity_enabled!=="boolean"||
      typeof data.automatic_action_classification!=="boolean"||
      data.automatic_action_classification_verified!==false||
      !data.estimated_actions||typeof data.estimated_actions!=="object"||
      !finiteCount(data.rally_candidates)||!finiteCount(data.estimated_hits)||
      !finiteCount(data.initial_serve_candidates)||!finiteCount(data.unassigned_initial_serve_candidates)||
      typeof data.mean_player_touches!=="number"||!Number.isFinite(data.mean_player_touches)||data.mean_player_touches<0||
      data.all_contacts_physically_verified!==true)
      throw Error("Сервер вернул неподтверждённую статистику");
    const selected=selection();
    if(data.project_id!==state.project||
      data.video_count!==selected.length||data.sources.length!==selected.length)
      throw Error("Не совпадает набор видео в статистике");
    const distinct=new Set();
    let perSourceTotal=0,perSourceUnknown=0,perSourceServes=0,perSourceUnknownServes=0;
    for(const src of data.sources){
      if(!state.selections.has(src.video)||state.selections.get(src.video)!==src.rally_run_id||
        distinct.has(src.video)||
        !hash(src.source_replay_sha256)||!hash(src.source_touch_sha256)||
        !hash(src.source_rally_sha256)||
        !(hash(src.identity_revision)||src.identity_revision==="absent")||
        !finiteCount(src.total_touch_candidates)||
        !finiteCount(src.unassigned_touch_candidates)||
        !finiteCount(src.initial_serve_candidates)||
        !finiteCount(src.unassigned_initial_serve_candidates)||
        src.unassigned_touch_candidates>src.total_touch_candidates||
        src.unassigned_initial_serve_candidates>src.initial_serve_candidates)
        throw Error("Нарушена версия или целостность источника статистики");
      distinct.add(src.video);
      perSourceTotal+=src.total_touch_candidates;
      perSourceUnknown+=src.unassigned_touch_candidates;
      perSourceServes+=src.initial_serve_candidates;
      perSourceUnknownServes+=src.unassigned_initial_serve_candidates;
    }
    const total=data.total_touch_candidates,assigned=data.assigned_touch_candidates,
      unknown=data.unassigned_touch_candidates;
    const roster=new Set(data.roster_player_ids);
    if(roster.size!==data.roster_player_ids.length||
      roster.size!==data.players.length||
      data.players.some(p=>!roster.has(p.player_id)||typeof p.name!=="string"||
        !p.name.trim())||
      new Set(data.players.map(p=>p.player_id)).size!==roster.size)
      throw Error("Отчёт может содержать только игроков из состава проекта");
    if(![total,assigned,unknown,perSourceTotal,perSourceUnknown].every(finiteCount)||
      total!==assigned+unknown||total!==perSourceTotal||unknown!==perSourceUnknown||
      data.players.some(p=>!finiteCount(p.touches)||!finiteCount(p.unclassified_touches)||
        !(p.activity===null||(typeof p.activity==="number"&&Number.isFinite(p.activity)&&p.activity>=0)))||
      data.players.reduce((sum,p)=>sum+p.touches,0)!==assigned)
      throw Error("Количество касаний не согласуется с проверенными источниками");
    if(!data.players.every(p=>finiteCount(p.initial_serve_candidates))||
      perSourceServes!==data.initial_serve_candidates||
      perSourceUnknownServes!==data.unassigned_initial_serve_candidates||
      data.players.reduce((sum,p)=>sum+p.initial_serve_candidates,0)+data.unassigned_initial_serve_candidates!==data.initial_serve_candidates)
      throw Error("Число начальных подач не соответствует проверенному источнику");
    if (![data.estimated_actions.attack ?? 0,data.estimated_actions.block ?? 0].every(finiteCount))
      throw Error("Некорректное количество игровых действий");
    // The public Report view shows only touches attributed to roster athletes;
    // unmatched source Track IDs remain internal diagnostics, never cards.
    ui.total.textContent=String(assigned);
    ui.rallies.textContent=String(data.rally_candidates);
    ui.serves.textContent=String(data.initial_serve_candidates);
    ui.attacks.textContent=data.automatic_action_classification?String(data.estimated_actions.attack ?? 0):"—";
    ui.blocks.textContent=data.automatic_action_classification?String(data.estimated_actions.block ?? 0):"—";
    ui.summary.classList.remove("hidden");
    clear(ui.players);
    for(const player of data.players){
      if(!["manually_reviewed_native_track_ids","photo_auto_or_reviewed_native_track_ids"].includes(player.identity_source))
        throw Error("Отчёт содержит игрока без привязки к составу");
      ui.players.append(playerCard(player,data.automatic_action_classification));
    }
    note(data.players.length?"Учтены касания игроков внутри розыгрышей. Для обновления после изменения привязок нажмите «Пересчитать статистику».":
      "Добавьте игроков и подтвердите их привязки, чтобы увидеть персональную статистику.");
  };
  const discover=async()=>{
    const project=ui.project.value;
    state.token++;const token=state.token;
    state.project=project;state.selections.clear();invalidate();
    window.VolleyballTasks?.clear("report-loading");
    if(!project){note("Выберите проект для просмотра статистики.");return;}
    note("Загружаем статистику игры…");
    window.VolleyballTasks?.set("report-loading",{label:"Отчёт",stage:"Загрузка статистики",progress:null});
    try{
      const result=await query("/api/projects");
      if(token!==state.token)return;
      const entry=(Array.isArray(result.projects)?result.projects:[]).find(item=>item.id===project);
      if(!entry)throw Error("Проект отсутствует в реестре");
      const videos=Array.isArray(entry.videos)?entry.videos:[];
      if(videos.length>40)throw Error("Более 40 видео: выберите меньший проект");
      const overviews=await window.VolleyballOverview.loadMany(project,videos);
      if(token!==state.token)return;
      for(const [video,overview] of overviews){
        const stage=(Array.isArray(overview.stages)?overview.stages:[]).find(row=>row.key==="rallies");
        const versions=(Array.isArray(stage?.items)?stage.items:[])
          .filter(row=>/^[a-f0-9]{32}$/.test(row.id)&&hash(row.sha256));
        if(stage?.truncated)throw Error("Слишком много версий CPU для "+video);
        if(versions.length)state.selections.set(video,versions[0].id);
      }
      if(!state.selections.size){note("Сначала выполните анализ видео, чтобы увидеть статистику.");return;}
      const [data,photos]=await Promise.all([post({project_id:project,sources:selection(),mode:"open"}),
        loadPhotos().catch(()=>new Map())]);
      if(token!==state.token)return;
      if(data.snapshot_persisted!==true||!hash(data.saved_revision))
        throw Error("Сохранённая статистика недоступна");
      state.photos=photos;
      show(data,"open");
      state.savedRevision=data.saved_revision;
      ui.recalculate.disabled=false;
      if(state.selections.size<videos.length)note("Учтены не все видео: для некоторых нет результатов анализа.");
    }catch(e){if(token===state.token){invalidate();note(e.message,true);}}
    finally{if(token===state.token)window.VolleyballTasks?.clear("report-loading");}
  };
  const recalculate=async()=>{
    const project=ui.project.value;
    if(!project||project!==state.project||!state.selections.size||
      !hash(state.savedRevision)||ui.recalculate.disabled)return;
    const token=++state.token;
    ui.recalculate.disabled=true;
    note("Пересчитываем статистику по текущим привязкам игроков…");
    window.VolleyballTasks?.set("report-recalculate",{label:"Отчёт",stage:"Пересчёт статистики",progress:null});
    try{
      const data=await post({project_id:project,sources:selection(),mode:"recalculate",
        expected_revision:state.savedRevision});
      if(token!==state.token)return;
      if(data.snapshot_persisted!==true||!hash(data.saved_revision))
        throw Error("Статистика не была сохранена");
      show(data,"recalculate");
      state.savedRevision=data.saved_revision;
      note("Статистика пересчитана по текущим привязкам игроков. Учтены только розыгрыши.");
    }catch(e){
      if(token===state.token)note(e.message,true);
    }finally{
      if(token===state.token)ui.recalculate.disabled=false;
      window.VolleyballTasks?.clear("report-recalculate");
    }
  };
  ui.recalculate.addEventListener("click",recalculate);
  window.addEventListener("volley:tab-change",event=>{
    if(event.detail?.tab==="report")discover();
  });
  window.addEventListener("volley:project-ready",()=>{
    if(location.hash==="#report")discover();
  });
  window.addEventListener("volley:pipeline-complete",()=>{
    if(location.hash==="#report")discover();
  });
  if(location.hash==="#report")discover();
})();
