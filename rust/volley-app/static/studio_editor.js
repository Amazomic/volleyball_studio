"use strict";
// Browser-local media preview, precise trim timeline and image-relative court geometry.
// Rust persists source edits; source video bytes remain unchanged.
(() => {
  const $=id=>document.getElementById(id);
  const wrapper=$("studio-analysis-editor");
  const video=$("studio-analysis-player"),still=$("studio-analysis-still"),canvas=$("studio-analysis-court");
  const stage=$("studio-analysis-stage"),frame=$("studio-analysis-frame");
  const media=stage.parentElement,context=canvas.getContext("2d");
  const begin=$("studio-analysis-trim-start"),end=$("studio-analysis-trim-end");
  const seek=$("studio-analysis-seek"),clock=$("studio-analysis-time");
  const rangeLabel=$("studio-analysis-range-label");
  const band=$("studio-analysis-selected-range");
  const startMarker=$("studio-analysis-start-marker"),endMarker=$("studio-analysis-end-marker");
  const playhead=$("studio-analysis-playhead"),playButton=$("studio-analysis-toggle-play");
  const status=$("studio-analysis-edit-status"),save=$("studio-analysis-edit-save");
  const modeButton=$("studio-analysis-court-mode");
  const zoomOut=$("studio-analysis-zoom-out"),zoomIn=$("studio-analysis-zoom-in");
  const zoomLabel=$("studio-analysis-zoom-label");
  const minZoom=0.52,maxZoom=1.24,zoomStep=0.08;
  let project="",source="",generation=0,corners=[],drag=-1,duration=0;
  let marking=false,zoom=0.84,settingsLoaded=false,settingsLoading=false,net=null,netDrag=-1,geometrySource="manual";
  let markingNet=false,netPoints=[];
  // HEVC and other unsupported browser codecs still allow exact source-frame
  // seeking, trimming and calibration, without transcoding the full match.
  let stillMode=false,stillCursor=0,stillReady=false,stillDelay=null,stillMeta=null;
  const cursor=()=>stillMode?stillCursor:(Number.isFinite(video.currentTime)?video.currentTime:0);
  const renderStill=()=>{
    if(stillDelay!==null)clearTimeout(stillDelay);
    if(!stillMode||!valid())return;
    const projectAtRequest=project,sourceAtRequest=source,token=generation;
    const at=stillCursor;
    stillDelay=setTimeout(()=>{
      stillDelay=null;
      if(!stillMode||token!==generation||projectAtRequest!==project||sourceAtRequest!==source)return;
      still.src=url("/api/video-preview",{
        video:source,mode:"source",time:at.toFixed(3),width:960
      });
    },120);
  };
  const fallbackFrames=(meta)=>{
    if(stillMode||!valid()||!meta)return;
    stillMode=true;stillCursor=0;stillReady=false;
    video.pause();video.removeAttribute("src");video.load();
    video.hidden=true;still.hidden=false;
    playButton.disabled=true;playButton.textContent="Покадровый просмотр";
    duration=Number(meta.duration_sec)||0;
    const width=Number(meta.width),height=Number(meta.height);
    if(width>0&&height>0)media.style.setProperty("--frame-ratio",String(width/height));
    begin.max=String(duration);end.max=String(duration);
    seek.max=String(Math.max(1,Math.round(duration*1000)));
    if(!end.value||Number(end.value)<=0||Number(end.value)>duration)
      end.value=duration.toFixed(2);
    updateClock();draw();renderStill();
    message("Формат исходника не поддерживается браузером. Доступны кадры по временной шкале, обрезка и разметка поля; непрерывное воспроизведение недоступно.");
  };

  let baseRevision="",dirty=false,draftTimer=null,editEpoch=0;
  const revision=data=>data.edit_revision||JSON.stringify([data.trim_start_sec??0,data.trim_end_sec??null,data.court||[],data.net||null,data.source_revision??null]);
  const draftKey=(id,name)=>"volley:editor-draft:v1:"+JSON.stringify([id,name]);
  const readDraft=(id,name)=>{
    try{
      const raw=localStorage.getItem(draftKey(id,name));
      if(!raw||raw.length>4096)return null;
      const row=JSON.parse(raw),value=row.value;
      if(!Number.isFinite(row.expires)||row.expires<=Date.now()||typeof row.base!=="string"||!value||
        !Number.isFinite(value.trim_start_sec)||value.trim_start_sec<0||
        !(value.trim_end_sec===null||Number.isFinite(value.trim_end_sec))||
        !Array.isArray(value.court)||value.court.length>4||
        value.court.some(p=>!Array.isArray(p)||p.length!==2||p.some(v=>!Number.isFinite(v)||v<-.5||v>1.5)))return null;
      return row;
    }catch{return null;}
  };
  const removeDraft=()=>{try{localStorage.removeItem(draftKey(project,source));}catch{}};
  const flushDraft=()=>{
    if(draftTimer!==null)clearTimeout(draftTimer);
    draftTimer=null;
    if(!dirty||!project||!source||!baseRevision)return;
    const row={base:baseRevision,expires:Date.now()+14*86400000,
      value:{trim_start_sec:Number(begin.value),trim_end_sec:Number(end.value)||null,court:corners,net,geometry_source:geometrySource}};
    try{const text=JSON.stringify(row);if(text.length<=4096)localStorage.setItem(draftKey(project,source),text);}catch{}
  };
  window.addEventListener("beforeunload",flushDraft);
  const clamp=(n,lo,hi)=>Math.min(hi,Math.max(lo,n));
  const time=n=>{
    if(!Number.isFinite(n))return "00:00.00";
    const cs=Math.floor(Math.max(0,n)*100+0.000001);
    const seconds=Math.floor(cs/100),hours=Math.floor(seconds/3600);
    const mm=String(Math.floor(seconds%3600/60)).padStart(2,"0");
    const ss=String(seconds%60).padStart(2,"0");
    const cc=String(cs%100).padStart(2,"0");
    return (hours?String(hours)+":":"")+mm+":"+ss+"."+cc;
  };
  const message=(text,failed=false)=>{status.textContent=text;status.classList.toggle("is-error",failed);};
  const url=(path,params)=>path+"?"+new URLSearchParams(params).toString();
  const api=async(path,options={})=>{
    const result=await window.VolleyballStudioActivity.fetch(path,{cache:"no-store",credentials:"same-origin",...options});
    const body=await result.json().catch(()=>({}));
    if(!result.ok)throw Error(body.message||body.error||"HTTP "+result.status);
    return body;
  };
  // Small edit metadata only. Never cache the original video or frame bytes.
  const editCache=(()=>{
    const key=(id,name)=>"volley:studio-video-edit:v1:"+id+":"+encodeURIComponent(name);
    const read=(id,name)=>{
      try{
        const raw=sessionStorage.getItem(key(id,name));
        if(!raw||raw.length>4096)return null;
        const item=JSON.parse(raw);
        return item.expires>Date.now()?item.value:null;
      }catch{return null;}
    };
    const write=(id,name,value)=>{
      try{sessionStorage.setItem(key(id,name),JSON.stringify({
        expires:Date.now()+600000,value
      }));}catch{}
    };
    return {read,write};
  })();
  const valid=()=>Boolean(project&&source);

  const updateZoomControls=()=>{
    zoomLabel.textContent=Math.round((marking?zoom:1)*100)+"%";
    zoomOut.disabled=!valid()||!marking||zoom<=minZoom+0.001;
    zoomIn.disabled=!valid()||!marking||zoom>=maxZoom-0.001;
  };
  const setZoom=next=>{
    zoom=clamp(Math.round(next*100)/100,minZoom,maxZoom);
    stage.style.setProperty("--editor-zoom",marking?String(zoom):"1");
    updateZoomControls();
    draw();
  };
  const setMarking=active=>{
    marking=Boolean(active);
    stage.classList.toggle("marking",marking);
    video.controls=!marking;
    modeButton.textContent=marking?"Завершить разметку":"Разметить поле";
    modeButton.setAttribute("aria-pressed",String(marking));
    stage.style.setProperty("--editor-zoom",marking?String(zoom):"1");
    updateZoomControls();
    draw();
  };
  const syncControls=()=>{
    for(const node of wrapper.querySelectorAll("button,input"))node.disabled=!valid();
    updateZoomControls();
  };
  // Three perspective-aware zones are derived from the four saved court corners
  // and the observed net top. Nothing extra or synthetic is persisted.
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const line=(a,b)=>cross([...a,1],[...b,1]);
  const intersect=(a,b)=>{
    const v=cross(a,b);
    return Number.isFinite(v[2])&&Math.abs(v[2])>1e-8?[v[0]/v[2],v[1]/v[2]]:null;
  };
  const zonesForMarkup=()=>{
    if(corners.length!==4)return null;
    const [a,b,c,d]=corners;
    const center=intersect(line(a,c),line(b,d));
    if(!center)return null;
    const transverseVanishing=intersect(line(a,d),line(b,c));
    const divider=transverseVanishing
      ?line(center,transverseVanishing)
      :line(center,[center[0]+d[0]-a[0],center[1]+d[1]-a[1]]);
    const left=intersect(divider,line(a,b));
    const right=intersect(divider,line(d,c));
    if(!left||!right)return null;
    // The near and far halves are 9 x 9 m, not arbitrary pixel halves.
    const near=[a,left,right,d],far=[left,b,c,right];
    let surface=null;
    if(net){
      const [u,v]=net;
      const dist=(p,q)=>Math.hypot(p[0]-q[0],p[1]-q[1]);
      const [topLeft,topRight]=dist(u,left)+dist(v,right)<=dist(v,left)+dist(u,right)?[u,v]:[v,u];
      surface=[topLeft,topRight,right,left];
    }
    return {near,far,divider:[left,right],surface};
  };
  const legend=document.createElement("div");
  legend.className="analysis-editor-zone-legend";
  legend.setAttribute("aria-label","Зоны разметки площадки");
  legend.style.cssText="display:flex;flex-wrap:wrap;gap:8px 14px;margin:8px 0;font-size:12px";
  for(const [name,color] of [["Ближняя половина","#34d399"],["Дальняя половина","#60a5fa"],["Сетка","#35e0ef"]]){
    const item=document.createElement("span");item.textContent="● "+name;item.style.color=color;legend.appendChild(item);
  }
  media.insertBefore(legend,stage.nextSibling);
  const pointXY=([x,y],stageRect,frameRect)=>[
    frameRect.left-stageRect.left+x*frameRect.width,
    frameRect.top-stageRect.top+y*frameRect.height
  ];
  const draw=()=>{
    const rect=canvas.getBoundingClientRect(),image=frame.getBoundingClientRect();
    if(!rect.width||!rect.height)return;
    const dpr=Math.min(2,window.devicePixelRatio||1);
    canvas.width=Math.round(rect.width*dpr);
    canvas.height=Math.round(rect.height*dpr);
    context.setTransform(dpr,0,0,dpr,0,0);
    context.clearRect(0,0,rect.width,rect.height);
    legend.children[2].style.opacity=net?"1":".42";
    legend.children[2].textContent=net?"● Сетка":"○ Сетка: не размечена";
    if((!corners.length&&!net)||!image.width||!image.height)return;
    const points=corners.map(p=>pointXY(p,rect,image));
    const zones=zonesForMarkup();
    const fillZone=(polygon,color,label)=>{
      const px=polygon.map(p=>pointXY(p,rect,image));
      context.beginPath();px.forEach(([x,y],i)=>i?context.lineTo(x,y):context.moveTo(x,y));
      context.closePath();context.fillStyle=color;context.fill();
      const x=px.reduce((sum,p)=>sum+p[0],0)/px.length;
      const y=px.reduce((sum,p)=>sum+p[1],0)/px.length;
      context.font="600 12px sans-serif";context.textAlign="center";context.textBaseline="middle";
      const w=context.measureText(label).width+14;
      context.fillStyle="rgba(12,24,38,.82)";context.fillRect(x-w/2,y-12,w,24);
      context.fillStyle="#fff";context.fillText(label,x,y);
    };
    if(zones){
      fillZone(zones.near,"rgba(16,185,129,.16)","Ближняя половина");
      fillZone(zones.far,"rgba(59,130,246,.17)","Дальняя половина");
      if(zones.surface)fillZone(zones.surface,"rgba(45,212,191,.28)","Сетка");
      const [a,b]=zones.divider.map(p=>pointXY(p,rect,image));
      context.beginPath();context.moveTo(...a);context.lineTo(...b);
      context.strokeStyle="#d6f1fa";context.lineWidth=1.8;context.setLineDash([6,5]);
      context.stroke();context.setLineDash([]);
    }
    if(points.length){
      context.beginPath();points.forEach(([x,y],i)=>i?context.lineTo(x,y):context.moveTo(x,y));
      if(points.length===4)context.closePath();
      context.strokeStyle="#ffd166";context.lineWidth=2.5;context.stroke();
    }
    if(net){
      const endpoints=net.map(p=>pointXY(p,rect,image));
      context.beginPath();
      context.moveTo(...endpoints[0]);context.lineTo(...endpoints[1]);
      context.strokeStyle="#35e0ef";context.lineWidth=3;context.stroke();
      endpoints.forEach(([x,y])=>{
        context.beginPath();context.arc(x,y,6,0,Math.PI*2);
        context.fillStyle="#35e0ef";context.fill();
      });
    }    points.forEach(([x,y],i)=>{
      const [nx,ny]=corners[i];
      context.beginPath();context.arc(x,y,7,0,Math.PI*2);
      context.fillStyle=nx<0||nx>1||ny<0||ny>1?"#ff984f":"#ffd166";
      context.fill();context.lineWidth=2;context.strokeStyle="#121b2b";context.stroke();
      context.fillStyle="#0b1421";context.font="bold 11px sans-serif";
      context.textAlign="center";context.textBaseline="middle";
      context.fillText(String(i+1),x,y);
    });
  };
  const point=event=>{
    const rect=frame.getBoundingClientRect();
    if(!rect.width||!rect.height)return [0,0];
    // Image-relative coordinates can be outside [0,1] in the margin.
    // Guard extreme values to match the bounded Rust API.
    return [clamp((event.clientX-rect.left)/rect.width,-0.5,1.5),
      clamp((event.clientY-rect.top)/rect.height,-0.5,1.5)];
  };
  const markModified=()=>{
    updateClock();
    dirty=true;editEpoch++;
    if(draftTimer!==null)clearTimeout(draftTimer);
    draftTimer=setTimeout(flushDraft,250);
    if(valid())message("Есть несохранённые изменения");
  };
  canvas.addEventListener("pointerdown",event=>{
    if(!marking||!valid()||!(stillMode?stillReady:video.videoWidth))return;
    const at=point(event),rect=canvas.getBoundingClientRect();
    const image=frame.getBoundingClientRect();
    if(markingNet){
      netPoints.push(at);
      if(netPoints.length===2){net=netPoints.map(p=>[...p]);netPoints=[];markingNet=false;geometrySource="manual";markModified();draw();}
      return;
    }
    netDrag=net?net.findIndex(p=>{const [x,y]=pointXY(p,rect,image);return Math.hypot(x-(event.clientX-rect.left),y-(event.clientY-rect.top))<17;}):-1;
    if(netDrag>=0){canvas.setPointerCapture(event.pointerId);return;}
    drag=corners.findIndex(p=>{
      const [x,y]=pointXY(p,rect,image);
      return Math.hypot(x-(event.clientX-rect.left),y-(event.clientY-rect.top))<17;
    });
    if(drag<0&&corners.length<4){
      geometrySource="manual";corners.push(at);drag=corners.length-1;
      markModified();draw();
    }
    if(drag>=0)canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove",event=>{
    if(netDrag>=0){net[netDrag]=point(event);draw();return;}
    if(drag<0)return;
    corners[drag]=point(event);draw();
  });
  const finishDrag=()=>{if(drag>=0||netDrag>=0){geometrySource="manual";markModified();}drag=-1;netDrag=-1;};
  canvas.addEventListener("pointerup",finishDrag);
  canvas.addEventListener("pointercancel",finishDrag);
  $("studio-analysis-net-mode").addEventListener("click",()=>{
    markingNet=true;netPoints=[];setMarking(true);message("Отметьте два конца верхней линии сетки");
  });
  $("studio-analysis-court-undo").addEventListener("click",()=>{
    corners.pop();geometrySource="manual";draw();markModified();
  });
  $("studio-analysis-court-clear").addEventListener("click",()=>{
    corners=[];net=null;markingNet=false;netPoints=[];geometrySource="manual";draw();markModified();
  });
  $("studio-analysis-court-auto").addEventListener("click",async()=>{
    if(!valid())return;
    const token=generation,epoch=editEpoch,button=$("studio-analysis-court-auto");
    button.disabled=true;message("Поиск линий площадки и сетки на нескольких кадрах…");
    try{
      const result=await api("/api/video-court/detect",{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({project_id:project,video:source,start_sec:Math.max(0,cursor())})
      });
      if(token!==generation||epoch!==editEpoch)return;
      if(!result.detected){message(result.message||"Не удалось уверенно распознать разметку. Отметьте углы вручную.",true);return;}
      corners=result.court;net=result.net||null;geometrySource="automatic";setMarking(true);markModified();draw();
      message((result.inferred?"Часть границ достроена по размерам площадки и перспективе — проверьте положение каждой точки. ":result.temporal_composite?"Разметка восстановлена из нескольких кадров — проверьте её. ":result.weak_consensus?"Предварительная разметка (мало совпавших кадров). ":"Найдена разметка. ")+"Подтверждений: "+result.supporting_frames+"/"+result.sampled_frames+". "+(net?"Три зоны выделены: две половины и сетка. ":"Две половины выделены; сетка не распознана, разметьте её вручную. ")+"Проверьте геометрию и сохраните.");
    }catch(error){if(token===generation)message(error.message,true);}
    finally{button.disabled=!valid();}
  });  modeButton.addEventListener("click",()=>setMarking(!marking));
  zoomOut.addEventListener("click",()=>{if(marking)setZoom(zoom-zoomStep);});
  zoomIn.addEventListener("click",()=>{if(marking)setZoom(zoom+zoomStep);});

  const updateClock=()=>{
    const position=cursor();
    const from=Number(begin.value),to=Number(end.value);
    const pct=value=>duration>0?clamp(value/duration*100,0,100):0;
    const left=pct(from),right=pct(to);
    clock.textContent=time(position)+" / "+time(duration);
    rangeLabel.textContent=duration>0?"Отрезок: "+time(from)+" – "+time(to):"Отрезок: —";
    startMarker.style.left=left+"%";
    endMarker.style.left=right+"%";
    playhead.style.left=pct(position)+"%";
    band.style.left=left+"%";
    band.style.width=Math.max(0,right-left)+"%";
    seek.value=duration>0?String(clamp(Math.round(position*1000),0,Math.round(duration*1000))):"0";
  };
  video.addEventListener("loadedmetadata",()=>{
    if(stillMode)return;
    duration=Number.isFinite(video.duration)?video.duration:0;
    if(video.videoWidth>0&&video.videoHeight>0)
      media.style.setProperty("--frame-ratio",String(video.videoWidth/video.videoHeight));
    begin.max=String(duration);end.max=String(duration);
    seek.max=String(Math.max(1,Math.round(duration*1000)));
    if(!end.value||Number(end.value)<=0||Number(end.value)>duration)
      end.value=duration.toFixed(2);
    updateClock();draw();
    message("Выберите отрезок и при необходимости разметьте 4 угла площадки.");
  });
  video.addEventListener("error",()=>{
    if(!valid()||stillMode)return;
    if(stillMeta)fallbackFrames(stillMeta);
    else message("Видео не воспроизводится, не удалось получить данные для покадрового просмотра.",true);
  });
  still.addEventListener("load",()=>{stillReady=true;draw();});
  still.addEventListener("error",()=>{if(stillMode)message("Не удалось декодировать выбранный кадр. Попробуйте другую секунду.",true);});
  video.addEventListener("timeupdate",updateClock);
  video.addEventListener("play",()=>{if(!stillMode)playButton.textContent="Пауза";});
  video.addEventListener("pause",()=>{if(!stillMode)playButton.textContent="▶ Воспроизвести";});
  playButton.addEventListener("click",()=>{
    if(!valid()||stillMode)return;
    if(video.paused){
      const started=video.play();
      if(started&&started.catch)started.catch(error=>message("Не удалось воспроизвести: "+error.message,true));
    }else video.pause();
  });
  seek.addEventListener("input",()=>{
    if(duration>0){
      const at=clamp(Number(seek.value)/1000,0,duration);
      if(stillMode){stillCursor=at;renderStill();}
      else video.currentTime=at;
    }
    updateClock();
  });
  const current=()=>clamp(cursor(),0,duration).toFixed(2);
  $("studio-analysis-set-start").addEventListener("click",()=>{
    begin.value=current();markModified();
  });
  $("studio-analysis-set-end").addEventListener("click",()=>{
    end.value=current();markModified();
  });
  [begin,end].forEach(node=>node.addEventListener("input",markModified));
  save.addEventListener("click",async()=>{
    if(!valid())return;
    const from=Number(begin.value),to=Number(end.value);
    if(!Number.isFinite(from)||from<0||!Number.isFinite(to)||to<=from
      ||(duration>0&&to>duration+.5)){
      message("Конец должен быть позже начала и не выходить за длительность видео.",true);return;
    }
    if(corners.length!==0&&corners.length!==4){
      message("Отметьте все 4 угла площадки или сбросьте разметку.",true);return;
    }
    const token=generation,epoch=editEpoch;
    const court=corners.map(p=>[...p]);
    save.disabled=true;
    try{
      const result=await api("/api/video-edit",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({project_id:project,video:source,trim_start_sec:from,
          trim_end_sec:to,court,net,geometry_source:geometrySource,expected_revision:baseRevision})});
      if(token===generation){
        const data={...result,trim_start_sec:from,trim_end_sec:to,court,net};
        editCache.write(project,source,data);
        window.dispatchEvent(new CustomEvent("volley:court-calibration-updated",{detail:{project_id:project,video:source}}));
        baseRevision=revision(data);
        if(epoch===editEpoch){dirty=false;removeDraft();}else{flushDraft();}
        message("Обрезка и разметка площадки сохранены.");
        if(court.length===4){
          try{
            const learned=await api("/api/training/examples",{
              method:"POST",headers:{"Content-Type":"application/json"},
              body:JSON.stringify({action:"court",project_id:project,video:source,
                frame_index:Math.max(0,Math.floor(cursor()*30)),corners:court,net})
            });
            if(token===generation&&learned.ok)
              message("Разметка сохранена и добавлена в обучающие примеры.");
          }catch(error){
            if(token===generation)
              message("Поле сохранено, но обучающий пример не записан: "+error.message,true);
          }
        }
      }
    }catch(error){
      if(token===generation)message(error.message,true);
    }finally{save.disabled=!valid();}
  });
  const load=async(projectId,videoName,reuse=false)=>{
    if(reuse&&project===projectId&&source===videoName&&(settingsLoaded||settingsLoading))return;
    flushDraft();
    const token=++generation;
    dirty=false;baseRevision="";
    settingsLoaded=false;settingsLoading=false;
    video.pause();
    video.removeAttribute("src");
    video.load();
    if(stillDelay!==null){clearTimeout(stillDelay);stillDelay=null;}
    still.removeAttribute("src");still.hidden=true;stillMode=false;stillReady=false;stillMeta=null;
    video.hidden=false;playButton.disabled=false;playButton.textContent="▶ Воспроизвести";
    project=projectId||"";source=videoName||"";corners=[];net=null;markingNet=false;netPoints=[];
    duration=0;begin.value="0";end.value="0";seek.value="0";seek.max="1000";
    media.style.setProperty("--frame-ratio","1.77778");
    zoom=0.84;setMarking(false);
    updateClock();syncControls();draw();
    if(!valid()){message("Выберите видео проекта.");return;}
    const pending=readDraft(project,source);
    const cached=pending?null:editCache.read(project,source);
    message(cached?"Восстановлены сохранённые настройки видео.":"Загрузка настроек видео…");
    settingsLoading=true;
    try{
      let data=cached||await api(url("/api/video-edit",{project_id:project,video:source}));
      if(token!==generation)return;
      if(!cached)editCache.write(project,source,data);
      baseRevision=revision(data);
      const restored=pending&&pending.base===baseRevision;
      if(restored){data=pending.value;dirty=true;}else if(pending){removeDraft();}
      begin.value=String(data.trim_start_sec??0);
      end.value=data.trim_end_sec==null?"0":String(data.trim_end_sec);
      corners=Array.isArray(data.court)?data.court.filter(p=>
        Array.isArray(p)&&p.length===2&&p.every(v=>Number.isFinite(v)&&v>=-0.5&&v<=1.5)):[];
      net=Array.isArray(data.net)&&data.net.length===2?data.net:null;geometrySource=data.geometry_source||"manual";
      // Metadata is cheap even for large linked files and supplies source
      // duration/aspect ratio when HTMLVideoElement cannot decode HEVC.
      stillMeta=await api(url("/api/video-info",{video:source})).catch(()=>null);
      if(token!==generation)return;
      const hevc=stillMeta&&/^(hevc|h265)$/i.test(stillMeta.codec||"");
      const supported=typeof video.canPlayType!=="function"||
        video.canPlayType('video/mp4; codecs="hvc1.1.6.L93.B0"')==="probably";
      if(hevc&&!supported)fallbackFrames(stillMeta);
      else{
        video.src=url("/api/video-source",{project_id:project,video:source});
        video.load();
        message(restored?"Восстановлен несохранённый черновик. Нажмите «Сохранить» для применения.":"Разметка загружена. Видео читается по частям.");
      }
      settingsLoaded=true;
      updateClock();draw();
    }catch(error){
      if(token===generation)message(error.message,true);
    }finally{
      if(token===generation)settingsLoading=false;
    }
  };
  window.addEventListener("volley:tab-change",event=>{
    if(event.detail?.tab!=="analysis")video.pause();
  });
  window.addEventListener("resize",draw);
  if(window.ResizeObserver){
    const ro=new ResizeObserver(draw);
    ro.observe(stage);ro.observe(frame);
  }
  syncControls();
  window.VolleyballEditor={load};
})();