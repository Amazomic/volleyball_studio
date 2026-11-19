 "use strict";
(() => {
  const $=id=>document.getElementById(id);
  const ui={
    toggle:$("studio-video-library-toggle"),panel:$("studio-video-library"),
    close:$("studio-video-library-close"),search:$("studio-video-library-search"),
    list:$("studio-video-library-list"),status:$("studio-video-library-status"),
    player:$("studio-video-library-player"),media:$("studio-video-player-media"),
    title:$("studio-video-player-title"),playerClose:$("studio-video-player-close")
  };
  if(!ui.toggle||!ui.panel)return;
  let opened=false,loadEpoch=0,rows=[];
  const icon=(kind)=>{
    const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
    svg.setAttribute("viewBox","0 0 24 24");svg.setAttribute("width","16");
    svg.setAttribute("height","16");svg.setAttribute("fill","none");
    svg.setAttribute("stroke","currentColor");svg.setAttribute("stroke-width","1.7");
    svg.setAttribute("stroke-linecap","round");svg.setAttribute("stroke-linejoin","round");
    const path=document.createElementNS(svg.namespaceURI,"path");
    path.setAttribute("d",{
      play:"M8 5v14l11-7z",download:"M12 3v12m-5-5 5 5 5-5M5 17v4h14v-4",
      trash:"M4 7h16M10 4h4m-8 3 1 14h10l1-14M10 11v6m4-6v6",
      folder:"M3 6h7l2 3h9v11H3z"
    }[kind]);
    svg.append(path);return svg;
  };
  const button=(kind,title,onClick)=>{
    const item=document.createElement("button");item.type="button";
    item.title=title;item.setAttribute("aria-label",title);
    item.append(icon(kind));item.addEventListener("click",onClick);return item;
  };
  const date=value=>{
    const n=Number(value);
    if(!Number.isFinite(n)||n<=0)return "Дата неизвестна";
    return new Date(n*1000).toLocaleString("ru-RU",{day:"2-digit",month:"2-digit",
      year:"2-digit",hour:"2-digit",minute:"2-digit"});
  };
  const duration=value=>{
    const n=Math.max(0,Math.round(Number(value)||0));
    return String(Math.floor(n/60)).padStart(2,"0")+":"+String(n%60).padStart(2,"0");
  };
  const closePlayer=()=>{ui.media.pause();ui.media.removeAttribute("src");
    ui.media.load();ui.player.close();};
  ui.playerClose.addEventListener("click",closePlayer);
  ui.player.addEventListener("close",()=>{ui.media.pause();ui.media.removeAttribute("src");
    ui.media.load();});
  const hide=()=>{opened=false;loadEpoch++;ui.panel.classList.add("hidden");
    ui.toggle.setAttribute("aria-expanded","false");};
  ui.close.addEventListener("click",hide);
  ui.toggle.addEventListener("click",()=>{
    if(opened){hide();return;}
    opened=true;ui.panel.classList.remove("hidden");
    ui.toggle.setAttribute("aria-expanded","true");ui.search.focus();void refresh();
  });
  document.addEventListener("keydown",event=>{
    if(event.key==="Escape"&&opened&&!ui.player.open)hide();
  });
  const request=async(path,options={})=>{
    const response=await window.VolleyballStudioActivity.fetch(path,{
      cache:"no-store",credentials:"same-origin",...options
    });
    const body=await response.json().catch(()=>({}));
    if(!response.ok)throw Error(body.message||body.error||"HTTP "+response.status);
    return body;
  };
  const refresh=async()=>{
    const epoch=++loadEpoch;ui.status.textContent="Загружаем готовые видео…";
    try{
      const data=await request("/api/montage/native/library");
      if(epoch!==loadEpoch||!opened)return;
      rows=Array.isArray(data.videos)?data.videos.filter(row=>
        /^[a-f0-9]{12}$/.test(row.project_id||"")&&
        /^montage_[a-f0-9]{32}$/.test(row.id||"")&&
        typeof row.url==="string"&&row.url.startsWith("/api/montage/native/media?")):[];
      draw();
    }catch(error){
      if(epoch!==loadEpoch||!opened)return;
      ui.list.replaceChildren();ui.status.textContent=error.message;
    }
  };
  const draw=()=>{
    const query=ui.search.value.trim().toLocaleLowerCase("ru");
    const filtered=rows.filter(row=>(String(row.project_name||"")+" "+String(row.label||"")+" "+String(row.name||""))
      .toLocaleLowerCase("ru").includes(query));
    ui.status.textContent="Видео: "+filtered.length+" из "+rows.length;
    ui.list.replaceChildren();
    if(!filtered.length){
      const empty=document.createElement("p");empty.className="studio-video-library-status";
      empty.textContent=rows.length?"Ничего не найдено.":"Готовых видео пока нет.";
      ui.list.append(empty);return;
    }
    const groups=new Map();
    for(const row of filtered){
      if(!groups.has(row.project_id))groups.set(row.project_id,[]);
      groups.get(row.project_id).push(row);
    }
    for(const [project,items] of groups){
      const section=document.createElement("section");section.className="studio-video-group";
      const heading=document.createElement("h3");heading.append(icon("folder"));
      const name=document.createElement("span");
      name.textContent=(items[0].project_name||project)+" · "+items.length;
      heading.append(name);section.append(heading);
      for(const item of items){
        const entry=document.createElement("div");entry.className="studio-video-entry";
        const details=document.createElement("div");details.className="studio-video-entry-details";
        const title=document.createElement("div");title.className="studio-video-entry-name";
        title.textContent=item.label||"Созданное видео";
        title.title=item.label||item.name||"Созданное видео";
        const meta=document.createElement("div");meta.className="studio-video-entry-meta";
        meta.textContent=date(item.created_at)+" · "+duration(item.duration_sec);
        details.append(title,meta);
        const sources=Array.isArray(item.source_run_ids)?item.source_run_ids.filter(id=>
          /^[a-f0-9]{32}$/.test(id)):[];
        const apply=document.createElement("button");
        apply.className="studio-video-apply";apply.type="button";
        apply.textContent="Применить анализ";
        apply.title=sources.length?"Открыть сохранённую версию постобработки":
          "У этого видео отсутствует сохранённая ссылка на CPU-анализ";
        apply.disabled=!sources.length;
        apply.addEventListener("click",async()=>{
          apply.disabled=true;ui.status.textContent="Восстанавливаем анализ…";
          try{
            if(typeof window.VolleyballStudioApplyMontageAnalysis!=="function")
              throw Error("Выбор анализа ещё не загружен");
            await window.VolleyballStudioApplyMontageAnalysis({
              project_id:project,source_run_ids:sources
            });
            hide();
          }catch(error){ui.status.textContent=error.message;apply.disabled=false;}
        });
        details.append(apply);
        const actions=document.createElement("div");actions.className="studio-video-entry-actions";
        actions.append(button("play","Открыть видео",()=>{
          ui.title.textContent=(item.project_name||project)+" · "+(item.label||"Видео");
          ui.media.src=item.url;ui.player.showModal();
        }));
        const download=document.createElement("a");
        download.href=item.download_url||item.url;
        download.download=project+"_"+item.id+".mp4";
        download.title="Скачать MP4";download.setAttribute("aria-label","Скачать MP4");
        download.append(icon("download"));actions.append(download);
        actions.append(button("trash","Удалить видео",async()=>{
          if(!window.confirm("Удалить готовое видео «"+(item.label||item.name)+"» из проекта «"+
             (item.project_name||project)+"»?"))return;
          ui.status.textContent="Удаляем видео…";
          try{
            const result=await request("/api/montage/native/delete",{
              method:"POST",headers:{"Content-Type":"application/json"},
              body:JSON.stringify({project_id:project,job_id:item.id})
            });
            if(result.deleted!==true)throw Error("Сервер не подтвердил удаление");
            await refresh();
          }catch(error){ui.status.textContent=error.message;}
        }));
        entry.append(details,actions);section.append(entry);
      }
      ui.list.append(section);
    }
  };
  ui.search.addEventListener("input",draw);
  window.addEventListener("volley:project-ready",()=>{if(opened)void refresh();});
  window.addEventListener("volley:montage-complete",()=>{if(opened)void refresh();});
})();
