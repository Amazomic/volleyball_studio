"use strict";
(()=>{
  const pending=new Map(),bypass=new WeakSet(),busy=new WeakMap();
  const original=(project,video)=>"/api/video-source?"+new URLSearchParams({project_id:project,video});
  const request=async(path,options={})=>{
    const response=await window.VolleyballStudioActivity.fetch(path,{
      cache:"no-store",credentials:"same-origin",...options
    });
    const data=await response.json();
    if(!response.ok)throw Error(data.error||data.message||"HTTP "+response.status);
    return data;
  };
  const primary=async(project,video)=>{
    const result=await request("/api/video-primary-preview?"+new URLSearchParams({project_id:project,video}));
    if(result.state!=="ready")return null;
    if(result.project_id!==project||result.video!==video||typeof result.media_url!=="string"||
       !result.media_url.startsWith("/api/video-primary-preview?"))throw Error("Некорректный предпросмотр видео");
    return result.media_url;
  };
  const directLoads=new WeakMap();
  const direct=(video,source,{seek=0,autoplay=false}={})=>{
    const token=Symbol("primary-load");
    directLoads.set(video,token);
    video.pause();
    video.addEventListener("loadedmetadata",()=>{
      if(directLoads.get(video)!==token||video.getAttribute("src")!==source)return;
      if(seek>0)video.currentTime=seek;
      if(autoplay)video.play().catch(()=>{});
    },{once:true});
    video.src=source;video.load();
  };
  const inspectPrimary=async video=>{
    const src=video.getAttribute("src");
    if(!src)return;
    const url=new URL(src,location.href);
    if(url.origin!==location.origin||!["/api/video-media","/api/video-source"].includes(url.pathname))return;
    const project=url.searchParams.get("project_id"),name=url.searchParams.get("video");
    if(!project||!name)return;
    try{
      const media=await primary(project,name);
      if(media&&video.getAttribute("src")===src)
        direct(video,media,{seek:video.currentTime||0,autoplay:!video.paused});
    }catch(error){console.warn("Предпросмотр после анализа:",error.message);}
  };
  const primaryWatch=typeof MutationObserver==="function"?new MutationObserver(mutations=>{
    for(const row of mutations)if(row.target?.tagName==="VIDEO"&&row.attributeName==="src")inspectPrimary(row.target);
  }):null;
  primaryWatch?.observe(document.documentElement,{subtree:true,attributes:true,attributeFilter:["src"]});
  const probe=src=>new Promise(resolve=>{
    const video=document.createElement("video");
    video.muted=true;video.playsInline=true;video.preload="auto";
    let timer,done=false;
    const finish=code=>{
      if(done)return;done=true;clearTimeout(timer);
      video.removeEventListener("loadeddata",loaded);
      video.removeEventListener("error",failed);
      video.pause();video.removeAttribute("src");video.load();resolve(code);
    };
    const loaded=()=>{if(video.readyState>=2&&video.videoWidth>0)finish(0);};
    const failed=()=>finish(video.error?.code??null);
    video.addEventListener("loadeddata",loaded);video.addEventListener("error",failed);
    timer=setTimeout(()=>finish(null),10000);
    video.src=src;video.load();
  });
  const compatible=async(project,video,code,onProgress=()=>{})=>{
    let result=await request("/api/project-preparation/retry",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({project_id:project,video,browser_error_code:code})
    });
    for(;;){
      onProgress(result);
      if(result.state==="original")return null;
      if(result.state==="ready"){
        if(typeof result.media_url!=="string"||!result.media_url.startsWith("/api/video-media?"))
          throw Error("Некорректный адрес копии для просмотра");
        return result.media_url;
      }
      if(!["queued","preparing"].includes(result.state))
        throw Error(result.error||"Подготовка копии для просмотра остановлена");
      await new Promise(resolve=>setTimeout(resolve,1000));
      result=await request("/api/project-preparation?"+new URLSearchParams({
        project_id:project,video,purpose:"playback"
      }));
    }
  };
  const resolve=(project,video)=>{
    const src=original(project,video);
    if(pending.has(src))return pending.get(src);
    const work=(async()=>{
      const ready=await primary(project,video).catch(()=>null);
      if(ready)return ready;
      const code=await probe(src);
      if(code===0||code===null)return src;
      return await compatible(project,video,code)||src;
    })().finally(()=>pending.delete(src));
    pending.set(src,work);return work;
  };
  const notify=(video,result)=>window.dispatchEvent(new CustomEvent("volley:playback-status",{
    detail:{...result,video}
  }));
  document.addEventListener("error",event=>{
    const video=event.target;
    if(video?.tagName!=="VIDEO"||bypass.delete(video))return;
    const src=video.getAttribute("src");
    if(!src)return;
    const url=new URL(src,window.location.origin);
    if(url.origin!==window.location.origin||
      !["/api/video-source","/api/video-media"].includes(url.pathname)||
      url.searchParams.get("playback")==="compatible")return;
    if(video.currentSrc&&new URL(video.currentSrc,window.location.origin).href!==url.href)return;
    const project=url.searchParams.get("project_id"),name=url.searchParams.get("video");
    if(!project||!name)return;
    const code=video.error?.code;
    if(!Number.isInteger(code))return;
    event.stopImmediatePropagation();
    if(busy.get(video)===src)return;
    busy.set(video,src);
    const time=Number.isFinite(video.currentTime)?video.currentTime:0,playing=!video.paused;
    const current=()=>video.getAttribute("src")===src;
    const failed=error=>{
      if(!current())return;
      notify(video,{state:"failed",error:String(error?.message||error||"")});
      bypass.add(video);video.dispatchEvent(new Event("error"));
    };
    compatible(project,name,code,result=>{if(current())notify(video,result);}).then(media=>{
      if(!current())return;
      if(!media){failed("Не удалось загрузить оригинальное видео");return;}
      video.addEventListener("loadedmetadata",()=>{
        if(video.getAttribute("src")!==media)return;
        if(time>0)video.currentTime=time;
        if(playing)video.play().catch(()=>{});
      },{once:true});
      video.src=media;video.load();
    }).catch(failed).finally(()=>{if(busy.get(video)===src)busy.delete(video);});
  },true);
  window.VolleyballPlayback={resolve,primary,direct};
})();
