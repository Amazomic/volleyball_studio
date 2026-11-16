"use strict";
(()=>{
  // The original HEVC file stays immutable; only short browser-compatible MP4 fragments are fetched.
  const SEGMENT_SECONDS=18;
  const MIME='video/mp4; codecs="avc1.42C01F, mp4a.40.2"';
  if(!window.MediaSource||!MediaSource.isTypeSupported(MIME))return;

  const sessions=new WeakMap();
  const sourceInfo=new Map();
  const urlFor=(path,params)=>path+"?"+new URLSearchParams(params);
  const announce=(video,status)=>window.dispatchEvent(
    new CustomEvent("volley:playback-status",{detail:{...status,video}}));

  function waitForUpdate(buffer,perform){
    return new Promise((resolve,reject)=>{
      const done=()=>{cleanup();resolve();};
      const failed=()=>{cleanup();reject(Error("MSE SourceBuffer error"));};
      const cleanup=()=>{
        buffer.removeEventListener("updateend",done);
        buffer.removeEventListener("error",failed);
      };
      buffer.addEventListener("updateend",done,{once:true});
      buffer.addEventListener("error",failed,{once:true});
      try{perform();}catch(error){cleanup();reject(error);}
    });
  }

  function start(video,source,meta){
    const url=new URL(source,location.href),project=url.searchParams.get("project_id"),
          name=url.searchParams.get("video");
    if(!project||!name)return;
    const previous=sessions.get(video);
    if(previous?.source===source)return;
    previous?.close();
    const initial=Number.isFinite(video.currentTime)?video.currentTime:0;
    const wasPlaying=!video.paused;
    const media=new MediaSource(),objectUrl=URL.createObjectURL(media);
    let buffer=null,closed=false,queue=Promise.resolve(),busy=new Set();
    const loaded=new Set();
    const duration=Number(meta.duration_sec);
    const last=Math.max(0,Math.ceil(duration/SEGMENT_SECONDS)-1);
    const isCurrent=()=>!closed&&sessions.get(video)?.source===source;
    const close=()=>{
      if(closed)return;
      closed=true;
      video.removeEventListener("timeupdate",bufferAhead);
      video.removeEventListener("seeking",onSeek);
      if(media.readyState==="open"){try{media.endOfStream();}catch{}}
      URL.revokeObjectURL(objectUrl);
    };

    const ensure=index=>{
      if(!isCurrent()||!Number.isInteger(index)||index<0||index>last||
         busy.has(index)||loaded.has(index))return queue;
      busy.add(index);
      queue=queue.catch(()=>{}).then(async()=>{
        if(!isCurrent())return;
        announce(video,{state:"preparing",progress:0});
        const response=await fetch(urlFor("/api/video-fragment",{
          project_id:project,video:name,index
        }),{credentials:"same-origin",cache:"no-store"});
        if(!response.ok){
          let message="HTTP "+response.status;
          try{const json=await response.json();message=json.error||json.message||message;}catch{}
          throw Error(message);
        }
        const bytes=await response.arrayBuffer();
        if(!isCurrent()||!buffer||bytes.byteLength<1024)return;
        const seekStart=index*SEGMENT_SECONDS;
        // Discard far-away buffered ranges before jumping several minutes.
        if(buffer.buffered.length){
          const first=buffer.buffered.start(0);
          const end=buffer.buffered.end(buffer.buffered.length-1);
          if(seekStart<first-SEGMENT_SECONDS||seekStart>end+SEGMENT_SECONDS*3){
            await waitForUpdate(buffer,()=>buffer.remove(0,Math.min(duration,end)));
            loaded.clear();
          }else if(end-first>SEGMENT_SECONDS*12 && video.currentTime>SEGMENT_SECONDS*7){
            const cutoff=Math.max(0,video.currentTime-SEGMENT_SECONDS*5);
            if(cutoff>first){
              await waitForUpdate(buffer,()=>buffer.remove(0,Math.min(cutoff,end)));
              for(const n of loaded)if((n+1)*SEGMENT_SECONDS<=cutoff)loaded.delete(n);
            }
          }
        }
        if(!isCurrent())return;
        buffer.timestampOffset=seekStart;
        await waitForUpdate(buffer,()=>buffer.appendBuffer(bytes));
        loaded.add(index);
        announce(video,{state:"ready",progress:100});
      }).catch(error=>{
        if(isCurrent()){
          announce(video,{state:"failed",error:String(error.message||error)});
          console.warn("Browser fragment playback:",error);
        }
      }).finally(()=>busy.delete(index));
      return queue;
    };

    const bufferAhead=()=>{
      if(!isCurrent()||!buffer||video.seeking)return;
      const position=video.currentTime;
      if(!Number.isFinite(position))return;
      let bufferedEnd=position;
      for(let i=0;i<video.buffered.length;i++){
        if(video.buffered.start(i)<=position+0.1&&video.buffered.end(i)>=position)
          bufferedEnd=Math.max(bufferedEnd,video.buffered.end(i));
      }
      const index=Math.floor(Math.max(0,bufferedEnd-0.01)/SEGMENT_SECONDS)+
        (bufferedEnd-position<12?1:0);
      if(bufferedEnd-position<12)ensure(index);
    };
    const onSeek=()=>{
      if(!isCurrent())return;
      const position=video.currentTime;
      if(!Number.isFinite(position))return;
      const index=Math.min(last,Math.floor(Math.max(0,position)/SEGMENT_SECONDS));
      let buffered=false;
      for(let i=0;i<video.buffered.length;i++){
        if(video.buffered.start(i)<=position&&video.buffered.end(i)>position)buffered=true;
      }
      if(!buffered)ensure(index);
    };
    sessions.set(video,{source,close});
    video.addEventListener("timeupdate",bufferAhead);
    video.addEventListener("seeking",onSeek);
    media.addEventListener("sourceopen",async()=>{
      if(!isCurrent())return;
      try{
        media.duration=duration;
        buffer=media.addSourceBuffer(MIME);
        await ensure(Math.min(last,Math.floor(initial/SEGMENT_SECONDS)));
        if(!isCurrent())return;
        if(initial>0)video.currentTime=Math.min(duration-0.01,initial);
        if(wasPlaying)video.play().catch(()=>{});
        bufferAhead();
      }catch(error){
        if(isCurrent())announce(video,{state:"failed",error:String(error)});
      }
    },{once:true});
    video.src=objectUrl;
    video.load();
  }

  function inspect(video,source){
    if(sessions.get(video)?.source===source)return Promise.resolve(true);
    const url=new URL(source,location.href);
    if(url.origin!==location.origin||
       !["/api/video-source","/api/video-media"].includes(url.pathname))
      return Promise.resolve(false);
    const name=url.searchParams.get("video");
    if(!name)return Promise.resolve(false);
    let promise=sourceInfo.get(name);
    if(!promise){
      promise=fetch(urlFor("/api/video-info",{video:name}),{cache:"no-store"})
        .then(response=>response.ok?response.json():null).catch(()=>null);
      sourceInfo.set(name,promise);
    }
    return promise.then(info=>{
      if(video.getAttribute("src")!==source)return true;
      if(info?.codec==="hevc"||info?.codec==="h265"){
        start(video,source,info);return true;
      }
      return false;
    });
  }

  function check(video){
    const source=video.getAttribute("src");
    if(!source)return;
    const previous=sessions.get(video);
    if(previous&&source.startsWith("blob:"))return;
    if(previous&&previous.source!==source){
      previous.close();sessions.delete(video);
    }
    inspect(video,source);
  }
  // Hold the first decode error until the codec is known. Otherwise the legacy
  // fallback starts transcoding the entire multi-hour HEVC file.
  const replaying=new WeakSet();
  document.addEventListener("error",event=>{
    const video=event.target;
    if(video?.tagName!=="VIDEO"||replaying.delete(video))return;
    const src=video.getAttribute("src");
    if(!src)return;
    const url=new URL(src,location.href);
    if(url.origin!==location.origin||
       !["/api/video-source","/api/video-media"].includes(url.pathname))return;
    event.stopImmediatePropagation();
    inspect(video,src).then(handled=>{
      if(!handled&&video.getAttribute("src")===src){
        replaying.add(video);
        video.dispatchEvent(new Event("error"));
      }
    });
  },true);
  const watch=new MutationObserver(mutations=>{
    for(const mutation of mutations){
      if(mutation.target?.tagName==="VIDEO"&&mutation.attributeName==="src")
        check(mutation.target);
    }
  });
  watch.observe(document.documentElement,{subtree:true,attributes:true,attributeFilter:["src"]});
  for(const video of document.querySelectorAll("video[src]"))check(video);
})();