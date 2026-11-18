#!/usr/bin/env python3
"""Native-resolution player detector benchmark (not the production pipeline).

Reads pixels from ORIGINAL 1920x1080 source, never from prepared 1280x720 or 640x360.
Samples source frames for reproducible testing without altering source file or FPS.
Compares RTMDet-Nano and RF-DETR Medium using identical overlapping source tiles.
Both use RTMPose-M top-down 17 COCO joints cropped directly from the source.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parent.parent
DEFAULT_SRC=Path("/mnt/c/Users/amazomic/Downloads/media (1).mp4")
DEFAULT_OUT=ROOT/"results"/"native_source_detector_20261115"
PALETTE=[(39,223,60),(255,160,60),(80,150,255),(250,75,210),(15,200,200),(210,95,40),(120,130,255),(180,255,70)]
EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
MEAN_RGB=np.array([0.485,0.456,0.406],np.float32)[None,None,:]
STD_RGB=np.array([0.229,0.224,0.225],np.float32)[None,None,:]
MEAN_BGR=np.array([103.53,116.28,123.675],np.float32)[None,None,:]
STD_BGR=np.array([57.375,57.12,58.395],np.float32)[None,None,:]
P_MEAN=np.array([123.675,116.28,103.53],np.float32)[None,None,:]
P_STD=np.array([58.395,57.12,57.375],np.float32)[None,None,:]
SCORE_THRESHOLD=0.30

def session(name:str):
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDA provider unavailable")
    s=ort.InferenceSession(str(ROOT/"models"/name),providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    if s.get_providers()[0]!="CUDAExecutionProvider":
        raise RuntimeError(f"{name}: not running on CUDA")
    return s

def view_rects(width,height):
    # Full frame plus six overlapping native-pixel tiles.
    # The far side of the beach court never gets pre-reduced to 640x360.
    tile_w=min(width,round(width*0.52)); tile_h=min(height,round(height*0.64))
    def anchors(total,size,count):
        if size>=total:return [0]
        return sorted(set(int(round((total-size)*i/(count-1))) for i in range(count)))
    views=[(0,0,width,height)]
    for y in anchors(height,tile_h,2):
        for x in anchors(width,tile_w,3):
            box=(x,y,x+tile_w,y+tile_h)
            if box not in views:views.append(box)
    return views

def iou(a,b):
    x0,y0=np.maximum(a[:2],b[:2]);x1,y1=np.minimum(a[2:],b[2:])
    inter=max(0,x1-x0)*max(0,y1-y0)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(1e-9,aa+bb-inter)

def nms(candidates,width,height):
    sorted_boxes=sorted(candidates,key=lambda x:x[0],reverse=True)
    kept=[]
    for score,box in sorted_boxes:
        if len(kept)>=16:break
        box=np.array(box,float)
        box[[0,2]]=box[[0,2]].clip(0,width-1)
        box[[1,3]]=box[[1,3]].clip(0,height-1)
        if box[2]-box[0]<8 or box[3]-box[1]<20:continue
        if any(iou(box,prev[1])>0.55 for prev in kept):continue
        kept.append((float(score),box))
    return kept

def rfdetr(session,frame,views):
    output=[]
    for x0,y0,x1,y1 in views:
        view=frame[y0:y1,x0:x1]
        rgb=cv2.cvtColor(cv2.resize(view,(576,576),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
        tensor=np.ascontiguousarray(((rgb.astype(np.float32)/255-MEAN_RGB)/STD_RGB).transpose(2,0,1))[None]
        boxes,logits=session.run(["pred_boxes","pred_logits"],{"input":tensor})
        scores=1/(1+np.exp(-np.clip(logits[0,:,1],-40,40)))
        for b,score in zip(boxes[0],scores):
            if not np.isfinite(b).all() or score<SCORE_THRESHOLD:continue
            cx,cy,bw,bh=map(float,b)
            if bw<=0 or bh<=0:continue
            box=[(cx-bw/2)*(x1-x0)+x0,(cy-bh/2)*(y1-y0)+y0,
                 (cx+bw/2)*(x1-x0)+x0,(cy+bh/2)*(y1-y0)+y0]
            output.append((float(score),box))
    return nms(output,frame.shape[1],frame.shape[0])

def rtmdet(session,frame,views):
    output=[]
    for x0,y0,x1,y1 in views:
        view=frame[y0:y1,x0:x1]
        h,w=view.shape[:2]
        scale=min(320/w,320/h)
        new_w=max(1,min(320,int(round(w*scale))))
        new_h=max(1,min(320,int(round(h*scale))))
        scaled=cv2.resize(view,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
        canvas=np.full((320,320,3),114,dtype=np.uint8)
        canvas[:new_h,:new_w]=scaled
        tensor=np.ascontiguousarray(((canvas.astype(np.float32)-MEAN_BGR)/STD_BGR).transpose(2,0,1))[None]
        dets,labels=session.run(["dets","labels"],{"input":tensor})
        for d,label in zip(dets[0],labels[0]):
            if int(label)!=0 or len(d)<5 or not np.isfinite(d).all() or float(d[4])<SCORE_THRESHOLD:continue
            bx0,by0,bx1,by1=map(float,d[:4])
            box=[bx0/scale+x0,by0/scale+y0,bx1/scale+x0,by1/scale+y0]
            output.append((float(d[4]),box))
    return nms(output,frame.shape[1],frame.shape[0])

def pose_crops(frame,boxes):
    height,width=frame.shape[:2]
    batches=[]
    geometries=[]
    for score,b in boxes:
        l,t,r,bot=b
        bw=r-l; bh=bot-t
        crop_w=max(bw,bh*.75)*1.25
        crop_h=crop_w/.75
        cx=(l+r)/2;cy=(t+bot)/2
        # Pixel-center inverse affine: exact full-resolution crop to model input.
        m=np.array([[crop_w/192,0,cx-crop_w/2+crop_w/(2*192)-0.5],
                    [0,crop_h/256,cy-crop_h/2+crop_h/(2*256)-0.5]],np.float32)
        crop=cv2.warpAffine(frame,m,(192,256),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderValue=(0,0,0))
        rgb=crop[:,:,::-1].astype(np.float32)
        batches.append(((rgb-P_MEAN)/P_STD).transpose(2,0,1))
        geometries.append((cx,cy,crop_w,crop_h))
    if batches:return np.ascontiguousarray(np.stack(batches)),geometries
    return None,[]

def poses(session,frame,boxes):
    batch,geom=pose_crops(frame,boxes)
    if batch is None:return [],0
    a=time.perf_counter()
    joints=[]
    for start in range(0,len(batch),8):
        sx,sy=session.run(["simcc_x","simcc_y"],{"input":batch[start:start+8]})
        x=np.argmax(sx,axis=2)/2.0
        y=np.argmax(sy,axis=2)/2.0
        confidence=np.minimum(np.max(sx,axis=2),np.max(sy,axis=2))
        for i,(cx,cy,cw,ch) in enumerate(geom[start:start+8]):
            xx=cx+(x[i]/192-0.5)*cw
            yy=cy+(y[i]/256-0.5)*ch
            c=confidence[i]
            c[((xx<0)|(xx>=frame.shape[1])|(yy<0)|(yy>=frame.shape[0]))]=0
            joints.append(np.stack([xx,yy,c],axis=1))
    return joints,time.perf_counter()-a

def draw(frame,detections,poses,label):
    h,w=frame.shape[:2]
    for i,((score,b),pts) in enumerate(zip(detections,poses)):
        color=PALETTE[i%len(PALETTE)]
        x0,y0,x1,y1=[int(n) for n in b]
        cv2.rectangle(frame,(x0,y0),(x1,y1),color,2)
        visible=pts[:,2]>=0.2
        coords=pts[:,:2].astype(int)
        for q,r in EDGES:
            if visible[q] and visible[r]:
                cv2.line(frame,tuple(coords[q]),tuple(coords[r]),color,2,cv2.LINE_AA)
        for xy,vis in zip(coords,visible):
            if vis:cv2.circle(frame,tuple(xy),3,color,-1,cv2.LINE_AA)
    cv2.rectangle(frame,(0,0),(780,60),(8,8,8),-1)
    cv2.putText(frame,label,(18,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    return frame

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=DEFAULT_SRC)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--samples",type=int,default=200)
    parser.add_argument("--step",type=int,default=12)
    parser.add_argument("--tile-mode",choices=["all","single"],default="all")
    args=parser.parse_args()
    if args.samples<=0 or args.step<=0:raise ValueError("samples/step must be positive")
    args.out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(args.source))
    if not cap.isOpened():raise RuntimeError(f"Can't open original: {args.source}")
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    views=view_rects(width,height) if args.tile_mode=="all" else [(0,0,width,height)]
    assert len(views)==(7 if args.tile_mode=="all" else 1) and width>=1920 and height>=1080, "Native original video should be 1080p; no silent downscale"
    detector_sessions={"RTMDet-Nano":session("RTMDet-Nano-Person.onnx"),"RF-DETR Medium":session("rf-detr-medium.onnx")}
    pose_session=session("RTMPose-M-Body7.onnx")
    writers={}
    results={name:{"frames":0,"proposals":0,"detector_s":0.0,"pose_s":0.0,"wall_s":0.0,
                   "sampled_indices":[],"visible_poses":0} for name in detector_sessions}
    print("NATIVE_SOURCE",width,height,fps,count,"views",views,flush=True)
    for i in range(min(args.samples,(count-1)//args.step+1)):
        idx=i*args.step
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,frame=cap.read()
        if not ok:raise RuntimeError(f"Source decode failed at frame {idx}")
        if frame.shape[:2]!=(height,width):raise RuntimeError(f"Source pixels not original size at frame {idx}")
        for name,s in detector_sessions.items():
            start=time.perf_counter()
            t=time.perf_counter()
            if name=="RTMDet-Nano":objects=rtmdet(s,frame,views)
            else:objects=rfdetr(s,frame,views)
            det_s=time.perf_counter()-t
            points,pose_s=poses(pose_session,frame,objects)
            render=draw(frame.copy(),objects,points,name+" | original "+str(width)+"x"+str(height))
            if name not in writers:
                f=args.out/(name.lower().replace(" ","_").replace("-","_")+"_native.mp4")
                writer=cv2.VideoWriter(str(f),cv2.VideoWriter_fourcc(*"mp4v"),min(30.,fps), (width,height))
                if not writer.isOpened():raise RuntimeError(f"Can't open writer: {f}")
                writers[name]=writer
            writers[name].write(render)
            r=results[name]
            r["frames"]+=1;r["proposals"]+=len(objects)
            r["detector_s"]+=det_s;r["pose_s"]+=pose_s
            r["wall_s"]+=time.perf_counter()-start
            r["visible_poses"]+=sum(int((p[:,2]>=0.2).sum()>=8) for p in points)
            r["sampled_indices"].append(idx)
        if i%25==0:print("NATIVE_PROGRESS",i+1,"/",args.samples,"frame",idx,flush=True)
    cap.release()
    for writer in writers.values():writer.release()
    for name,r in results.items():
        r["detector_fps"]=round(r["frames"]/max(1e-9,r["detector_s"]),3)
        r["full_pose_fps"]=round(r["frames"]/max(1e-9,r["wall_s"]),3)
        r["pose_fps_crops"]=round(r["proposals"]/max(1e-9,r["pose_s"]),3)
        r["source"]=str(args.source)
        r["source_dimensions"]=[width,height]
        r["source_fps"]=fps
        r["native_tiling"]=views
        r["frames_tested"]=len(r.pop("sampled_indices"))
    (args.out/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print("NATIVE_RESULTS",json.dumps(results,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
