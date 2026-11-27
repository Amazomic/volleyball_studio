#!/usr/bin/env python3
"""Reproducible RF-DETR Medium + top-down pose comparison on Volley project videos.

Does not modify the production inference pipeline. Dependencies: numpy, opencv-python,
onnxruntime-gpu. Run in WSL GPU venv. Shared detector proposals are cached on disk
so the pose models see exactly the same bounding boxes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "pose_model_benchmark_20261115"
POSE_H, POSE_W = 256, 192
DET_H = DET_W = 576
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.229], dtype=np.float32).reshape(1, 1, 3)
POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
JOINT_EDGES = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]
PALETTE = [(40,225,50),(255,150,30),(80,130,255),(240,70,210),
           (10,235,225),(160,70,255),(255,160,180),(100,180,100)]

def model_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def get_session(path, prefer="cuda"):
    providers=ort.get_available_providers()
    if prefer=="cuda" and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDA execution provider missing; cannot silently fall back to CPU")
    ep=["CUDAExecutionProvider","CPUExecutionProvider"] if prefer=="cuda" else ["CPUExecutionProvider"]
    opts=ort.SessionOptions()
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    s=ort.InferenceSession(str(path),sess_options=opts,providers=ep)
    if prefer=="cuda" and s.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA not active for {path}: {s.get_providers()}")
    print("MODEL_LOADED",str(path),"providers",s.get_providers(),
          "input",[(x.name,x.shape) for x in s.get_inputs()],
          "output",[(x.name,x.shape) for x in s.get_outputs()],flush=True)
    return s

def detect_frame(session, frame):
    # Same full-image 640x360 -> direct 576x576 RGB geometry used in person.rs
    h,w=frame.shape[:2]
    low=cv2.resize(frame,(640,360),interpolation=cv2.INTER_LINEAR)
    img=cv2.cvtColor(cv2.resize(low,(DET_W,DET_H)),cv2.COLOR_BGR2RGB)
    x=np.transpose((img.astype(np.float32)/255.-MEAN)/STD,(2,0,1))[None]
    output=session.run(["pred_boxes","pred_logits"],{"input":np.ascontiguousarray(x)})
    bb,raw=output
    scores=1/(1+np.exp(-np.clip(raw[0,:,1],-40,40)))
    x0=bb[0,:,0]-bb[0,:,2]/2
    y0=bb[0,:,1]-bb[0,:,3]/2
    x1=bb[0,:,0]+bb[0,:,2]/2
    y1=bb[0,:,1]+bb[0,:,3]/2
    boxes=np.clip(np.stack([x0,y0,x1,y1],axis=1),0.,1.)
    mask=(scores>=0.35)&np.isfinite(scores)&np.isfinite(boxes).all(axis=1)&(
        (boxes[:,2]-boxes[:,0])*640>=8)&((boxes[:,3]-boxes[:,1])*360>=20)
    boxes=boxes[mask]
    scores=scores[mask]
    order=np.argsort(-scores)
    kept=[]
    for idx in order:
        b=boxes[idx]
        overlap=False
        for k in kept:
            q=boxes[k]
            wh=np.maximum(0,np.minimum(b[2:],q[2:])-np.maximum(b[:2],q[:2]))
            intersection=wh[0]*wh[1]
            area=(b[2]-b[0])*(b[3]-b[1])
            qb=(q[2]-q[0])*(q[3]-q[1])
            iou=intersection/max(1e-8,area+qb-intersection)
            if iou>0.55:
                overlap=True
                break
        if not overlap:
            kept.append(idx)
    return [{"box":boxes[i].round(7).tolist(),"score":round(float(scores[i]),6)}
            for i in kept[:16]]

def geometry(box, shape):
    height,width=shape[:2]
    l,t,r,b=box
    bw=(r-l)*width
    bh=(b-t)*height
    crop_w=max(bw,bh*0.75)*1.25
    crop_h=crop_w/0.75
    cx=(l+r)*width/2
    cy=(t+b)*height/2
    return cx,cy,crop_w,crop_h

def pose_input(frame,box):
    cx,cy,cw,ch=geometry(box,frame.shape)
    # Sample source to pixel-center aligned 192x256 human crop.
    a=np.array([[cw/POSE_W,0,cx-cw/2+cw/(2*POSE_W)-0.5],
                [0,ch/POSE_H,cy-ch/2+ch/(2*POSE_H)-0.5]],np.float32)
    crop=cv2.warpAffine(frame,a,(POSE_W,POSE_H),
        flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    rgb=crop[:,:,::-1].astype(np.float32)
    data=((rgb-POSE_MEAN)/POSE_STD).transpose(2,0,1)
    return np.ascontiguousarray(data), (cx,cy,cw,ch)

def decode_pose(outputs, geometries, source_shape, typ):
    h,w=source_shape[:2]
    n=len(geometries)
    if typ.startswith("rtmpose"):
        sx,sy=outputs
        sx=np.asarray(sx); sy=np.asarray(sy)
        xi=np.argmax(sx,axis=2).astype(np.float32)/2
        yi=np.argmax(sy,axis=2).astype(np.float32)/2
        conf=np.minimum(np.max(sx,axis=2),np.max(sy,axis=2))
    else:
        a=np.asarray(outputs[0])
        if a.ndim!=4 or a.shape[0]!=n:
            raise RuntimeError(f"ViTPose expected [N,17,H,W] heatmaps, got {a.shape}")
        if a.shape[1]!=17 and a.shape[-1]==17:
            a=a.transpose(0,3,1,2)
        heat_h,heat_w=a.shape[-2:]
        flat=a.reshape(n,17,-1)
        arg=np.argmax(flat,axis=-1)
        yi=(arg//heat_w).astype(np.float32)
        xi=(arg%heat_w).astype(np.float32)
        conf=np.max(flat,axis=-1)
        # Standard heatmap refinement: local gradient selects +/-0.25 peak offset.
        for i in range(n):
            for j in range(17):
                x,y=int(xi[i,j]),int(yi[i,j])
                if 1<=x<heat_w-1 and 1<=y<heat_h-1:
                    dx=a[i,j,y,x+1]-a[i,j,y,x-1]
                    dy=a[i,j,y+1,x]-a[i,j,y-1,x]
                    xi[i,j]+=np.sign(dx)*0.25
                    yi[i,j]+=np.sign(dy)*0.25
        xi=(xi+0.5)*POSE_W/heat_w
        yi=(yi+0.5)*POSE_H/heat_h
    poses=[]
    for i,(cx,cy,cw,ch) in enumerate(geometries):
        xx=cx+(xi[i]/POSE_W-0.5)*cw
        yy=cy+(yi[i]/POSE_H-0.5)*ch
        valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        # ViTPose heatmap scores may not be calibrated on exactly the same scale as RTMPose.
        cc=np.maximum(0,conf[i])*valid
        pts=np.stack([xx,yy,cc],axis=-1)
        poses.append(pts)
    return poses

def detect(args):
    output=args.out
    output.mkdir(parents=True,exist_ok=True)
    s=get_session(ROOT/"models"/"rf-detr-medium.onnx",args.device)
    cap=cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError("Cannot read source video")
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    count=min(total-args.start,args.frames or total)
    cap.set(cv2.CAP_PROP_POS_FRAMES,args.start)
    frames=[]
    t0=time.perf_counter()
    inference=0.
    for i in range(count):
        ok,frame=cap.read()
        if not ok:
            break
        a=time.perf_counter()
        boxes=detect_frame(s,frame)
        inference+=time.perf_counter()-a
        frames.append(boxes)
        if i%100==0:
            print("DETECT",i+1,"/",count,"boxes",len(boxes),flush=True)
    elapsed=time.perf_counter()-t0
    output_file=output/"shared_detections.json"
    body={"source":str(args.source),"start_frame":args.start,"source_fps":fps,
          "source_size":[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
          "sample_frames":len(frames),"detector":"rf-detr-medium.onnx",
          "detector_sha256":model_sha(ROOT/"models"/"rf-detr-medium.onnx"),
          "runtime_provider":s.get_providers()[0],
          "detector_inference_fps":len(frames)/inference if inference else None,
          "detector_pipeline_fps":len(frames)/elapsed if elapsed else None,
          "detector_wall_s":elapsed,"total_proposals":sum(map(len,frames)),"boxes":frames}
    output_file.write_text(json.dumps(body,ensure_ascii=False),encoding="utf-8")
    print("DETECTION_DONE",output_file,"pipeline_fps",body["detector_pipeline_fps"],
          "inference_fps",body["detector_inference_fps"],"proposals",body["total_proposals"],flush=True)

def annotate(frame,poses,model_name,threshold=0.20):
    h,w=frame.shape[:2]
    for i,points in enumerate(poses):
        color=PALETTE[i%len(PALETTE)]
        joints=points[:,:2].astype(np.int32)
        visible=points[:,2]>=threshold
        for p,q in JOINT_EDGES:
            if visible[p] and visible[q]:
                cv2.line(frame,tuple(joints[p]),tuple(joints[q]),color,max(2,int(w/640)),cv2.LINE_AA)
        for j,(pt,v) in enumerate(zip(joints,visible)):
            if v:
                cv2.circle(frame,tuple(pt),max(3,int(w/450)),color,-1,cv2.LINE_AA)
                cv2.circle(frame,tuple(pt),max(3,int(w/450)),(20,20,20),1,cv2.LINE_AA)
    cv2.rectangle(frame,(10,10),(650,66),(15,15,15),-1)
    cv2.putText(frame,model_name+" | 17 keypoints | RF-DETR Medium",(20,49),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(250,250,250),2,cv2.LINE_AA)

def pose(args):
    output=args.out
    if args.model=="rtmpose":
        path=ROOT/"models"/"RTMPose-M-Body7.onnx"
    elif args.model=="rtmpose_l384":
        path=Path("/home/amazomic/.cache/volley-pose-benchmark/models/rtmpose-l384.onnx")
    else:
        path=Path(args.vitpose)
    global POSE_H, POSE_W
    if args.model=="rtmpose_l384":
        POSE_H, POSE_W = 384, 288
    if not path.is_file():
        raise RuntimeError(f"Model file not found: {path}")
    meta=json.loads((output/"shared_detections.json").read_text(encoding="utf-8"))
    if Path(meta["source"]).resolve()!=args.source.resolve():
        raise RuntimeError("Source mismatch between detection and pose")
    if meta["start_frame"]!=args.start:
        raise RuntimeError("Start frame mismatch")
    s=get_session(path,args.device)
    cap=cv2.VideoCapture(str(args.source))
    cap.set(cv2.CAP_PROP_POS_FRAMES,args.start)
    fps=meta["source_fps"]
    total=len(meta["boxes"])
    writer=None
    inference=preprocess=decode=0.
    crops_count=0
    reliable=0
    pose_points_count=0
    t0=time.perf_counter()
    model=args.model
    log=[]
    for i,boxes in enumerate(meta["boxes"]):
        ok,frame=cap.read()
        if not ok:
            raise RuntimeError("Video terminated before detection records")
        if writer is None:
            height,width=frame.shape[:2]
            target=output/(model+"_keypoints.mp4")
            writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter could not open MP4 output")
        t=time.perf_counter()
        crops=[]
        gs=[]
        for person in boxes:
            crop,g=pose_input(frame,person["box"])
            crops.append(crop)
            gs.append(g)
        preprocess+=time.perf_counter()-t
        poses=[]
        if crops:
            x=np.stack(crops)
            # Fixed-batch models may require single-item runs; record which path was used.
            t=time.perf_counter()
            try:
                o=s.run(None,{s.get_inputs()[0].name:x})
                inference+=time.perf_counter()-t
                t=time.perf_counter()
                poses=decode_pose(o,gs,frame.shape,model)
                decode+=time.perf_counter()-t
            except Exception as exc:
                if len(crops)==1:
                    raise
                if i==0:
                    print("BATCH_FALLBACK",str(exc)[:300],flush=True)
                for crop,g in zip(crops,gs):
                    t=time.perf_counter()
                    o=s.run(None,{s.get_inputs()[0].name:crop[None]})
                    inference+=time.perf_counter()-t
                    t=time.perf_counter()
                    poses.extend(decode_pose(o,[g],frame.shape,model))
                    decode+=time.perf_counter()-t
            crops_count+=len(crops)
            for p in poses:
                reliable+=int((p[:,2]>=0.2).sum()>=8)
                pose_points_count+=int((p[:,2]>=0.2).sum())
        annotate(frame,poses,{"rtmpose":"RTMPose-M","rtmpose_l384":"RTMPose-L (384x288)","vitpose":"ViTPose-L"}[model])
        writer.write(frame)
        if i%100==0:
            print("POSE",model,i+1,"/",total,"persons",len(crops),flush=True)
    wall=time.perf_counter()-t0
    if writer:
        writer.release()
    cap.release()
    result={
        "model":model,"checkpoint":str(path),"checkpoint_sha256":model_sha(path),
        "source":str(args.source),"source_frames":total,"source_duration_s":total/fps,
        "source_resolution":meta["source_size"],"device_requested":args.device,
        "provider":s.get_providers()[0],"engine":"ONNX Runtime "+ort.__version__,
        "gpu":platform.platform(),
        "person_crops":crops_count,"reliable_person_poses_8points":reliable,
        "visible_keypoints_0.2":pose_points_count,"processing_wall_s":round(wall,3),
        "total_pipeline_fps_excluding_shared_detector":round(total/wall,3),
        "pose_model_fps_person_crops":round(crops_count/inference,3) if inference else None,
        "pose_inference_ms_per_person":round(inference/crops_count*1000,3) if crops_count else None,
        "preprocess_s":round(preprocess,3),"pose_inference_s":round(inference,3),
        "postprocess_s":round(decode,3),
        "detector_pipeline_fps_shared":meta["detector_pipeline_fps"],
        "note":"Per-frame un-interpolated pose; detector is shared and run on every frame. "
               "GPU video decode/encode not counted as model inference. "
               "No labeled ground truth; keypoint coverage is not accuracy/mAP.",
        "video":str(target)
    }
    (output/(model+"_report.json")).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("POSE_DONE",json.dumps(result,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["detect","rtmpose","rtmpose_l384","vitpose"],required=True)
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=OUT_DIR)
    ap.add_argument("--vitpose",default="/home/amazomic/.cache/volley-pose-benchmark/models/vitpose-l.onnx")
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--start",type=int,default=0)
    ap.add_argument("--frames",type=int,default=300,
                    help="Number of consecutive source frames to include; 0 means full source.")
    args=ap.parse_args()
    if args.stage=="detect":
        detect(args)
    else:
        args.model=args.stage
        pose(args)
if __name__=="__main__":
    main()
