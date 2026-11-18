#!/usr/bin/env python3
"""Compare production-shaped RTMDet-Nano (four source-space views)
vs RF-DETR Medium (one full source frame), with the SAME RTMPose-M and
per-joint confidence thresholds. Read every frame at native size/FPS.

Separate experiment only: does not change Studio production code.
"""
from __future__ import annotations
import argparse
import collections
import json
import time
from pathlib import Path

import cv2
import numpy as np
import benchmark_native_resolution as b

THRESHOLDS=(0.20,0.36,0.50)
MIN_VALID_POINTS=8
MAX_PEOPLE=12

def nano_four_views(w,h):
    # Exactly the same relative geometry as person.rs::VIEWS from 640x360
    return [
        (0,0,w,h),
        (0,int(round(h*55/360)),w//2,int(round(h*300/360))),
        (w//2,int(round(h*55/360)),w,int(round(h*300/360))),
        (w//4,int(round(h*40/360)),3*w//4,int(round(h*285/360))),
    ]

def nano_proposals(session, frame, views):
    candidates=[]
    for view_index,(x0,y0,x1,y1) in enumerate(views):
        crop=frame[y0:y1,x0:x1]
        h,w=crop.shape[:2]
        ratio=min(320/w,320/h)
        tw=max(1,min(320,round(w*ratio)))
        th=max(1,min(320,round(h*ratio)))
        scaled=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_LINEAR)
        input_image=np.full((320,320,3),114,dtype=np.uint8)
        input_image[:th,:tw]=scaled
        x=np.ascontiguousarray(((input_image.astype(np.float32)-b.MEAN_BGR)/b.STD_BGR).transpose(2,0,1))[None]
        detections,labels=session.run(["dets","labels"],{"input":x})
        min_score=0.24 if view_index==0 else 0.12
        for d,label in zip(detections[0],labels[0]):
            if int(label)!=0 or not np.isfinite(d).all() or float(d[4])<min_score:
                continue
            xa,ya,xb,yb=map(float,d[:4])
            candidates.append((float(d[4]),[xa/ratio+x0,ya/ratio+y0,xb/ratio+x0,yb/ratio+y0]))
    return b.nms(candidates,frame.shape[1],frame.shape[0])[:MAX_PEOPLE]

def rf_proposals(session,frame):
    # Same architecture and top-down player class (COCO label 1)
    old=b.SCORE_THRESHOLD
    b.SCORE_THRESHOLD=0.28
    try:
        return b.rfdetr(session,frame,[(0,0,frame.shape[1],frame.shape[0])])[:MAX_PEOPLE]
    finally:
        b.SCORE_THRESHOLD=old

def pose_valid(pts):
    good=np.isfinite(pts).all(axis=1)&(pts[:,2]>=0.36)
    torso=good[5] and good[6] and (good[11] or good[12])
    return int(good.sum())>=MIN_VALID_POINTS and torso

def outcome(boxes,poses,w,h):
    entries=[]
    for (det_score,box),points in zip(boxes,poses):
        kp={str(t):int((np.isfinite(points).all(axis=1)&(points[:,2]>=t)).sum()) for t in THRESHOLDS}
        valid=pose_valid(points)
        entries.append({"box":list(map(float,box)),"score":float(det_score),
                        "kp":kp,"valid":bool(valid),
                        "points":points})
    return entries

def metric_container():
    return {"frames":0,"roi_candidates":0,"confirmed_poses":0,
            "kp_raw_sum":{str(t):0 for t in THRESHOLDS},
            "kp_confirmed_sum":{str(t):0 for t in THRESHOLDS},
            "full_17_confirmed":0, "poses_ge_12":0,
            "poses_ge_8":0,
            "high_confidence_wrists":0,
            "high_confidence_ankles":0,
            "invalid_bbox_candidates":0,
            "far_confirmed":0,
            "near_confirmed":0,
            "detector_s":0.0,"pose_s":0.0,"drawing_encoding_s":0.0,
            "processing_wall_s":0.0,
            "frames_no_confirmed_players":0,
            "per_frame_poses":[],"per_frame_kp":[]}

def track_quality(stats,entries,frame_h):
    stats["roi_candidates"]+=len(entries)
    confirmed=[p for p in entries if p["valid"]]
    stats["confirmed_poses"]+=len(confirmed)
    stats["invalid_bbox_candidates"]+=len(entries)-len(confirmed)
    if not confirmed:stats["frames_no_confirmed_players"]+=1
    frame_points=0
    for p in entries:
        for t in THRESHOLDS:stats["kp_raw_sum"][str(t)]+=p["kp"][str(t)]
    for p in confirmed:
        for t in THRESHOLDS:stats["kp_confirmed_sum"][str(t)]+=p["kp"][str(t)]
        n=p["kp"]["0.36"]
        frame_points+=n
        stats["full_17_confirmed"]+=int(n==17)
        stats["poses_ge_12"]+=int(n>=12)
        stats["poses_ge_8"]+=int(n>=8)
        joints=p["points"]
        stats["high_confidence_wrists"]+=int(joints[9,2]>=0.36)+int(joints[10,2]>=0.36)
        stats["high_confidence_ankles"]+=int(joints[15,2]>=0.36)+int(joints[16,2]>=0.36)
        center_y=(p["box"][1]+p["box"][3])*0.5
        key="far_confirmed" if center_y < frame_h*0.51 else "near_confirmed"
        stats[key]+=1
    stats["per_frame_poses"].append(len(confirmed))
    stats["per_frame_kp"].append(frame_points)
    return confirmed

def annotate(frame,entries,name,original_idx,fps):
    for i,p in enumerate(entries):
        color=b.PALETTE[i%len(b.PALETTE)] if p["valid"] else (125,125,125)
        x0,y0,x1,y1=map(int,p["box"])
        cv2.rectangle(frame,(x0,y0),(x1,y1),color,2 if p["valid"] else 1)
        pts=p["points"]
        good=(np.isfinite(pts).all(axis=1)&(pts[:,2]>=0.36))
        points=pts[:,:2].astype(int)
        for k,l in b.EDGES:
            if good[k] and good[l]:
                cv2.line(frame,tuple(points[k]),tuple(points[l]),color,2,cv2.LINE_AA)
        for j,vis in enumerate(good):
            if vis:cv2.circle(frame,tuple(points[j]),3,color,-1,cv2.LINE_AA)
        title=f'{p["kp"]["0.36"]}/17'+('' if p["valid"] else '?')
        cv2.putText(frame,title,(max(0,x0),max(70,y0-8)),cv2.FONT_HERSHEY_SIMPLEX,.53,color,2,cv2.LINE_AA)
    cv2.rectangle(frame,(0,0),(950,60),(5,5,5),-1)
    cv2.putText(frame,f'{name} | {original_idx}/{fps:.2f}s | 1920x1080 original',(12,40),
                cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2,cv2.LINE_AA)
    return frame

def match_poses(a_entries,b_entries):
    a=[x for x in a_entries if x["valid"]]
    bb=[x for x in b_entries if x["valid"]]
    proposals=[]
    for i,x in enumerate(a):
        for j,y in enumerate(bb):
            iou=b.iou(np.array(x["box"]),np.array(y["box"]))
            if iou>=0.35:proposals.append((iou,i,j))
    used_a=set(); used_b=set();pairs=[]
    for score,i,j in sorted(proposals,reverse=True):
        if i in used_a or j in used_b:continue
        used_a.add(i);used_b.add(j)
        pairs.append([a[i]["kp"]["0.36"],bb[j]["kp"]["0.36"],round(float(score),3)])
    return pairs,len(a)-len(used_a),len(bb)-len(used_b)

def slim(stats):
    pts=stats.pop("per_frame_kp")
    poses=stats.pop("per_frame_poses")
    frames=stats["frames"]
    conf=stats["confirmed_poses"]
    stats["avg_confirmed_poses_per_frame"]=round(conf/frames,3)
    stats["avg_points_036_per_frame"]=round(stats["kp_confirmed_sum"]["0.36"]/frames,3)
    stats["avg_points_036_per_confirmed_pose"]=round(stats["kp_confirmed_sum"]["0.36"]/max(1,conf),3)
    stats["percent_poses_all17_036"]=round(100*stats["full_17_confirmed"]/max(1,conf),2)
    stats["percent_poses_12plus_036"]=round(100*stats["poses_ge_12"]/max(1,conf),2)
    stats["detector_fps"]=round(frames/max(0.001,stats["detector_s"]),2)
    stats["pose_and_detector_fps"]=round(frames/max(0.001,stats["processing_wall_s"]),2)
    stats["model_pipeline_fps_without_encoding"]=round(frames/max(0.001,stats["detector_s"]+stats["pose_s"]),2)
    stats["median_poses_per_frame"]=float(np.median(poses))
    stats["frames_with_4plus_poses"]=sum(n>=4 for n in poses)
    stats["frames_with_5plus_poses"]=sum(n>=5 for n in poses)
    stats["median_keypoints_per_frame"]=float(np.median(pts))
    return stats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=b.DEFAULT_SRC)
    ap.add_argument("--out",type=Path,default=b.ROOT/"results"/"detector_pose_points_full_20261115")
    ap.add_argument("--frames",type=int,default=0,help="0 means all selected frames; optional for smoke test")
    ap.add_argument("--sample-stride",type=int,default=1,help="Read native-source frames every N source indices; 1 = full video")
    ap.add_argument("--video",action=argparse.BooleanOptionalAction,default=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(args.source))
    if not cap.isOpened():raise RuntimeError(f"Cannot open source video: {args.source}")
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (width,height)!=(1920,1080) or abs(fps-60)>0.01:
        raise RuntimeError(f"Unexpected source dimensions or fps: {width}x{height}, {fps}")
    if args.sample_stride<=0:raise RuntimeError("sample stride must be positive")
    sampled_count=(frame_count+args.sample_stride-1)//args.sample_stride
    total=min(sampled_count,args.frames) if args.frames else sampled_count
    model_names=["RTMDet-Nano 4views","RF-DETR Medium 1view"]
    engines={"RTMDet-Nano 4views":b.session("RTMDet-Nano-Person.onnx"),
             "RF-DETR Medium 1view":b.session("rf-detr-medium.onnx")}
    pose_engine=b.session("RTMPose-M-Body7.onnx")
    views=nano_four_views(width,height)
    outputs={}
    summaries={name:metric_container() for name in model_names}
    pairs={"count":0,"nano_point_wins":0,"rf_point_wins":0,"ties":0,
           "nano_matched_points":0,"rf_matched_points":0,
           "nano_only_confirmed":0,"rf_only_confirmed":0}
    decode_s=0.0
    start=time.perf_counter()
    for sample_index in range(total):
        idx=sample_index*args.sample_stride
        t=time.perf_counter()
        if args.sample_stride!=1:cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,frame=cap.read();decode_s+=time.perf_counter()-t
        if not ok:raise RuntimeError(f"Unable to decode frame {idx}")
        if frame.shape[:2]!=(height,width):raise RuntimeError(f"Frame {idx} was silently resized")
        # Alternate order to mitigate a first-mover CUDA cache bias
        order=model_names if sample_index%2==0 else list(reversed(model_names))
        frame_entries={}
        for name in order:
            eng=engines[name]
            stat=summaries[name]
            tstart=time.perf_counter()
            t=time.perf_counter()
            rects=nano_proposals(eng,frame,views) if name.startswith("RTMDet") else rf_proposals(eng,frame)
            stat["detector_s"]+=time.perf_counter()-t
            t=time.perf_counter()
            pose_result,_=b.poses(pose_engine,frame,rects)
            stat["pose_s"]+=time.perf_counter()-t
            entries=outcome(rects,pose_result,width,height)
            track_quality(stat,entries,height)
            stat["frames"]+=1
            frame_entries[name]=entries
            if args.video:
                t=time.perf_counter()
                label=name
                annotated=annotate(frame.copy(),entries,label,idx,fps)
                if name not in outputs:
                    output_name="rtmdet_nano_4views.mp4" if name.startswith("RTMDet") else "rfdetr_medium_1view.mp4"
                    video_path=args.out/output_name
                    writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),fps/args.sample_stride,(width,height))
                    if not writer.isOpened():raise RuntimeError(f"VideoWriter failure: {video_path}")
                    outputs[name]=writer
                outputs[name].write(annotated)
                stat["drawing_encoding_s"]+=time.perf_counter()-t
            stat["processing_wall_s"]+=time.perf_counter()-tstart
        matched,na,rf=match_poses(frame_entries[model_names[0]],frame_entries[model_names[1]])
        pairs["count"]+=len(matched)
        pairs["nano_only_confirmed"]+=na
        pairs["rf_only_confirmed"]+=rf
        for n,r,iou in matched:
            pairs["nano_matched_points"]+=n
            pairs["rf_matched_points"]+=r
            pairs["nano_point_wins"]+=int(n>r)
            pairs["rf_point_wins"]+=int(r>n)
            pairs["ties"]+=int(n==r)
        if sample_index%50==0 or sample_index==total-1:
            status={x:{"confirmed_poses":summaries[x]["confirmed_poses"],"keypoints":summaries[x]["kp_confirmed_sum"]["0.36"]} for x in model_names}
            print("PROGRESS",sample_index+1,total,"source_frame",idx,round(time.perf_counter()-start,2),json.dumps(status),flush=True)
    cap.release()
    for out in outputs.values():out.release()
    report={"source":str(args.source),"source_resolution":[width,height],"source_fps":fps,
            "source_frames":frame_count,"processed_frames":total,"sampling_step_original_frames":args.sample_stride,"view_rectangles":views,
            "counts_only_high_confidence":True,"confidence_thresholds":THRESHOLDS,
            "pose_validity":"8+ joints confidence>=0.36, shoulders(5,6) and hip(11 or 12)",
            "nms_iou":0.55,"matched_iou":0.35,"decoding_s_shared":round(decode_s,3),
            "total_wall_s":round(time.perf_counter()-start,3),
            "paired_same_player":pairs,
            "stats":{n:slim(stats) for n,stats in summaries.items()},
            "caveats":["Total joint count is a proxy, not mAP nor precision/recall",
                       "Model-specific confidence thresholds Nano global .24/tile .12; RFDETR .28",
                       "RTMDet native original 4 views; production 640x360 4 views is different",
                       "No track ID interpolation or virtual skeletons",
                       "Model-only FPS excludes shared source decoding and output encoding"]}
    (args.out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("DONE",json.dumps({"frames":total,"pairing":pairs,"summary":{n:{"detector_fps":d["detector_fps"],"pose_and_detector_fps":d["pose_and_detector_fps"],
        "confirmed_poses":d["confirmed_poses"],"kp_036":d["kp_confirmed_sum"]["0.36"],
        "avg_points_036_per_pose":d["avg_points_036_per_confirmed_pose"]} for n,d in report["stats"].items()}},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
