#!/usr/bin/env python3
"""Offline TCN trainer; not a Python Studio runtime.

JSONL: {"match_id":"match1","width":1920,"height":1080,
"ball_xy":[[x_px,y_px],null,...],"contacts":[zero_based_local_frame,...]}.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

def load_rallies(path):
    rallies = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        w, h = int(row["width"]), int(row["height"])
        xy = row["ball_xy"]
        assert 64 <= w <= 16384 and 64 <= h <= 16384 and len(xy) >= 17, lineno
        contacts = {int(i) for i in row["contacts"]}
        assert all(0 <= i < len(xy) for i in contacts), lineno
        features, previous = [], None
        for v in xy:
            if v is None:
                features.append([0.0] * 4)
                previous = None
                continue
            x, y = float(v[0])/w, float(v[1])/h
            assert math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1
            dx, dy = (x-previous[0], y-previous[1]) if previous is not None else (0.0, 0.0)
            features.append([x,y,dx,dy])
            previous = (x,y)
        labels = [0.0] * len(xy)
        for center in contacts:
            for d in range(-3,4):
                i = center + d
                if 0 <= i < len(xy):
                    labels[i] = max(labels[i], math.exp(-0.5*d*d))
        rallies.append((str(row["match_id"]),features,labels))
    return rallies

def windows(rallies):
    out = []
    for group, features, labels in rallies:
        last = len(features) - 17
        starts = list(range(0,last+1,3))
        if starts[-1] != last:
            starts.append(last)
        out += [(group,features[s:s+17],labels[s:s+17]) for s in starts]
    return out

def main():
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Block(nn.Module):
        def __init__(self, ci, co, dilation):
            super().__init__()
            self.a=nn.Conv1d(ci,co,5,padding=2*dilation,dilation=dilation)
            self.a_bn=nn.BatchNorm1d(co)
            self.b=nn.Conv1d(co,co,5,padding=2*dilation,dilation=dilation)
            self.b_bn=nn.BatchNorm1d(co)
            self.skip=nn.Identity() if ci==co else nn.Conv1d(ci,co,1)
            self.drop=nn.Dropout(0.5)
        def forward(self,x):
            z=self.drop(F.relu(self.a_bn(self.a(x))))
            z=self.drop(F.relu(self.b_bn(self.b(z))))
            return F.relu(z+self.skip(x))

    class Tcn(nn.Module):
        def __init__(self):
            super().__init__()
            self.entry=nn.Sequential(nn.Conv1d(4,128,5,padding=2),
                nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.5))
            self.blocks=nn.Sequential(Block(128,128,2),Block(128,256,4))
            self.head=nn.Sequential(nn.Conv1d(256,1,1),nn.Sigmoid())
        def forward(self,x):
            return self.head(self.blocks(self.entry(x)))

    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("models/volley3d_touch_tcn.onnx"))
    parser.add_argument("--epochs",type=int,default=60)
    parser.add_argument("--seed",type=int,default=26)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"Refusing overwrite: {args.out} (pass --overwrite)")
    if not 1 <= args.epochs <= 1000:
        parser.error("epochs must be 1..1000")
    rng=random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rallies=load_rallies(args.dataset)
    ids=sorted({r[0] for r in rallies})
    if len(ids)<2:
        parser.error("Provide at least two independent match_id groups")
    rng.shuffle(ids)
    val_ids=set(ids[:max(1,round(len(ids)*0.2))])
    train=windows([r for r in rallies if r[0] not in val_ids])
    valid=windows([r for r in rallies if r[0] in val_ids])
    if not train or not valid:
        parser.error("No validation or training windows")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train={len(train)} valid={len(valid)} heldout={sorted(val_ids)} device={device}")
    model=Tcn().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)

    def batch(rows,augment=False):
        x=np.array([r[1] for r in rows],dtype="float32")
        y=np.array([r[2] for r in rows],dtype="float32")
        if augment:
            for item in x:
                if rng.random()<0.5:
                    item[:,0]=1-item[:,0]
                    item[:,2]=-item[:,2]
                if rng.random()<0.5:
                    item[:,:2]+=rng.uniform(-0.08,0.08)
                if rng.random()<0.5:
                    item[:,:2]+=np.random.normal(0,0.006,(17,2))
                if rng.random()<0.3:
                    n=rng.randint(1,2)
                    k=rng.randrange(18-n)
                    item[k:k+n]=0
        return (torch.tensor(x.transpose(0,2,1),device=device),
                torch.tensor(y[:,None,:],device=device))
    def focal(p,y):
        p=p.clamp(1e-6,1-1e-6)
        pt=p*y+(1-p)*(1-y)
        alpha=0.25*y+0.75*(1-y)
        return (alpha*(1-pt).square()*F.binary_cross_entropy(p,y,reduction="none")).mean()
    best=float("inf")
    weights=None
    for epoch in range(args.epochs):
        rng.shuffle(train)
        model.train()
        for i in range(0,len(train),64):
            x,y=batch(train[i:i+64],True)
            optimizer.zero_grad(set_to_none=True)
            loss=focal(model(x),y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val=sum(focal(model(x),y).item()*x.shape[0]
                for j in range(0,len(valid),64)
                for x,y in [batch(valid[j:j+64])])/len(valid)
        print(f"epoch {epoch+1}/{args.epochs} validation_loss={val:.6f}",flush=True)
        if val<best:
            best=val
            weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(weights)
    model.cpu().eval()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,torch.zeros(1,4,17),str(args.out),
        input_names=["trajectory"],output_names=["probabilities"],
        opset_version=17,do_constant_folding=True,dynamo=False)
    print(f"exported={args.out} sha256={hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    print("Locally trained checkpoint; not the paper authors' released weights.")

if __name__=="__main__":
    main()
