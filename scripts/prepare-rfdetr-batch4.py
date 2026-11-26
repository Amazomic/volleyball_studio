from pathlib import Path
import hashlib, onnx, numpy as np, copy, sys, os
from onnx import numpy_helper as nh, helper as H
SRC=Path(sys.argv[1])
TARGET=Path(sys.argv[2])
EXPECTED="9055ef9b1b76b79412069590654baf913cd173972be4ff4686c1535bb021e1f6"
if hashlib.sha256(SRC.read_bytes()).hexdigest()!="18250b5a3a9552ae117d592f8e8c27fe52438d70c914a77a8913fc45d8f29e69":
    raise RuntimeError("RF-DETR source model SHA-256 mismatch")
if TARGET.exists() and hashlib.sha256(TARGET.read_bytes()).hexdigest()==EXPECTED:
    sys.exit(0)
m=onnx.shape_inference.infer_shapes(onnx.load(SRC))
sh={v.name:tuple(d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim) for v in [*m.graph.input,*m.graph.output,*m.graph.value_info]}
const={t.name:nh.to_array(t) for t in m.graph.initializer}
def build(b):
    g=copy.deepcopy(m)
    del g.graph.value_info[:]
    for x in [*g.graph.input,*g.graph.output]:
        x.type.tensor_type.shape.dim[0].dim_value=b
    nodes=list(g.graph.node)
    del g.graph.node[:]
    def init(name, value):
        g.graph.initializer.append(nh.from_array(np.asarray(value,dtype=np.int64),name))
        return name
    for i,n in enumerate(nodes):
        if i==3:
            cls=n.input[0]
            out="batch_class_tokens"
            g.graph.node.append(H.make_node("Tile",[cls,init("cls_repeat",[b,1,1])],[out]))
            n.input[0]=out
        if i==10:
            src=n.input[0];out=n.output[0]
            g.graph.node.extend([
                H.make_node("Unsqueeze",[src,init("cls_axis",[1])],["cls_unsqueezed"]),
                H.make_node("Tile",["cls_unsqueezed",init("window_repeat",[1,4,1,1])],["cls_tiled"]),
                H.make_node("Reshape",["cls_tiled",init("cls_shape",[b*4,1,384])],[out])])
            continue
        if i in [8,441,447,453,459]:
            for a in n.attribute:
                if a.name=="perm":
                    vals=[0]+[v+1 for v in a.ints]
                    del a.ints[:];a.ints.extend(vals)
        if n.op_type=="Reshape":
            dims=list(sh[n.output[0]])
            assert all(isinstance(x,int) and x>0 for x in dims),(i,dims)
            if i in [7,440,446,452,458]:
                dims=[b]+dims
            elif "/self_attn/" in n.name and dims[0]==300 and len(dims)==3:
                dims[1]*=b
            else:
                dims[0]*=b
            n.input[1]=init("batch_shape_"+str(i),dims)
        g.graph.node.append(n)
    g=onnx.shape_inference.infer_shapes(g,strict_mode=True)
    onnx.checker.check_model(g)
    path=TARGET.with_name(TARGET.name+".tmp-"+str(os.getpid()))
    try:
        onnx.save(g,path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED:
            raise RuntimeError("RF-DETR Batch 4 export SHA-256 differs from verified model")
        os.replace(path,TARGET)
    finally:
        if path.exists():
            path.unlink()
    print(TARGET, flush=True)
build(4)
