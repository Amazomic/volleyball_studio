import sys
import onnx

model = onnx.load(sys.argv[1])
nodes = model.graph.node
for index, node in enumerate(nodes):
    if node.name == "Where_557":
        original = node.input[0]
        expanded = original + "_batch_axis"
        nodes.insert(index, onnx.helper.make_node("Unsqueeze", [original], [expanded], name="BatchProbeUnsqueeze", axes=[-1]))
        nodes[index + 1].input[0] = expanded
        break
else:
    raise ValueError("Where_557 not found")
onnx.checker.check_model(model)
onnx.save(model, sys.argv[2])
