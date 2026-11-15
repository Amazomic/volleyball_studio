import sys
import onnx

model = onnx.load(sys.argv[1])
nodes = model.graph.node
for index, node in enumerate(nodes):
    if node.name == "Where_557":
        for nearby in nodes[index + 65:]:
            print(nearby.name, nearby.op_type, "inputs", list(nearby.input), "outputs", list(nearby.output))
            for attr in nearby.attribute:
                if attr.name != "value":
                    print(" ", attr)
        break
else:
    raise ValueError("Where_557 not found")
