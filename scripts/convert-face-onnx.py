#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "numpy", "gguf"]
# ///
"""Convert the OpenCV model-zoo YuNet and SFace ONNX files to gguf-runner face GGUFs.

Usage:
    uv run scripts/convert-face-onnx.py yunet face_detection_yunet_2023mar.onnx face-yunet-2023mar-f32.gguf
    uv run scripts/convert-face-onnx.py sface face_recognition_sface_2021dec.onnx face-sface-2021dec-f32.gguf

The output follows the `face_yunet` / `face_sface` contracts in docs/face-recognition.md:
weights stay float32, batch normalization is folded into the neighbouring convolution or
linear layer, and the graph is written as a flat operation list.
"""

import sys

import gguf
import numpy as np
import onnx
from onnx import numpy_helper

OP_WIDTH = 8
OP_CONV, OP_MAX_POOL, OP_UPSAMPLE, OP_ADD, OP_LINEAR = range(5)
ACT_NONE, ACT_RELU, ACT_PRELU = range(3)

# OpenCV FaceRecognizerSF alignment targets for a 112x112 crop, in detector landmark order:
# right eye, left eye, nose tip, right mouth corner, left mouth corner (subject's view).
SFACE_REFERENCE = [
    38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041,
]


def attrs(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


class Graph:
    """Lowers an ONNX graph to the runner's operation list while folding affine layers."""

    def __init__(self, model, input_name):
        self.init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        self.nodes = list(model.graph.node)
        self.slots = {input_name: 0}
        self.slot_count = 1
        self.ops = []
        # Output name -> op that may still absorb a following batch norm or activation.
        self.open = {}
        # Output name -> per-channel (scale, shift) waiting for the next linear layer.
        self.pending = {}
        self.input_mean = 0.0
        self.input_scale = 1.0

    def new_slot(self, name):
        self.slots[name] = self.slot_count
        self.slot_count += 1
        return self.slots[name]

    def alias(self, new, old):
        self.slots[new] = self.slots[old]
        if old in self.open:
            self.open[new] = self.open.pop(old)
        if old in self.pending:
            self.pending[new] = self.pending.pop(old)

    def scalar(self, name):
        value = self.init[name]
        assert value.size == 1, f"{name} is not a scalar"
        return float(value.reshape(-1)[0])

    def lower(self):
        for node in self.nodes:
            getattr(self, "op_" + node.op_type)(node, attrs(node))

    def op_Sub(self, node, _):
        assert self.slots[node.input[0]] == 0 and not self.ops, "Sub is only supported on the input"
        self.input_mean = self.scalar(node.input[1])
        self.alias(node.output[0], node.input[0])

    def op_Mul(self, node, _):
        assert self.slots[node.input[0]] == 0 and not self.ops, "Mul is only supported on the input"
        self.input_scale = self.scalar(node.input[1])
        self.alias(node.output[0], node.input[0])

    def op_Conv(self, node, a):
        weight = self.init[node.input[1]].astype(np.float32)
        bias = self.init[node.input[2]] if len(node.input) > 2 else np.zeros(weight.shape[0])
        kernel = a["kernel_shape"]
        strides, pads = a.get("strides", [1, 1]), a.get("pads", [0, 0, 0, 0])
        assert kernel[0] == kernel[1] and strides[0] == strides[1], "non-square convolution"
        assert len(set(pads)) == 1 and a.get("dilations", [1, 1]) == [1, 1], "unsupported padding"
        assert node.input[0] not in self.pending, "affine before convolution is not supported"
        op = dict(kind=OP_CONV, a=self.slots[node.input[0]], stride=strides[0], pad=pads[0],
                  groups=a.get("group", 1), act=ACT_NONE, weight=weight,
                  bias=bias.astype(np.float32), slope=None)
        op["out"] = self.new_slot(node.output[0])
        self.ops.append(op)
        self.open[node.output[0]] = op

    def op_BatchNormalization(self, node, a):
        gamma, beta, mean, var = (self.init[n].astype(np.float64) for n in node.input[1:5])
        scale = gamma / np.sqrt(var + a.get("epsilon", 1e-5))
        shift = beta - mean * scale
        op = self.open.get(node.input[0])
        if op is not None and op["act"] == ACT_NONE:
            shape = (-1,) + (1,) * (op["weight"].ndim - 1)
            op["weight"] = (op["weight"].astype(np.float64) * scale.reshape(shape)).astype(np.float32)
            op["bias"] = (op["bias"].astype(np.float64) * scale + shift).astype(np.float32)
        else:
            assert node.input[0] not in self.pending
            self.pending[node.input[0]] = (scale, shift)
        self.alias(node.output[0], node.input[0])

    def activation(self, node, act, slope):
        op = self.open.get(node.input[0])
        assert op is not None and op["kind"] == OP_CONV and op["act"] == ACT_NONE, \
            "activation must directly follow a convolution"
        op["act"], op["slope"] = act, slope
        self.alias(node.output[0], node.input[0])

    def op_Relu(self, node, _):
        self.activation(node, ACT_RELU, None)

    def op_PRelu(self, node, _):
        slope = self.init[node.input[1]].astype(np.float32).reshape(-1)
        self.activation(node, ACT_PRELU, slope)

    def simple(self, node, kind, inputs):
        for name in inputs:
            assert name not in self.pending
        op = dict(kind=kind, a=self.slots[inputs[0]], b=self.slots[inputs[-1]] if len(inputs) > 1 else 0,
                  stride=0, pad=0, groups=0, act=ACT_NONE)
        op["out"] = self.new_slot(node.output[0])
        self.ops.append(op)

    def op_MaxPool(self, node, a):
        assert a["kernel_shape"] == [2, 2] and a["strides"] == [2, 2] and a.get("pads", [0] * 4) == [0] * 4
        self.simple(node, OP_MAX_POOL, [node.input[0]])

    def op_Resize(self, node, a):
        assert a["mode"] in (b"nearest", "nearest")
        assert self.init[node.input[2]].tolist() == [1.0, 1.0, 2.0, 2.0], "only 2x nearest upsampling"
        self.simple(node, OP_UPSAMPLE, [node.input[0]])

    def op_Add(self, node, _):
        self.simple(node, OP_ADD, [node.input[0], node.input[1]])

    def passthrough(self, node, _):
        self.alias(node.output[0], node.input[0])

    # Layout-only or inference-identity nodes. The runner reads head outputs in CHW order and
    # applies the sigmoid during decoding.
    op_Transpose = op_Reshape = op_Sigmoid = op_Dropout = op_Flatten = passthrough

    def op_Gemm(self, node, a):
        assert a.get("transB", 0) == 1 and a.get("alpha", 1.0) == 1.0 and a.get("beta", 1.0) == 1.0
        weight = self.init[node.input[1]].astype(np.float64)
        bias = self.init[node.input[2]].astype(np.float64)
        if node.input[0] in self.pending:
            scale, shift = self.pending.pop(node.input[0])
            per_channel = weight.shape[1] // scale.shape[0]
            bias = bias + weight @ np.repeat(shift, per_channel)
            weight = weight * np.repeat(scale, per_channel)[None, :]
        op = dict(kind=OP_LINEAR, a=self.slots[node.input[0]], stride=0, pad=0, groups=0,
                  act=ACT_NONE, weight=weight.astype(np.float32), bias=bias.astype(np.float32))
        op["out"] = self.new_slot(node.output[0])
        self.ops.append(op)
        self.open[node.output[0]] = op

    def write(self, writer, prefix):
        assert not self.pending, f"unconsumed affine layers: {list(self.pending)}"
        flat, convs, linears = [], 0, 0
        for op in self.ops:
            flat += [op["kind"], op["a"], op.get("b", 0), op["out"], op["stride"], op["pad"],
                     op["groups"], op["act"]]
            if op["kind"] == OP_CONV:
                base = f"face.conv.{convs}"
                convs += 1
                writer.add_tensor(f"{base}.weight", op["weight"])
                writer.add_tensor(f"{base}.bias", op["bias"])
                if op["act"] == ACT_PRELU:
                    writer.add_tensor(f"{base}.slope", op["slope"])
            elif op["kind"] == OP_LINEAR:
                base = f"face.linear.{linears}"
                linears += 1
                writer.add_tensor(f"{base}.weight", op["weight"])
                writer.add_tensor(f"{base}.bias", op["bias"])
        assert len(flat) == OP_WIDTH * len(self.ops)
        writer.add_array(f"{prefix}.graph.ops", flat)
        writer.add_uint32(f"{prefix}.graph.slot_count", self.slot_count)
        writer.add_float32(f"{prefix}.input.mean", self.input_mean)
        writer.add_float32(f"{prefix}.input.scale", self.input_scale)


def convert_yunet(model, writer):
    prefix = "face_yunet"
    graph = Graph(model, "input")
    graph.lower()
    writer.add_name("YuNet 2023mar")
    writer.add_string("general.license", "MIT")
    writer.add_string("general.source.url", "https://huggingface.co/opencv/face_detection_yunet")
    graph.write(writer, prefix)
    writer.add_string(f"{prefix}.input.channel_order", "bgr")
    writer.add_uint32(f"{prefix}.input.max_side", 640)
    writer.add_uint32(f"{prefix}.input.align", 32)
    strides = [8, 16, 32]
    writer.add_array(f"{prefix}.detect.strides", strides)
    for head in ("cls", "obj", "bbox", "kps"):
        writer.add_array(f"{prefix}.detect.{head}_slots", [graph.slots[f"{head}_{s}"] for s in strides])
    writer.add_uint32(f"{prefix}.detect.landmark_count", 5)
    writer.add_float32(f"{prefix}.detect.score_threshold", 0.9)
    writer.add_float32(f"{prefix}.detect.nms_threshold", 0.3)
    writer.add_uint32(f"{prefix}.detect.top_k", 5000)


def convert_sface(model, writer):
    prefix = "face_sface"
    graph = Graph(model, "data")
    graph.lower()
    weight = graph.ops[-1]["weight"]
    writer.add_name("SFace 2021dec")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source.url", "https://huggingface.co/opencv/face_recognition_sface")
    graph.write(writer, prefix)
    writer.add_string(f"{prefix}.input.channel_order", "rgb")
    writer.add_uint32(f"{prefix}.input.size", 112)
    writer.add_array(f"{prefix}.align.reference", SFACE_REFERENCE)
    writer.add_uint32(f"{prefix}.embedding.output_slot", graph.slots["fc1"])
    writer.add_uint32(f"{prefix}.embedding.dimension", int(weight.shape[0]))
    writer.add_float32(f"{prefix}.threshold.cosine", 0.363)


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("yunet", "sface"):
        sys.exit(__doc__)
    kind, source, target = sys.argv[1:]
    model = onnx.load(source)
    writer = gguf.GGUFWriter(target, f"face_{kind}")
    (convert_yunet if kind == "yunet" else convert_sface)(model, writer)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
