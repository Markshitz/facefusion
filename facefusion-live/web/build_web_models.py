"""Build the compact model bundle preloaded by the web page.

Weight-only int8: every large float weight is stored as int8 + a per-channel fp32 scale and restored by a
DequantizeLinear node, which ONNX Runtime constant-folds at load (with QDQ fusion disabled). Inference then runs
in fp32, so quality matches the original far better than activation quantization, at ~1/4 of the file size.
The inswapper embedding map ("emap", unused by the graph) is written separately as emap.wasm.

  python web/build_web_models.py <models_dir> <out_dir>
Output is split into <15 MB parts (the artifact per-file limit) plus manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

PART = 14 * 1024 * 1024
MODELS = {  # name -> (source file, quantize?)
    "det": ("scrfd_2.5g.onnx", False),
    "arc": ("arcface_w600k_r50.onnx", True),
    "swap": ("inswapper_128.onnx", True),
    "nsfw": ("open_nsfw.onnx", True),
    "enh": ("gpen_bfr_256.onnx", True),
}


def weight_only_int8(model: onnx.ModelProto, min_size: int = 4096) -> onnx.ModelProto:
    g = model.graph
    used = {i for n in g.node for i in n.input}
    init_names = {i.name for i in g.initializer}
    real_inputs = [i for i in g.input if i.name not in init_names]  # some exporters list weights as inputs
    del g.input[:]
    g.input.extend(real_inputs)
    keep, new_nodes = [], []
    for init in g.initializer:
        w = numpy_helper.to_array(init)
        if init.name not in used:
            continue  # drops the emap
        if w.dtype != np.float32 or w.size < min_size or w.ndim < 2:
            keep.append(init)
            continue
        flat = np.abs(w.reshape(w.shape[0], -1)).max(1)
        scale = np.where(flat > 1e-12, flat / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.round(w / scale.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int8)
        keep += [numpy_helper.from_array(q, init.name + "_q"), numpy_helper.from_array(scale, init.name + "_s")]
        new_nodes.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name],
                                          axis=0, name=init.name + "_dq"))
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    return model


def main(src: str, out: str) -> None:
    src_dir, out_dir = Path(src), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, (fname, quant) in MODELS.items():
        model = onnx.load(str(src_dir / fname))
        if key == "swap":
            emap = numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)
            assert emap.shape == (512, 512)
            (out_dir / "emap.wasm").write_bytes(emap.tobytes())
        if quant:
            model = weight_only_int8(model)
        data = model.SerializeToString()
        parts = []
        for i in range(0, len(data), PART):
            name = f"{key}.part{i // PART:02d}.wasm"  # .wasm: an artifact-served binary type
            (out_dir / name).write_bytes(data[i:i + PART])
            parts.append(name)
        manifest[key] = {"parts": parts, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        print(f"{key}: {fname} -> {len(data) / 1048576:.1f} MB in {len(parts)} parts")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
