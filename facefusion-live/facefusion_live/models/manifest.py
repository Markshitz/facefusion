"""Model catalogue: URLs, sizes, checksums and licenses.

Weights are hosted on GitHub releases (facefusion-assets / insightface) and
are NOT redistributed with this application. Several carry non-commercial
licenses — see docs/LICENSES.md. Users must confirm every download.
"""
from __future__ import annotations

from dataclasses import dataclass

FF = "https://github.com/facefusion/facefusion-assets/releases/download"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    file: str
    url: str
    size: int
    sha256: str
    purpose: str
    license: str
    required: bool = False


MODELS: dict[str, ModelInfo] = {m.name: m for m in [
    ModelInfo("retinaface_10g", "retinaface_10g.onnx", f"{FF}/models-3.0.0/retinaface_10g.onnx", 16926877,
              "808399c3d0b40ed340634ed590ae564a7caa09e5659916923a95d4470b54ff5a",
              "Face detector (SCRFD-10G, accurate)", "InsightFace: non-commercial research", True),
    ModelInfo("scrfd_2.5g", "scrfd_2.5g.onnx", f"{FF}/models-3.0.0/scrfd_2.5g.onnx", 3295067,
              "2c07342347cef21a619c49dd5664fb8c09570ae9eda5bff3e385c11eafc45ada",
              "Face detector (SCRFD-2.5G, fast)", "InsightFace: non-commercial research"),
    ModelInfo("arcface_w600k_r50", "arcface_w600k_r50.onnx", f"{FF}/models-3.0.0/arcface_w600k_r50.onnx", 174388474,
              "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
              "Face identity embedding (ArcFace R50)", "InsightFace: non-commercial research", True),
    ModelInfo("inswapper_128", "inswapper_128.onnx", f"{FF}/models-3.0.0/inswapper_128.onnx", 555303150,
              "a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7",
              "Face swapper FP32 (best on CPU)", "InsightFace: non-commercial research"),
    ModelInfo("inswapper_128_fp16", "inswapper_128_fp16.onnx", f"{FF}/models-3.0.0/inswapper_128_fp16.onnx", 277680829,
              "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420",
              "Face swapper FP16 (best on GPU)", "InsightFace: non-commercial research"),
    ModelInfo("fan_68_5", "fan_68_5.onnx", f"{FF}/models-3.0.0/fan_68_5.onnx", 944321,
              "ea05f7d9d014ae1d9cdc6e9c72643d8f00d198b5db3fdd384148ddff499a613e",
              "68-point landmarks from 5 points (mouth mask)", "MIT (FaceFusion)", True),
    ModelInfo("gfpgan_1.4", "gfpgan_1.4.onnx", f"{FF}/models-3.0.0/gfpgan_1.4.onnx", 340299087,
              "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
              "Face restoration (GFPGAN 1.4)", "Apache-2.0 (GFPGAN; StyleGAN2 parts NVIDIA non-commercial)"),
    ModelInfo("gpen_bfr_256", "gpen_bfr_256.onnx", f"{FF}/models-3.0.0/gpen_bfr_256.onnx", 75792988,
              "bad8bf0426873828df2dbf4e3b3d9ababba9da7965b8b72426569486f7ae5c25",
              "Face restoration, sharp detail (GPEN-BFR 256) — default", "GPEN: non-commercial research", True),
    ModelInfo("gpen_bfr_512", "gpen_bfr_512.onnx", f"{FF}/models-3.0.0/gpen_bfr_512.onnx", 284340240,
              "d5f066b9068a8b74217f9712e28e875a6144629b108a6f7355acbdb3a2832c54",
              "Face restoration for large faces (GPEN-BFR 512)", "GPEN: non-commercial research"),
    ModelInfo("codeformer", "codeformer.onnx", f"{FF}/models-3.0.0/codeformer.onnx", 376951650,
              "21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f",
              "Face restoration (CodeFormer)", "S-Lab License 1.0: non-commercial"),
    ModelInfo("xseg_1", "xseg_1.onnx", f"{FF}/models-3.1.0/xseg_1.onnx", 70324286,
              "c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3",
              "Occlusion mask (hands, objects in front of face)", "GPL-3.0 (DeepFaceLab XSeg)"),
    ModelInfo("bisenet_resnet_34", "bisenet_resnet_34.onnx", f"{FF}/models-3.0.0/bisenet_resnet_34.onnx", 93632546,
              "4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96",
              "Face parsing / region mask", "MIT (face-parsing.PyTorch)"),
    ModelInfo("open_nsfw", "open_nsfw.onnx", f"{FF}/models-3.0.0/open_nsfw.onnx", 23598282,
              "922b01d7b5b1b6aea5875c36564f5e13bdcf4e945b13175ae42bdf7fea5b44c7",
              "Explicit-content safeguard (Yahoo OpenNSFW)", "BSD-2-Clause", True),
]}


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)
