# Licenses

FaceFusion Live source code: **AGPL-3.0-or-later** (see `LICENSE`). It is an independent
implementation; no code was copied from Deep-Live-Cam (AGPL-3.0), which was studied as a reference.

## Model weights (downloaded on demand, never bundled)

| Model | Source | License | Commercial use |
|---|---|---|---|
| retinaface_10g (SCRFD-10G), scrfd_2.5g | InsightFace | Non-commercial research | **No** |
| arcface_w600k_r50 | InsightFace | Non-commercial research | **No** |
| inswapper_128 / _fp16 | InsightFace | Non-commercial research | **No** |
| fan_68_5 | FaceFusion | MIT | Yes |
| gfpgan_1.4 | TencentARC GFPGAN | Apache-2.0 (includes NVIDIA StyleGAN2 ops, non-commercial) | Check |
| codeformer | S-Lab | S-Lab License 1.0 (non-commercial) | **No** |
| xseg_1 | DeepFaceLab | GPL-3.0 | Yes (GPL) |
| bisenet_resnet_34 | face-parsing.PyTorch | MIT | Yes |
| open_nsfw | Yahoo | BSD-2-Clause | Yes |

ONNX conversions are fetched from the `facefusion/facefusion-assets` GitHub releases; SHA-256
checksums are verified after download. **The face-swap model (inswapper) is non-commercial, so the
application as a whole must not be used commercially** unless you replace it with a model you are
licensed to use commercially.

## Runtime dependencies

| Component | License |
|---|---|
| PySide6 / Qt 6 | LGPL-3.0 (dynamically linked) |
| ONNX Runtime (CPU / DirectML / GPU) | MIT |
| OpenCV | Apache-2.0 |
| NumPy | BSD-3-Clause |
| onnx | Apache-2.0 |
| psutil | BSD-3-Clause |
| pygrabber | MIT |
| pyvirtualcam (optional) | GPL-2.0 |
| FFmpeg (bundled `ffmpeg.exe`/`ffprobe.exe`, separate executables, BtbN GPL build) | GPL-3.0 — source: https://github.com/BtbN/FFmpeg-Builds |
| NVIDIA CUDA / cuDNN redistributables (CUDA build only) | NVIDIA EULA |
