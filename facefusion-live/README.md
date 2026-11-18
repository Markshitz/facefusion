# FaceFusion Live

Local-first face swapping for **photos, videos and live webcams** in one desktop app
(PySide6 + ONNX Runtime + FFmpeg). Everything runs on your computer. Nothing is uploaded.

| Mode | What it does |
|---|---|
| **IMAGE SWAP** | Source photo → target photo. Pick faces by clicking (different sources per face), drag to compare before/after, save PNG/JPG/WEBP. |
| **VIDEO SWAP** | Source photo → MP4/MOV/AVI/MKV. Scans and clusters the people in the video, you click who to replace, preview 3 s, process with tracking, export with the **original audio and FPS**. Batch mode for several videos. |
| **LIVE SWAP** | Source photo → webcam in real time (720p/1080p), FPS/latency/GPU stats, adaptive quality, optional virtual camera. |

All three modes share one engine (`core/engine.py`). Video and live modes also share the tracker and temporal smoothing (`face_tracking/tracker.py`).

> **Responsible use.** You must have consent from every person whose face is used or replaced. Swap buttons
> stay disabled until you confirm consent. Explicit content is detected and refused. Exports always carry a visible
> "AI-generated" label plus metadata. See `docs/RESPONSIBLE_USE.txt`.

---

## Download for Windows (single .exe, live webcam)

**[FaceFusionLive.exe](https://github.com/Markshitz/facefusion/releases/download/windows-app/FaceFusionLive.exe)**
is about 265 MB. Download it and double-click; there is nothing to install. It covers photo, video and **live
webcam** swapping, plus a virtual camera for Zoom, Teams or OBS, and runs on NVIDIA, AMD or Intel graphics
(DirectML) or the CPU.

CI builds it and tests it with a real swap on Windows. It is republished to the
[windows-app release](https://github.com/Markshitz/facefusion/releases/tag/windows-app) on every push, next to the
DirectML and CUDA installers.

On first launch it asks before downloading the models (about 450 MB). SmartScreen may warn because the file is not
code-signed: choose *More info* › *Run anyway*.

## Run without installing (portable, Windows)

Download the `FaceFusionLive-directml-portable` (or `-cuda-portable`) artifact, extract it anywhere (USB stick
included) and double-click `FaceFusionLive\FaceFusionLive.exe`. Because `portable.txt` sits next to the exe,
settings, logs and models are kept in `FaceFusionLive\data\`, and nothing is written elsewhere on the system.
Delete the folder to remove everything. To skip the download prompt, you can copy existing `.onnx` files into
`data\models\`.

## Run in a browser (no installation, nothing to load)

`web/index.html` runs the same SCRFD → ArcFace → inswapper → GPEN pipeline in the browser with ONNX Runtime Web
1.30, and the models are **built in** (203 MB, cached by the browser after the first visit). Nothing is uploaded.

- **Photo swap:** source photo onto target photo. Quality: Fast, Balanced, High (default) or Max. Progress bar with
  time left; full-resolution output; Save PNG with an "AI-generated" label.
- **Live webcam:** real-time swap of the camera feed (largest face or every face), with a burned-in "AI-generated"
  label, landmark smoothing, FPS and latency display, Show original toggle, camera picker and a content check every
  10 s.
  - **Inside claude.ai the camera is blocked.** Artifacts are not allowed to use the camera, and the page explains
    this.
  - **Live works on a normal web address.** Deploy to GitHub Pages (below) or serve `web/` locally.
- **Speed:** models run in a Web Worker, so the page stays responsive. They use the graphics card through WebGPU
  when the browser supports it; the detector, swapper and restorer each do a test run and fall back to CPU if it
  fails or takes over 15 s.
  - The CPU path uses up to 4 threads on hosts where `coi-sw.js` can enable cross-origin isolation.
  - Measured on a 4-core sandbox CPU (no GPU): about 3 s per live frame at 640×360 with 4 threads, 11 s with 1
    thread. That is a slow preview, not smooth video. Real-time needs a GPU (WebGPU); GPU frame rates are **not
    measured yet**.
- **Choosing the processor:** "Processor: CPU only" forces CPU if a GPU driver misbehaves.

**GitHub Pages (enables Live webcam):** `.github/workflows/facefusion-live-web.yml` builds the site:
1. It downloads the models, runs `build_web_models.py` and `fetch_runtime.sh`, and adds the HTML wrapper.
2. It deploys from the default branch.

One-time setup: *Settings › Pages › Source: GitHub Actions*. The site is then
`https://<owner>.github.io/<repo>/`. Deploying publishes the model weights publicly; they are licensed for
non-commercial use only.

**Model bundle:** `web/build_web_models.py <models_dir> web/models` stores large weights as int8 with a per-channel
scale. They are expanded to fp32 when the model loads, so inference runs at full precision. Measured against the
originals:

| Model | Agreement |
|---|---|
| Swapper | 46 dB PSNR |
| GPEN | 44 dB PSNR |
| ArcFace | cosine 0.9997 |

Parts are under 15 MB and named `.wasm`, because the artifact host only serves known binary types.

## Install (Windows, no Python needed)

1. Download `FaceFusionLive-<version>-directml-setup.exe` (works on NVIDIA, AMD and Intel GPUs through DirectML),
   or `…-cuda-setup.exe` (NVIDIA only and usually fastest; needs a recent NVIDIA driver, R535 or newer).
2. Run it. It installs per-user (no admin rights needed) and adds Start-menu and desktop shortcuts.
3. On first launch, accept the responsible-use terms. The app then lists the required models with their sizes and
   asks before downloading anything. The basic set is about 450 MB (CPU build) or 300 MB (GPU build). Optional
   enhancers and masks can be added later in **Settings › Models**.
4. Uninstall from *Settings › Apps*. The uninstaller asks whether to also delete the downloaded models and settings
   (`%LOCALAPPDATA%\FaceFusionLive`).

FFmpeg is bundled. A virtual camera needs OBS Studio 26 or newer (see `docs/VIRTUAL_CAMERA.md`).

## Run from source (Windows / Linux / macOS)

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# GPU (pick one instead of the plain onnxruntime):
pip uninstall -y onnxruntime && pip install "onnxruntime-gpu[cuda,cudnn]"   # NVIDIA
pip uninstall -y onnxruntime && pip install onnxruntime-directml            # Windows, any GPU
python -m facefusion_live                     # GUI (FFmpeg must be on PATH, or set FFL_FFMPEG_DIR)
```

Command line (same engine):

```bash
python -m facefusion_live models --required          # shows sizes, asks before downloading
python -m facefusion_live swap-image -s me.jpg -t photo.jpg -o out.png --consent [--all-faces] [--enhancer gfpgan_1.4]
python -m facefusion_live swap-video -s me.jpg -t clip.mp4 -o out.mp4 --consent [--person 0 | --all-faces]
python -m facefusion_live benchmark -t clip_with_face.mp4
python -m facefusion_live diagnostics
```

Paths come from configuration, not code: `%LOCALAPPDATA%\FaceFusionLive\config.json` (or `~/.config/facefusion-live`).
You can override them with `FFL_HOME`, `FFL_MODELS_DIR` and `FFL_FFMPEG_DIR`. Logs are written to `<home>/logs/ffl.log`.

## Models

All models are ONNX files with SHA-256 checksums, downloaded from GitHub releases only after you confirm, and resumable.
Several are **non-commercial** (see `docs/LICENSES.md`).

| Model | Size | Used for |
|---|---|---|
| retinaface_10g (SCRFD-10G) * / scrfd_2.5g | 16 / 3 MB | detection + 5 landmarks |
| arcface_w600k_r50 * | 166 MB | identity embedding |
| inswapper_128 (FP32, CPU) / inswapper_128_fp16 (GPU) * | 530 / 265 MB | the swap |
| fan_68_5 * | 1 MB | 68 landmarks (mouth mask) |
| open_nsfw * | 23 MB | explicit-content safeguard |
| gpen_bfr_256 * | 72 MB | face restoration (default, sharp detail) |
| gpen_bfr_512 / gfpgan_1.4 / codeformer | 271 / 325 / 360 MB | optional face restoration |
| xseg_1 | 67 MB | occlusion mask (hands, microphones) |
| bisenet_resnet_34 | 89 MB | face-region mask (keeps hair/beard edges) |

## Quality features

- **Pixel boost**: the target is cropped at 256 or 512 px and swapped as 4 or 16 interleaved 128 px grids. The result
  has real detail at 2–4× resolution instead of a 128 px patch scaled up.
- **Masks**: a feathered box mask, combined with optional XSeg occlusion and BiSeNet face-region masks. These cover
  occlusion handling and natural edges.
- **Skin tone match**: transfers LAB colour statistics, with adjustable strength. **Blend strength** sets the opacity.
- **Mouth preservation**: keeps the target's real mouth interior, so speech looks natural.
- **Enhancement (on by default)**: GPEN-256 restoration (sharp eyes, lashes, teeth) after the swap, blended by strength. GPEN-512, GFPGAN or CodeFormer are optional. Default detail is pixel boost ×2. Measured on a large face: sharpness (Laplacian variance) 17.7 → 140, identity 0.85 → 0.84.
- **Temporal stability**: a One-Euro filter on landmarks (smooths jitter, little lag on fast motion). Full detection
  runs only every N frames; in between, a cheap detection in a small region around each tracked face. Identity
  embeddings are averaged per track, and face-to-source assignment is "sticky" to prevent identity flicker.
- **Only the chosen people are swapped**: in video mode faces are matched by ArcFace identity, so strangers are left
  untouched.

## Lip sync and motion accuracy (measured)

`tools/fidelity_bench.py` runs the 2DFAN4 landmark model on the original video and on the swapped result and compares
them frame by frame:
- **Lip-sync correlation:** how closely the swapped mouth opening tracks the original (1.0 = identical).
- **Lip error:** the average difference in mouth opening.
- **Motion error:** the average drift of the eyes, nose and mouth, as a fraction of the distance between the eyes.

| Clip (50 frames) | Pipeline | Lip-sync correlation | Lip error | Motion error | Identity |
|---|---|---|---|---|---|
| talking (LivePortrait d11) | old | 0.885 | 0.120 | 0.039 | 0.795 |
| | **new default** | **0.988** | **0.065** | **0.028** | 0.693 |
| fast head motion (d12) | old | 0.983 | 0.016 | 0.031 | 0.744 |
| | **new default** | **0.988** | **0.011** | **0.027** | 0.698 |

The new default pipeline:
- **Precise landmarks:** 2DFAN4 measures 68 points per frame; the eyes, nose tip and mouth corners from it drive the
  alignment.
- **Zero-lag video smoothing:** a Savitzky–Golay fit over 2 frames before and 2 after, so there is no trailing behind
  head turns.
- **Original lips:** the target's own lips and mouth opening are kept after restoration. The swap model under-opens
  the mouth, and this is the largest gain.

Keeping the lips costs some measured identity (ArcFace similarity to the source, about 0.80 → 0.69), so *Lip sync
strength* trades the two:

| Lip sync strength | Lip-sync correlation | Identity |
|---|---|---|
| 1.0 (default) | 0.988 | 0.693 |
| 0.6 | 0.976 | 0.725 |
| 0.35 | 0.951 | 0.754 |

### Compared with the real Deep-Live-Cam and FaceFusion programs

The real tools were installed in separate virtual environments and run headless, with their own default settings,
on the same 50-frame clips and source photo:
- **Deep-Live-Cam** at commit `08896a5` (2026-11-08): `run.py -s source -t clip -o out --keep-fps`, on CPU, once
  with defaults and once with `--mouth-mask`.
- **FaceFusion 3.9.2**: `facefusion.py headless-run -s source -t clip -o out`, with all defaults (HyperSwap 1a).
  The only change was disabling its GitHub reachability ping, because the build sandbox's proxy rejects it.

Every output was then scored by the same `tools/fidelity_bench.py` metrics against the original frames.

| Clip | Program | Lip-sync correlation | Lip error | Motion error | Identity |
|---|---|---|---|---|---|
| talking (d11) | Deep-Live-Cam (real, default) | 0.893 | 0.133 | 0.045 | **0.836** |
| | Deep-Live-Cam (real, `--mouth-mask`) | 0.919 | 0.106 | 0.040 | 0.783 |
| | FaceFusion 3.9.2 (real, default) | 0.912 | 0.070 | 0.030 | 0.687 |
| | **FaceFusion Live (default)** | **0.991** | **0.027** | **0.019** | 0.729 |
| fast head (d12) | Deep-Live-Cam (real, default) | 0.976 | 0.023 | 0.033 | **0.809** |
| | Deep-Live-Cam (real, `--mouth-mask`) | 0.961 | 0.021 | 0.030 | 0.772 |
| | FaceFusion 3.9.2 (real, default) | 0.968 | 0.039 | 0.037 | 0.625 |
| | **FaceFusion Live (default)** | **0.994** | **0.009** | **0.019** | 0.707 |

On both clips FaceFusion Live follows the lips and head motion more closely than both real programs on every lip and
motion measure. Its motion error is about half of theirs. Deep-Live-Cam keeps a higher identity score because it
pastes the source's own mouth shape without reshaping it to the speech. FaceFusion Live's identity is higher than
real FaceFusion's.

### Earlier: reconstructed defaults (same engine)

Before the real programs could be run, each tool's default settings were reconstructed in this engine. The real runs
above agree closely with these numbers. The settings come from
each project's source code:
- **Deep-Live-Cam:** inswapper on the detector's 5 points, re-detected every frame, no smoothing, mouth mask off, no
  enhancer.
- **FaceFusion 3.x:** HyperSwap 1a (256 px), 2DFAN4 alignment, box mask with blur 0.3, no smoothing, no mouth keeping,
  no enhancer.

The detector and paste-back details differ slightly from the real tools. All runs use the same source photo and the
same 50 frames.

| Clip | Pipeline | Lip-sync correlation | Lip error | Motion error | Identity |
|---|---|---|---|---|---|
| talking (d11) | Deep-Live-Cam default | 0.895 | 0.129 | 0.044 | 0.884 |
| | FaceFusion default | 0.917 | 0.082 | 0.033 | 0.738 |
| | **FaceFusion Live default** | **0.988** | 0.065 | 0.028 | 0.693 |
| | FaceFusion Live + HyperSwap | 0.983 | **0.048** | **0.023** | 0.644 |
| fast head (d12) | Deep-Live-Cam default | 0.980 | 0.028 | 0.034 | 0.842 |
| | FaceFusion default | 0.982 | 0.028 | 0.036 | 0.671 |
| | **FaceFusion Live default** | 0.988 | 0.011 | **0.027** | 0.698 |
| | FaceFusion Live + HyperSwap | **0.994** | **0.008** | 0.032 | 0.602 |

**Mouth warp (now the default).** Instead of keeping the target's own lips, the swapped (source) lips are
reshaped onto the original mouth's measured landmarks. This uses a piecewise-affine warp over a Delaunay mesh,
anchored on the nose and jaw, followed by the original mouth opening. Skin-tone match and forehead blend were lowered
to 0.15 to recover identity.

| Clip | Pipeline | Lip-sync correlation | Lip error | Motion error | Identity |
|---|---|---|---|---|---|
| talking (d11) | Deep-Live-Cam default | 0.895 | 0.129 | 0.044 | 0.884 |
| | FaceFusion default | 0.917 | 0.082 | 0.033 | 0.738 |
| | **FaceFusion Live, final default** | **0.990** | **0.028** | **0.019** | 0.729 |
| fast head (d12) | Deep-Live-Cam default | 0.980 | 0.028 | 0.034 | 0.842 |
| | FaceFusion default | 0.982 | 0.028 | 0.036 | 0.671 |
| | **FaceFusion Live, final default** | **0.994** | **0.010** | **0.019** | 0.706 |

The web page, with the same mouth warp but the smaller `2d106det` landmarks, scores 0.974 / 0.046 / 0.027 (identity
0.645) on the talking clip. On the fast-head clip it scores 0.973 / 0.018 / 0.024: lower lip and motion error than
both reconstructed defaults, but slightly lower correlation than them (0.98).

**Results:** FaceFusion Live follows lips and motion more closely than both reconstructed defaults on every lip and
motion measure. The trade-off is the identity score: Deep-Live-Cam keeps more of the source person's lips and face
texture because it doesn't keep the original lips. HyperSwap (*Settings › Swap model*) follows lips even more
closely on these clips, at a further identity cost.

The web page uses InsightFace `2d106det` (1.4 MB) instead of 2DFAN4 to fit the artifact size limit. It scores
correlation 0.962, lip error 0.078 and motion error 0.032 on the same talking clip.

## Performance architecture

- Live: capture thread (keeps only the newest frame) → swap thread → latest-result slot drawn by a 60 Hz UI timer.
  Stale frames are dropped, so latency stays at roughly one processing interval even when inference is slow.
- Adaptive quality (live): if a frame takes longer than the frame budget, the app re-detects less often, then drops
  pixel boost and the enhancer, then lowers detection resolution. It restores them when there is headroom again.
- Video: FFmpeg decode pipe → swap → FFmpeg encode pipe, on three threads. There are **no temporary PNG frames**.
  Uses NVENC, AMF or QSV automatically when the hardware encoder actually works (test-encoded), otherwise x264.
- Paste-back warps only the face region, never the full frame. Sessions are cached and warmed up, and provider
  fallback goes CUDA/DirectML → CPU.

## Benchmarks (measured)

`python -m facefusion_live benchmark -t <video>` prints JSON. Results so far:

**Intel Xeon @ 2.30 GHz, 4 vCPU, no GPU (CPU provider, inswapper FP32, development container):**

| Stage | ms |
|---|---|
| full detection 640 px (SCRFD-10G / 2.5G) | 78 / 31 |
| tracked detection inside a 192 px region | 9 / 4 |
| ArcFace embedding | 68 |
| inswapper (one face) | 620–740 |
| GFPGAN (512 px) | 1030 |

End-to-end on 720p and 1080p: **1.3–1.4 FPS**, p50 670–750 ms. On CPU the swap model is 90% of the time. Memory
was flat over three repeated passes (2619.9 MB each), so no leak was observed.

**NVIDIA GPU: not measured yet.** No GPU was available during development. The 30 FPS at 720p target is an
expectation, not a measured result. Please run the benchmark on your GPU and record the numbers here.

## Testing

```bash
pip install pytest
pytest                                               # 18 tests that need no models (pipeline, ffmpeg, safety, tracking)
FFL_MODELS_DIR=… FFL_TEST_MEDIA=… pytest             # + 12 real-inference tests + 5 GUI end-to-end tests
```

`FFL_TEST_MEDIA` must contain `t1.jpg` (the InsightFace group photo) and `lena.jpg` (OpenCV sample). CI downloads
both. Windows CI (`.github/workflows/facefusion-live-windows.yml`) runs the suite, builds the executable, launches it
headless for a real swap, builds the installer, then installs, launches and uninstalls it.

| Scenario | Automated test |
|---|---|
| No face / multiple faces / partly covered face / poor lighting / 30° rotation | ✅ `test_models.py` |
| Swap only the selected person in a moving video (≥ 22/24 frames) | ✅ |
| Audio + FPS + frame count preserved (MP4/MKV/MOV/AVI) | ✅ `test_core.py` |
| Cancel during processing deletes partial output | ✅ |
| Webcam disconnection reported, auto-reconnect loop | ✅ (file-backed camera) |
| Missing models → clear message / GPU unavailable → CPU fallback | ✅ |
| Live pipeline stop latency < 2 s ("shutdown during inference") | ✅ |
| GUI: image, video and live workflows driven through real widgets | ✅ `test_ui.py` |
| Real USB webcam, real NVIDIA/AMD GPU FPS, virtual camera in Zoom/OBS | ⚠️ **manual**, needs hardware |

## Troubleshooting

| Problem | Fix |
|---|---|
| "Required models are not installed" | Settings › Models › tick › Download selected. Or copy the `.onnx` files into the models folder shown there. |
| Status bar says `Inference: CPU` on an NVIDIA PC | Use the CUDA installer, update the NVIDIA driver, or choose DirectML in Settings. Run *Hardware diagnostics*. |
| Camera does not open | Close Zoom/Teams/OBS (they lock the camera). Check *Windows Settings › Privacy › Camera › Let desktop apps access*. Press ↻. |
| Low FPS | Keep *Adaptive quality* on, set Detail to 128 px, turn off the enhancer and masks, use the SCRFD-2.5G detector, choose 720p. |
| "No face found" | Use a sharp, front-facing, well-lit source photo. The face should be at least about 100 px wide. |
| Video export fails | The *Diagnostics* panel lists working encoders. Choose `libx264` in Settings › Encoder. |
| Explicit-content refusal on a normal photo | The check is deliberately conservative and is not configurable in the UI. Crop the photo closer to the face. |
| Anything else | Attach `%LOCALAPPDATA%\FaceFusionLive\logs\ffl.log` to your report. |

## Layout

```
facefusion_live/
  app/          config, Qt bootstrap          inference/   ORT sessions + provider fallback
  core/         engine, blend, masks, align, mapping, safety
  face_detection/ SCRFD, 68-pt landmarks     face_swap/   ArcFace, inswapper (+pixel boost)
  face_tracking/ tracker + One-Euro filter   enhancement/ GFPGAN / CodeFormer
  video/        ffmpeg pipes, processor       camera/      capture, live pipeline, virtual cam
  models/       manifest + verified downloader  ui/        tabs, widgets, style
  utils/        logging, GPU diagnostics, benchmark, selftest
packaging/      PyInstaller spec, Inno Setup script
tests/          pytest suite
```

See `docs/RESEARCH.md` for the Deep-Live-Cam analysis and `PROGRESS.md` for the development log.
