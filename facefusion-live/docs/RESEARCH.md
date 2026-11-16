# Deep-Live-Cam analysis (reference project)

Inspected: `hacksider/Deep-Live-Cam` (main branch, Nov 2026), licensed AGPL-3.0. No code was copied.

**Pipeline.** `insightface.app.FaceAnalysis('buffalo_l')` runs detection (SCRFD det_10g) plus ArcFace recognition
plus 2d106 landmarks on every analysed frame. Swapping uses `inswapper_128_fp16` through insightface's model zoo,
then a custom paste-back with optional mouth mask, Poisson blend and opacity. The enhancers are GFPGAN and GPEN
256/512 (ONNX). The NSFW filter is `opennsfw2`, which brings in Keras/TensorFlow.

**Live mode.** A `VideoCapturer` thread with an optional `AsyncLiveSwapper`. Detection is cached between frames
only on Apple Silicon (`get_faces_optimized`). On other platforms it detects on every frame.

**Video mode.** FFmpeg extracts all frames to temporary PNG files, processes them, re-encodes and then restores
the audio. This means heavy disk I/O and temporary storage of several GB for long videos.

**Models.** Downloaded from HuggingFace. The insightface models are non-commercial.

## Weak points and how this project addresses them

| Deep-Live-Cam | FaceFusion Live |
|---|---|
| `insightface` pip package needs a C++ compiler on Windows (a common install failure) | Own ~60-line SCRFD / ArcFace / inswapper wrappers on plain ONNX Runtime; no compiler, no insightface |
| TensorFlow/Keras just for the NSFW check | OpenNSFW as a 23 MB ONNX model |
| Full detection + recognition on every frame (outside Apple Silicon) | Tracker: full detection every N frames, region-limited detection in between, embeddings only when needed |
| No landmark smoothing → jitter / flicker | One-Euro filter, sticky identity assignment |
| PNG frame dump for video | Streaming FFmpeg pipes, three-stage threaded pipeline |
| Video "map faces" relies on per-frame clustering | People clustered once by identity; tracks matched by embedding with hysteresis |
| 128 px swap output upscaled | Pixel boost (256/512 px from interleaved 128 px swaps) |
| Hand/object occlusions pasted over | XSeg occlusion + BiSeNet region masks |

## Risks identified

1. Model licences (insightface: non-commercial). Documented, and the user confirms each download.
2. GPU runtime packaging: CUDA DLLs make the installer about 2 GB. DirectML is shipped as the default build.
3. Webcam and virtual-camera behaviour depends on the hardware and can only be verified manually.
4. CPU-only machines are about 1–2 FPS live, because inswapper costs about 600 ms on 4 cores.
