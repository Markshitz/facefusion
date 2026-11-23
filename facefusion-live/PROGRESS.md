# Progress log

- **Research.** Studied the Deep-Live-Cam pipeline and its licences (`docs/RESEARCH.md`). HuggingFace is blocked in
  the development sandbox, so all models come from the GitHub releases of facefusion-assets and insightface, with
  SHA-256 checksums pinned in `models/manifest.py`.
- **Engine.** Wrote ONNX wrappers for SCRFD, ArcFace, inswapper (with pixel boost), fan_68_5, XSeg, BiSeNet,
  GFPGAN/CodeFormer and OpenNSFW. Verified: 6/6 faces found in the group photo; mirrored copies of the same face
  score 0.97 similarity, different people about 0; after a swap, identity similarity to the source is 0.89
  (it was 0.10 before).
- **Image mode.** Done, with multi-face and per-face source mapping and labelled export (PNG metadata).
- **Video mode.** Streaming FFmpeg pipes, scanning and clustering of people, tracking, preview segment, cancel,
  batch, audio mux with an AAC fallback, AVI Annex-B fix (found by a test).
- **Live mode.** Capture / swap / UI threads, frame dropping, adaptive quality, stats, virtual camera (untested,
  no driver available).
- **UI.** PySide6 dark UI with 4 tabs and a Face Settings dock, a model manager that asks before downloading, and
  diagnostics. The first-run responsible-use dialog and the consent checkbox gate every swap.
- **Bugs fixed from tests and benchmarks.**
  - One-Euro speed term was 100× too high, so it gave no smoothing.
  - The tracker forced a full re-detection whenever any one track was lost; it now does so only when all are lost.
  - Wrong operator precedence in the live pipeline.
- **Tests.** 18 model-free tests, 12 real-inference tests and 5 GUI end-to-end tests, all passing on Linux.
- **Packaging.** PyInstaller one-folder build verified on Linux: the frozen binary runs the self-test with a real
  swap. Windows build, installer and the install/launch/uninstall check run in GitHub Actions.

## Open items

- GPU benchmarks (needs NVIDIA hardware).
- Real webcam and virtual-camera verification on Windows.
- Optional: an INT8 inswapper for faster CPU-only use (it costs some quality).

## Windows CI (run #4, green: DirectML and CUDA builds)
- All 35 tests pass on `windows-latest` from source, with real inference.
- PyInstaller exe launched headless: "GUI OK: 4 tabs", "Swap OK: 6 faces".
- Inno Setup installers built: DirectML 181 MB, CUDA 664 MB (artifact sizes). Each was silently installed, the
  installed exe ran a real swap, and the uninstaller removed it.
- The runner has no GPU, so DirectML failed to initialise and ONNX Runtime fell back to CPU on its own. The
  fallback works, but GPU speed is still unmeasured.
- Fixed on the way: CI FFmpeg asset name; `-vsync` was removed in new FFmpeg (now `-fps_mode`); decode errors are
  now reported instead of silently returning zero frames.
