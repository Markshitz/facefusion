"""Hardware diagnostics (NVIDIA via nvidia-smi; no extra dependencies)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def nvidia_query() -> Optional[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                           creationflags=NO_WINDOW)
        name, util, used, total, drv = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "util": int(util), "mem_used": int(used), "mem_total": int(total), "driver": drv}
    except Exception:  # noqa: BLE001
        return None


def process_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        return 0.0


class GPUMonitor:
    """Polls nvidia-smi every `interval` s in the background."""

    def __init__(self, interval: float = 2.0):
        self.info: Optional[dict] = None
        self._stop = threading.Event()
        self.available = shutil.which("nvidia-smi") is not None
        if self.available:
            threading.Thread(target=self._run, args=(interval,), daemon=True, name="gpu-mon").start()

    def _run(self, interval):
        while not self._stop.is_set():
            self.info = nvidia_query()
            time.sleep(interval)

    def stop(self):
        self._stop.set()


def diagnostics_text() -> str:
    import cv2
    import numpy
    import onnxruntime as ort
    lines = [f"OS: {platform.platform()}", f"Python: {sys.version.split()[0]}",
             f"ONNX Runtime: {ort.__version__} — providers: {', '.join(ort.get_available_providers())}",
             f"OpenCV: {cv2.__version__}", f"NumPy: {numpy.__version__}", f"CPU threads: {os.cpu_count()}"]
    g = nvidia_query()
    if g:
        lines.append(f"GPU: {g['name']} (driver {g['driver']}), {g['mem_used']}/{g['mem_total']} MB used")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            lines.append("⚠ NVIDIA GPU found but the CUDA provider is unavailable: install the GPU build "
                         "(onnxruntime-gpu) with CUDA 12 + cuDNN 9, or use DirectML.")
    else:
        lines.append("NVIDIA GPU: not detected (nvidia-smi missing)")
    try:
        from facefusion_live.video.ffmpeg import ffmpeg_bin, working_encoders
        lines.append(f"FFmpeg: {ffmpeg_bin()} — encoders: {', '.join(working_encoders())}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"FFmpeg: NOT FOUND ({e})")
    from facefusion_live.camera.virtualcam import VirtualCamera
    lines.append(f"Virtual camera module: {'installed' if VirtualCamera.available() else 'not installed'}")
    return "\n".join(lines)
