"""Camera enumeration and a latest-frame-only capture thread."""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CameraDevice:
    index: int
    name: str


def list_cameras(max_probe: int = 6) -> list[CameraDevice]:
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph  # DirectShow order == CAP_DSHOW index
            return [CameraDevice(i, n) for i, n in enumerate(FilterGraph().get_input_devices())]
        except Exception as e:  # noqa: BLE001
            log.warning("DirectShow enumeration failed (%s); probing indices", e)
    elif sys.platform.startswith("linux"):
        import glob
        devs = sorted(glob.glob("/dev/video*"), key=lambda p: int(p.replace("/dev/video", "") or 0))
        out = []
        for d in devs:
            idx = int(d.replace("/dev/video", ""))
            name = d
            try:
                with open(f"/sys/class/video4linux/video{idx}/name") as f:
                    name = f"{f.read().strip()} ({d})"
            except OSError:
                pass
            out.append(CameraDevice(idx, name))
        return out
    found = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(CameraDevice(i, f"Camera {i}"))
        cap.release()
    return found


def open_camera(source: Union[int, str], width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open a webcam index (or a file/URL for testing) with MJPG to avoid USB bandwidth limits."""
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source '{source}'.")
        return cap
    params = [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"), cv2.CAP_PROP_FRAME_WIDTH, width,
              cv2.CAP_PROP_FRAME_HEIGHT, height, cv2.CAP_PROP_FPS, fps]
    if sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform.startswith("linux"):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    for be in backends:
        try:
            cap = cv2.VideoCapture(source, be, params)
        except Exception:  # noqa: BLE001  (older OpenCV without params overload)
            cap = cv2.VideoCapture(source, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise RuntimeError(f"Camera {source} could not be opened. Close other apps using it (Zoom, Teams, OBS) "
                       "and check Windows Settings > Privacy > Camera.")


class CaptureThread:
    """Reads frames continuously and keeps only the newest one (no queue build-up = low latency)."""

    def __init__(self, source: Union[int, str], width: int = 1280, height: int = 720, fps: int = 30,
                 loop_file: bool = True):
        self.source, self.width, self.height, self.fps = source, width, height, fps
        self.loop_file = loop_file
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._ts = 0.0
        self._seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self.actual = (0, 0, 0.0)
        self.capture_fps = 0.0

    def start(self) -> "CaptureThread":
        cap = open_camera(self.source, self.width, self.height, self.fps)
        self.actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       float(cap.get(cv2.CAP_PROP_FPS)))
        log.info("Capture opened %s at %dx%d@%.0f", self.source, *self.actual)
        self._thread = threading.Thread(target=self._run, args=(cap,), name="capture", daemon=True)
        self._thread.start()
        return self

    def _run(self, cap: cv2.VideoCapture) -> None:
        failures = 0
        is_file = isinstance(self.source, str)
        frame_dt = 1.0 / (self.actual[2] or 30)
        last = time.perf_counter()
        n, t0 = 0, time.perf_counter()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file and self.loop_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    failures += 1
                    if failures >= 30:
                        self.error = "Camera disconnected or stopped delivering frames."
                        log.error(self.error)
                        with self._cond:
                            self._cond.notify_all()
                        # try to reopen every second until stopped
                        cap.release()
                        while not self._stop.wait(1.0):
                            try:
                                cap = open_camera(self.source, self.width, self.height, self.fps)
                                self.error, failures = None, 0
                                log.info("Camera reconnected")
                                break
                            except RuntimeError:
                                continue
                    else:
                        time.sleep(0.01)
                    continue
                failures = 0
                if is_file:  # emulate real-time camera pacing
                    wait = frame_dt - (time.perf_counter() - last)
                    if wait > 0:
                        time.sleep(wait)
                    last = time.perf_counter()
                with self._cond:
                    self._frame, self._ts = frame, time.perf_counter()
                    self._seq += 1
                    self._cond.notify_all()
                n += 1
                if n == 30:
                    now = time.perf_counter()
                    self.capture_fps = 30 / (now - t0)
                    n, t0 = 0, now
        finally:
            cap.release()

    def latest(self, after_seq: int, timeout: float = 0.5):
        """Block until a frame newer than `after_seq` exists. Returns (frame, ts, seq) or None."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq or self._stop.is_set() or self.error,
                                       timeout):
                return None
            if self._seq <= after_seq:
                return None
            return self._frame, self._ts, self._seq

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(3)
