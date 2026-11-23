"""Real-time pipeline: capture thread -> swap thread -> latest-result slot read by the UI.

Stale frames are dropped (the swap thread always takes the newest capture),
so latency stays bounded when inference is slower than the camera.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from facefusion_live.camera.capture import CaptureThread
from facefusion_live.camera.virtualcam import VirtualCamera
from facefusion_live.core.engine import FaceEngine
from facefusion_live.core.mapping import FaceMapping
from facefusion_live.core.safety import draw_label
from facefusion_live.face_tracking.tracker import FaceTracker

log = logging.getLogger(__name__)


@dataclass
class LiveStats:
    fps: float = 0.0          # processed frames per second
    capture_fps: float = 0.0
    latency_ms: float = 0.0   # capture -> result available
    process_ms: float = 0.0
    faces: int = 0
    dropped: int = 0
    adaptive_level: int = 0
    resolution: tuple = (0, 0)


class LivePipeline:
    def __init__(self, engine: FaceEngine, mapping: FaceMapping, source: Union[int, str],
                 virtual_cam: bool = False):
        self.engine = engine
        self.mapping = mapping
        self.source = source
        self.want_vcam = virtual_cam
        self.vcam: Optional[VirtualCamera] = None
        self.capture: Optional[CaptureThread] = None
        self.stats = LiveStats()
        self.error: Optional[str] = None
        self.enabled = True              # swap on/off without stopping the camera
        self._result: Optional[np.ndarray] = None
        self._result_seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._level = 0

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        s = self.engine.settings
        self.capture = CaptureThread(self.source, s.camera_width, s.camera_height, s.camera_fps).start()
        w, h, fps = self.capture.actual
        self.stats.resolution = (w, h)
        if self.want_vcam:
            self.vcam = VirtualCamera()
            try:
                self.vcam.start(w, h, fps or 30)
            except RuntimeError as e:
                self.vcam = None
                self.error = str(e)
        self._thread = threading.Thread(target=self._run, name="live-swap", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.capture:
            self.capture.stop()
        if self._thread:
            self._thread.join(5)
        if self.vcam:
            self.vcam.stop()
        log.info("Live stopped. %s", self.stats)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def result(self, after_seq: int = 0):
        with self._lock:
            if self._result_seq <= after_seq:
                return None
            return self._result, self._result_seq

    # ---- adaptive quality ---------------------------------------------------
    def _effective(self) -> FaceEngine:
        """Engine view with adaptive overrides; user settings are never mutated."""
        s = copy.copy(self.engine.settings)
        if s.adaptive:
            if self._level >= 1:
                s.detect_interval = max(s.detect_interval, 8)
            if self._level >= 2:
                s.pixel_boost = 1
                s.enhancer = "none"
            if self._level >= 3:
                s.det_size = 320
                s.occlusion_mask = s.region_mask = False
                s.landmarker = "none"  # last resort: skip the 68-point landmark pass
                s.mouth_preserve = False
        view = copy.copy(self.engine)
        view.settings = s
        return view

    def _adapt(self, proc_s: float) -> None:
        if not self.engine.settings.adaptive:
            self._level = 0
            return
        budget = 1.0 / max(self.engine.settings.camera_fps, 1)
        if proc_s > budget * 1.15 and self._level < 3:
            self._level += 1
        elif proc_s < budget * 0.5 and self._level > 0:
            self._level -= 1
        self.stats.adaptive_level = self._level

    # ---- worker ----------------------------------------------------------------
    def _run(self) -> None:
        tracker = FaceTracker(self.engine)
        seq = 0
        last_t = time.perf_counter()
        ema_proc = None
        frames, t_window = 0, time.perf_counter()
        adapt_acc, adapt_n = 0.0, 0
        try:
            while not self._stop.is_set():
                got = self.capture.latest(seq, 0.5)
                if self.capture.error:
                    self.error = self.capture.error
                if got is None:
                    continue
                frame, ts, new_seq = got
                self.stats.dropped += max(0, new_seq - seq - 1)
                seq = new_seq
                t0 = time.perf_counter()
                eng = self._effective()
                tracker.engine = eng
                has_source = self.mapping.default is not None or bool(self.mapping.references)
                if self.enabled and has_source:
                    faces = tracker.update(frame, max(t0 - last_t, 1e-3), self.mapping.needs_embeddings())
                    out = eng.process(frame, self.mapping.assign(faces))
                    self.stats.faces = len(faces)
                else:
                    out = frame
                last_t = t0
                if self.engine.settings.live_watermark:
                    out = draw_label(out.copy() if out is frame else out)
                if self.vcam:
                    self.vcam.send(out)
                now = time.perf_counter()
                proc = now - t0
                ema_proc = proc if ema_proc is None else 0.9 * ema_proc + 0.1 * proc
                self.stats.process_ms = ema_proc * 1000
                self.stats.latency_ms = (now - ts) * 1000
                with self._lock:
                    self._result = out
                    self._result_seq += 1
                frames += 1
                adapt_acc += proc
                adapt_n += 1
                if adapt_n >= 15:
                    self._adapt(adapt_acc / adapt_n)
                    adapt_acc, adapt_n = 0.0, 0
                if now - t_window >= 1.0:
                    self.stats.fps = frames / (now - t_window)
                    self.stats.capture_fps = self.capture.capture_fps
                    frames, t_window = 0, now
        except Exception as e:  # noqa: BLE001
            log.exception("Live pipeline crashed")
            self.error = f"Live processing stopped: {e}"
