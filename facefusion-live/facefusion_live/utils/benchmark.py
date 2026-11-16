"""Measured (not estimated) per-stage timings, end-to-end throughput and memory stability."""
from __future__ import annotations

import json
import platform
import time
from typing import Optional

import cv2
import numpy as np

from facefusion_live.core.mapping import FaceMapping
from facefusion_live.face_tracking.tracker import FaceTracker
from facefusion_live.utils.gpu import nvidia_query, process_memory_mb


def _t(fn, n=10):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def _frames(target: Optional[str], n: int, size: tuple[int, int]):
    if target:
        cap = cv2.VideoCapture(target)
        out = []
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not out:
                    raise RuntimeError(f"Cannot read {target}")
                continue
            out.append(cv2.resize(f, size))
        return out
    raise RuntimeError("benchmark needs -t <video with a face>")


def run_benchmark(engine, target: Optional[str], frames: int = 60) -> dict:
    s = engine.settings
    rep = {"platform": platform.platform(), "processor": platform.processor(), "provider": engine.provider,
           "swapper": engine.swapper_name(), "gpu": nvidia_query()}
    base = _frames(target, 1, (1280, 720))[0]
    faces = engine.analyze(base)
    if not faces:
        raise RuntimeError("No face in the first frame of the benchmark video")
    f = faces[0]
    src = engine.make_source(base, f)
    stage = {
        "detect_full_640_ms": _t(lambda: engine.detect(base, 640)),
        "detect_roi_192_ms": _t(lambda: engine.detector.detect(base[int(f.bbox[1]):int(f.bbox[3]), int(f.bbox[0]):int(f.bbox[2])], 192)),
        "embed_ms": _t(lambda: engine.embedder.embed(base, f.kps)),
        "swap_model_ms": _t(lambda: engine.swapper.swap(np.zeros((128, 128, 3), np.uint8), src.latent)),
        "swap_face_total_ms": _t(lambda: engine.swap_face(base.copy(), f, src)),
    }
    if engine.store.is_installed("gfpgan_1.4"):
        stage["gfpgan_ms"] = _t(lambda: engine.restorer("gfpgan_1.4").restore_crop(np.zeros((512, 512, 3), np.uint8)), 3)
    rep["stages"] = {k: round(v, 1) for k, v in stage.items()}

    for name, size in (("720p", (1280, 720)), ("1080p", (1920, 1080))):
        fr = _frames(target, frames, size)
        tr = FaceTracker(engine)
        m = FaceMapping(mode="largest", default=src)
        mem0 = process_memory_mb()
        t0 = time.perf_counter()
        lat = []
        for x in fr:
            a = time.perf_counter()
            engine.process(x, m.assign(tr.update(x)))
            lat.append((time.perf_counter() - a) * 1000)
        el = time.perf_counter() - t0
        rep[name] = {"fps": round(len(fr) / el, 2), "p50_ms": round(float(np.median(lat)), 1),
                     "p95_ms": round(float(np.percentile(lat, 95)), 1), "tracker": dict(tr.stats),
                     "rss_growth_mb": round(process_memory_mb() - mem0, 1)}
    # leak check: repeat 720p loop and compare RSS
    fr = _frames(target, frames, (1280, 720))
    tr = FaceTracker(engine)
    m = FaceMapping(mode="largest", default=src)
    rss = []
    for _ in range(3):
        for x in fr:
            engine.process(x, m.assign(tr.update(x)))
        rss.append(round(process_memory_mb(), 1))
    rep["rss_after_each_pass_mb"] = rss
    rep["settings"] = {k: getattr(s, k) for k in ("pixel_boost", "enhancer", "detect_interval", "det_size", "detector")}
    print(json.dumps(rep, indent=2))
    return rep
