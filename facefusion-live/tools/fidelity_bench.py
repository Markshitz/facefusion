"""Measure how faithfully a swap follows the target's lips and head motion.

For each frame, 2DFAN4 landmarks are measured on the ORIGINAL frame and on the SWAPPED frame:
  * lip sync   : correlation and mean abs error of the inner-mouth opening ratio (orig vs swapped)
  * motion     : mean distance of eye centres, nose tip, mouth corners and lip centres, in eye-distance units
Usage: python tools/fidelity_bench.py <video> <source.jpg> <frames> <config>...
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facefusion_live.app.config import Settings  # noqa: E402
from facefusion_live.core.engine import FaceEngine  # noqa: E402
from facefusion_live.core.mapping import FaceMapping  # noqa: E402
from facefusion_live.face_detection import fan68  # noqa: E402

KEY = [30, 48, 54, 51, 57, 62, 66]


def keypts(lm):
    return np.vstack([lm[36:42].mean(0), lm[42:48].mean(0), lm[KEY]])


def measure(engine, fan, orig_frames, out_frames):
    o_open, s_open, err = [], [], []
    for o, s in zip(orig_frames, out_frames):
        fo, fs = engine.detect(o), engine.detect(s)
        if not fo or not fs:
            continue
        lo, _ = fan(o, fo[0].bbox)
        ls, _ = fan(s, fs[0].bbox)
        iod = np.linalg.norm(lo[36:42].mean(0) - lo[42:48].mean(0))
        o_open.append(fan68.mouth_open_ratio(lo)); s_open.append(fan68.mouth_open_ratio(ls))
        err.append(np.linalg.norm(keypts(lo) - keypts(ls), axis=1).mean() / iod)
    o_open, s_open = np.array(o_open), np.array(s_open)
    return {"lip_corr": round(float(np.corrcoef(o_open, s_open)[0, 1]), 3),
            "lip_mae": round(float(np.abs(o_open - s_open).mean()), 4),
            "motion_err": round(float(np.mean(err)), 4)}


def run(video, source, n, overrides):
    s = Settings(nsfw_check=False)
    for k, v in overrides.items():
        setattr(s, k, v)
    e = FaceEngine(s)
    src = cv2.imread(source)
    S = e.make_source(src, e.analyze(src)[0])
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    from facefusion_live.video.processor import swap_frames
    t = time.perf_counter()
    outs = swap_frames(e, FaceMapping(mode="largest", default=S), frames, fps=25)
    dt = (time.perf_counter() - t) / len(frames)
    fan = fan68.FAN68(e.sessions, e.store.require("2dfan4"))
    r = measure(e, fan, frames, outs)
    r["ms_per_frame"] = round(dt * 1000)
    return r, outs


if __name__ == "__main__":
    video, source, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    for cfg in sys.argv[4:]:
        name, _, js = cfg.partition("=")
        r, outs = run(video, source, n, json.loads(js or "{}"))
        print(f"{Path(video).name:8s} {name:22s} {r}", flush=True)
