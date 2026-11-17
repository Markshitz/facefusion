"""Image→video face swapping: scan, preview, process with tracking, export with audio.

Frames stream through FFmpeg pipes (decode → swap → encode) on three threads;
no temporary PNG sequences are written to disk.
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import tempfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from facefusion_live.core.align import warp_face
from facefusion_live.core.engine import FaceEngine
from facefusion_live.core.mapping import FaceMapping
from facefusion_live.core.safety import METADATA_COMMENT, draw_label
from facefusion_live.face_tracking.tracker import FaceTracker
from facefusion_live.video import ffmpeg as ff

log = logging.getLogger(__name__)
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


class Cancelled(Exception):
    pass


@dataclass
class Progress:
    done: int
    total: int
    fps: float
    eta: float


@dataclass
class Person:
    embedding: np.ndarray
    thumbnail: np.ndarray
    count: int
    first_time: float


def iter_frames(path: str, info: ff.VideoInfo, start: float = 0.0, duration: Optional[float] = None,
                cancel: Optional[threading.Event] = None):
    p = ff.open_reader(path, start, duration)
    size = info.width * info.height * 3
    n = 0
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            n += 1
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
        if n == 0 and p.wait() != 0:
            err = p.stderr.read().decode(errors="replace").strip()
            raise ff.FFmpegError(f"Could not decode '{Path(path).name}': {err[-300:]}")
    finally:
        p.kill()
        p.wait()


def scan_people(engine: FaceEngine, path: str, samples: int = 24, threshold: float = 0.45,
                cancel: Optional[threading.Event] = None) -> list[Person]:
    """Sample frames, detect faces and cluster them by identity."""
    info = ff.probe(path)
    people: list[Person] = []
    dur = max(info.duration, 0.1)
    for k in range(samples):
        t = dur * (k + 0.5) / samples
        frame = next(iter_frames(path, info, t, 0.5, cancel), None)
        if frame is None:
            continue
        if k % 6 == 0:
            engine.check_content(frame, "video")
        for f in engine.analyze(frame):
            best = max(people, key=lambda p: float(np.dot(p.embedding, f.embedding)), default=None)
            if best is not None and float(np.dot(best.embedding, f.embedding)) > threshold:
                e = best.embedding * best.count + f.embedding
                best.embedding = e / np.linalg.norm(e)
                best.count += 1
            else:
                thumb, _ = warp_face(frame, f.kps, "arcface_128", 128)
                people.append(Person(f.embedding, thumb, 1, t))
    people.sort(key=lambda p: -p.count)
    return people


def process_video(engine: FaceEngine, mapping: FaceMapping, src: str, out: str,
                  progress: Optional[Callable[[Progress], None]] = None,
                  cancel: Optional[threading.Event] = None, start: float = 0.0, duration: Optional[float] = None,
                  on_frame: Optional[Callable[[np.ndarray], None]] = None, workers: int = 0) -> str:
    """Swap faces throughout `src` and write `out` (original FPS, original audio, AI label)."""
    cancel = cancel or threading.Event()
    s = engine.settings
    info = ff.probe(src)
    total = info.frames
    if duration:
        total = min(total, int(round(duration * float(info.fps))) + 1)
    if start:
        total = max(1, min(total, int(round((info.duration - start) * float(info.fps)))))
    encoder = ff.pick_encoder(s.video_encoder)
    tmpdir = Path(tempfile.mkdtemp(prefix="ffl_"))
    video_only = str(tmpdir / ("v" + (Path(out).suffix or ".mp4")))
    if Path(out).suffix.lower() == ".avi":
        video_only = str(tmpdir / "v.mkv")
    writer = ff.open_writer(video_only, info.width, info.height, info.fps, encoder, s.video_quality, METADATA_COMMENT)
    tracker = FaceTracker(engine)
    mapping.reset()
    q_in: queue.Queue = queue.Queue(maxsize=8)
    q_out: queue.Queue = queue.Queue(maxsize=8)
    errors: list[BaseException] = []
    dt = 1.0 / float(info.fps)

    def read():
        try:
            for i, fr in enumerate(iter_frames(src, info, start, duration, cancel)):
                if i % 300 == 0 and s.nsfw_check:
                    engine.check_content(fr, "video")
                q_in.put(fr)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            q_in.put(None)

    def write():
        try:
            while True:
                fr = q_out.get()
                if fr is None:
                    break
                writer.stdin.write(fr.tobytes())
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            cancel.set()

    rt = threading.Thread(target=read, daemon=True)
    wt = threading.Thread(target=write, daemon=True)
    rt.start()
    wt.start()
    t0 = time.perf_counter()
    done = 0
    # Tracking and face assignment must run in frame order (main thread); the swap itself runs on worker
    # threads so a GPU is fed continuously. Results are written strictly in order.
    workers = max(1, int(workers or (2 if engine.provider != "CPU" else 1)))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="swap")
    pending: deque = deque()

    def render(fr, pairs):
        res = engine.process(fr, pairs)
        return draw_label(res if res is not fr else fr.copy())

    def emit_one():
        nonlocal done
        res = pending.popleft().result()
        _put(q_out, res, cancel)
        done += 1
        if on_frame and done % 5 == 1:
            on_frame(res)
        if progress and (done % 5 == 0 or done == total):
            el = time.perf_counter() - t0
            rate = done / el if el > 0 else 0.0
            progress(Progress(done, max(total, done), rate, (max(total - done, 0) / rate) if rate else 0.0))

    try:
        while True:
            fr = q_in.get()
            if fr is None or cancel.is_set():
                break
            faces = tracker.update(fr, dt, need_embedding=mapping.needs_embeddings())
            pending.append(pool.submit(render, fr, mapping.assign(faces)))
            while len(pending) > workers:
                emit_one()
        while pending and not cancel.is_set():
            emit_one()
        pool.shutdown(wait=True)
        if errors and not isinstance(errors[0], Cancelled):
            raise errors[0]
        if cancel.is_set():
            raise Cancelled()
    except BaseException:
        cancel.set()
        pool.shutdown(wait=True, cancel_futures=True)
        _drain(q_in)
        q_out.put(None)
        wt.join(5)
        writer.kill()
        writer.wait()
        rt.join(5)
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    q_out.put(None)
    wt.join()
    writer.stdin.close()
    err = writer.stderr.read().decode(errors="replace")
    if writer.wait() != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise ff.FFmpegError(f"Encoder {encoder} failed: {err[-400:]}")
    rt.join()
    try:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if info.has_audio:
            if start or duration:
                _mux_segment(video_only, src, out, start, duration, encoder)
            else:
                ff.mux_audio(video_only, src, out, METADATA_COMMENT, encoder)
        else:
            if Path(out).suffix.lower() == ".avi":
                ff.run([ff.ffmpeg_bin(), "-v", "error", "-y", "-i", video_only, "-c", "copy", *ff.container_args(out, encoder), out])
            else:
                shutil.move(video_only, out)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    log.info("Video done: %s (%d frames, %.1f fps, encoder %s, tracker %s)", out, done,
             done / max(time.perf_counter() - t0, 1e-6), encoder, tracker.stats)
    return out


def _mux_segment(video_only: str, src: str, out: str, start: float, duration: Optional[float], encoder: str) -> None:
    args = [ff.ffmpeg_bin(), "-v", "error", "-y", "-i", video_only, "-ss", f"{start:.3f}"]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    args += ["-i", src, "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", *ff.container_args(out, encoder),
             "-c:a", "aac", "-shortest", out]
    r = ff.run(args)
    if r.returncode != 0:
        raise ff.FFmpegError(r.stderr.decode(errors="replace")[-300:])


def _put(q: queue.Queue, item, cancel: threading.Event) -> None:
    while not cancel.is_set():
        try:
            q.put(item, timeout=0.2)
            return
        except queue.Full:
            pass
    raise Cancelled()


def _drain(q: queue.Queue) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def default_output_path(src: str, folder: Optional[str] = None) -> str:
    p = Path(src)
    ext = p.suffix.lower() if p.suffix.lower() in VIDEO_EXTS else ".mp4"
    return str(Path(folder or p.parent) / f"{p.stem}_swapped{ext}")


def remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
