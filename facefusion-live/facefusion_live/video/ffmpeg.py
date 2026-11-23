"""FFmpeg discovery, probing, raw-frame pipes and encoder selection."""
from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

log = logging.getLogger(__name__)
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW


class FFmpegError(RuntimeError):
    pass


def _find(name: str) -> str:
    exe = name + (".exe" if sys.platform == "win32" else "")
    candidates = []
    if os.environ.get("FFL_FFMPEG_DIR"):
        candidates.append(Path(os.environ["FFL_FFMPEG_DIR"]) / exe)
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    candidates += [base / "ffmpeg" / exe, Path(sys.executable).parent / "ffmpeg" / exe]
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which(name)
    if not found:
        raise FFmpegError(f"{name} was not found. Reinstall the application or put {exe} on PATH.")
    return found


@functools.lru_cache(None)
def ffmpeg_bin() -> str:
    return _find("ffmpeg")


@functools.lru_cache(None)
def ffprobe_bin() -> str:
    return _find("ffprobe")


def run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, timeout=timeout, creationflags=NO_WINDOW)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Fraction
    frames: int
    duration: float
    has_audio: bool
    codec: str


def probe(path: str | Path) -> VideoInfo:
    r = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)], 60)
    if r.returncode != 0:
        raise FFmpegError(f"Cannot read video '{Path(path).name}': {r.stderr.decode(errors='replace').strip()[-300:]}")
    d = json.loads(r.stdout)
    vs = next((s for s in d["streams"] if s.get("codec_type") == "video"), None)
    if vs is None:
        raise FFmpegError(f"'{Path(path).name}' has no video stream.")
    w, h = int(vs["width"]), int(vs["height"])
    rot = 0
    for sd in vs.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = int(vs.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps = Fraction(vs.get("avg_frame_rate") or "0/1")
    if fps <= 0 or fps > 240:
        fps = Fraction(vs.get("r_frame_rate") or "30/1")
    duration = float(vs.get("duration") or d.get("format", {}).get("duration") or 0)
    frames = int(vs.get("nb_frames") or 0) or int(round(duration * float(fps)))
    has_audio = any(s.get("codec_type") == "audio" for s in d["streams"])
    return VideoInfo(str(path), w, h, fps, frames, duration, has_audio, vs.get("codec_name", ""))


def open_reader(path: str, start: float = 0.0, duration: float | None = None) -> subprocess.Popen:
    args = [ffmpeg_bin(), "-v", "error", "-nostdin"]
    if start > 0:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    args += ["-an", "-sn", "-f", "rawvideo", "-pix_fmt", "bgr24", "-fps_mode", "passthrough", "-"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW,
                            bufsize=10 ** 8)


ENCODERS = {
    "h264_nvenc": ["-preset", "p5", "-rc", "vbr", "-cq", "{q}", "-b:v", "0"],
    "hevc_nvenc": ["-preset", "p5", "-rc", "vbr", "-cq", "{q}", "-b:v", "0"],
    "h264_amf": ["-quality", "quality", "-rc", "cqp", "-qp_i", "{q}", "-qp_p", "{q}"],
    "h264_qsv": ["-global_quality", "{q}"],
    "libx264": ["-preset", "medium", "-crf", "{q}"],
    "libx265": ["-preset", "medium", "-crf", "{q}"],
}


@functools.lru_cache(None)
def working_encoders() -> list[str]:
    """Encoders that actually initialise on this machine (hardware ones are test-encoded)."""
    r = run([ffmpeg_bin(), "-hide_banner", "-encoders"], 30)
    listed = r.stdout.decode(errors="replace")
    ok = []
    for enc in ENCODERS:
        if f" {enc} " not in listed:
            continue
        if enc.startswith("lib"):
            ok.append(enc)
            continue
        t = run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2", "-c:v", enc,
                 "-f", "null", "-"], 30)
        if t.returncode == 0:
            ok.append(enc)
    log.info("Working video encoders: %s", ok)
    return ok


def pick_encoder(pref: str = "auto") -> str:
    ok = working_encoders()
    if pref != "auto" and pref in ok:
        return pref
    for e in ("h264_nvenc", "h264_amf", "h264_qsv", "libx264"):
        if e in ok:
            return e
    if ok:
        return ok[0]
    raise FFmpegError("No usable H.264 encoder in ffmpeg.")


def open_writer(path: str, width: int, height: int, fps: Fraction, encoder: str, quality: int,
                comment: str) -> subprocess.Popen:
    q = str(int(quality))
    enc_args = [a.replace("{q}", q) for a in ENCODERS.get(encoder, [])]
    args = [ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-", "-c:v", encoder, *enc_args,
            "-pix_fmt", "yuv420p", "-metadata", f"comment={comment}", str(path)]
    return subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)


def container_args(out: str, encoder: str) -> list[str]:
    """AVI needs Annex-B H.264/HEVC when stream-copying from MP4/MKV."""
    if out.lower().endswith(".avi"):
        return ["-bsf:v", ("hevc" if "265" in encoder or "hevc" in encoder else "h264") + "_mp4toannexb"]
    return []


def mux_audio(video_only: str, audio_src: str, out: str, comment: str, encoder: str = "libx264") -> None:
    """Copy original audio into the processed video; re-encode to AAC if the container rejects it."""
    base = [ffmpeg_bin(), "-v", "error", "-y", "-i", video_only, "-i", audio_src, "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "copy", *container_args(out, encoder), "-metadata", f"comment={comment}"]
    if out.lower().endswith((".mp4", ".mov")):
        base += ["-movflags", "+faststart"]
    for audio in (["-c:a", "copy"], ["-c:a", "aac", "-b:a", "192k"]):
        r = run(base + audio + [out])
        if r.returncode == 0:
            return
        log.warning("Audio mux attempt failed: %s", r.stderr.decode(errors="replace")[-300:])
    raise FFmpegError("Could not add the original audio to the output video.")
