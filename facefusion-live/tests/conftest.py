import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
MEDIA = Path(os.environ.get("FFL_TEST_MEDIA", Path(__file__).parent / "media"))


def models_dir():
    from facefusion_live.app.config import Settings
    return Settings().resolved_models_dir()


def have_models(*names):
    from facefusion_live.models.store import ModelStore
    st = ModelStore(models_dir())
    return all(st.is_installed(n) for n in names)


CORE = ("retinaface_10g", "arcface_w600k_r50", "inswapper_128")
needs_models = pytest.mark.skipif(not have_models(*CORE), reason="core models not installed (set FFL_MODELS_DIR)")


def media(name):
    p = MEDIA / name
    if not p.exists():
        pytest.skip(f"test media {p} missing (set FFL_TEST_MEDIA)")
    return str(p)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FFL_HOME", str(tmp_path / "home"))


@pytest.fixture(scope="session")
def engine():
    if not have_models(*CORE):
        pytest.skip("core models missing")
    from facefusion_live.app.config import Settings
    from facefusion_live.core.engine import FaceEngine
    return FaceEngine(Settings(provider="cpu", swapper="inswapper_128", pixel_boost=1, enhancer="none"))


def make_video(path, img, frames=50, fps=25, size=(1280, 720), audio=True, motion=True):
    """Synthetic camera move (rotation/zoom/pan) over a still photo, with a sine audio track."""
    import subprocess

    import cv2
    import numpy as np
    from facefusion_live.video.ffmpeg import ffmpeg_bin
    W, H = size
    args = [ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / fps}"]
    args += ["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    p = subprocess.Popen(args + [str(path)], stdin=subprocess.PIPE)
    base = cv2.resize(img, (int(W * 1.1), int(img.shape[0] * W * 1.1 / img.shape[1])))
    cx, cy = base.shape[1] / 2, base.shape[0] / 2
    for i in range(frames):
        t = i / frames
        m = cv2.getRotationMatrix2D((cx, cy), 8 * np.sin(2 * np.pi * t) if motion else 0, 1 + 0.08 * np.sin(np.pi * t))
        m[0, 2] -= (base.shape[1] - W) / 2 + (30 * np.sin(2 * np.pi * t) if motion else 0)
        m[1, 2] -= (base.shape[0] - H) / 2
        p.stdin.write(cv2.warpAffine(img if False else base, m, (W, H)).tobytes())
    p.stdin.close()
    p.wait()
    return str(path)


class FakeDetector:
    def __init__(self, faces=None):
        self.faces = faces or []

    def detect(self, img, size=640, threshold=0.5, max_faces=0):
        import copy
        return copy.deepcopy(self.faces)


class FakeEngine:
    """Engine stand-in for pipeline tests that must run without model files."""

    def __init__(self, faces=None):
        from facefusion_live.app.config import Settings
        self.settings = Settings(nsfw_check=False)
        self.detector = FakeDetector(faces)
        self.provider = "CPU"
        self.processed = 0

    def detect(self, img, size=None, max_faces=0):
        return self.detector.detect(img)

    def check_content(self, img, what):
        pass

    def process(self, frame, pairs):
        self.processed += 1
        if not pairs:
            return frame
        out = frame.copy()
        out[:20, :20] = 255
        return out
