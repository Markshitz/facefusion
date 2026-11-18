"""Responsible-use safeguards: explicit-content check and AI-content labelling."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

from facefusion_live import AI_LABEL, APP_NAME, __version__
from facefusion_live.inference.session import Sessions

NSFW_LIMIT = 0.80
LABEL_TEXT = f"{AI_LABEL} | {APP_NAME}"
METADATA_COMMENT = f"{AI_LABEL} content: face swapped with {APP_NAME} {__version__}"


class ExplicitContentError(RuntimeError):
    pass


class NSFWChecker:
    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.sess = sessions.get(path)
        self.inp = self.sess.get_inputs()[0].name

    def probability(self, img: np.ndarray) -> float:
        x = cv2.resize(img, (224, 224)).astype(np.float32) - np.array([104, 117, 123], np.float32)
        return float(self.sessions.run(self.sess, {self.inp: x[None]})[0][0][1])

    def check(self, img: np.ndarray, what: str = "media") -> None:
        if self.probability(img) > NSFW_LIMIT:
            raise ExplicitContentError(
                f"The {what} appears to contain explicit content. {APP_NAME} does not process explicit "
                "material, to prevent non-consensual intimate imagery.")


def draw_label(img: np.ndarray, text: str = LABEL_TEXT, corner: str = "br") -> np.ndarray:
    """Visible AI label, scaled to the image, drawn in-place."""
    h, w = img.shape[:2]
    scale = max(0.35, min(w, h) / 1400)
    th = max(1, int(round(scale * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    pad = int(6 * scale) + 2
    x = w - tw - 2 * pad if corner.endswith("r") else pad
    y = h - pad - base if corner.startswith("b") else tht + 2 * pad
    x0, y0, x1, y1 = max(x - pad, 0), max(y - tht - pad, 0), min(x + tw + pad, w), min(y + base + pad, h)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * 0.45).astype(np.uint8)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
    return img


def png_with_text(png_bytes: bytes, key: str = "Comment", text: str = METADATA_COMMENT) -> bytes:
    """Insert a tEXt chunk right after IHDR."""
    data = key.encode("latin-1") + b"\x00" + text.encode("latin-1", "replace")
    chunk = struct.pack(">I", len(data)) + b"tEXt" + data + struct.pack(">I", zlib.crc32(b"tEXt" + data) & 0xFFFFFFFF)
    ihdr_end = 8 + 4 + 4 + 13 + 4
    return png_bytes[:ihdr_end] + chunk + png_bytes[ihdr_end:]


def save_labeled_image(img: np.ndarray, path: str | Path, quality: int = 95) -> Path:
    """Write an exported image with visible label and (PNG) metadata. Unicode-path safe."""
    path = Path(path)
    out = draw_label(img.copy())
    ext = path.suffix.lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif ext != ".png":
        raise ValueError(f"Unsupported image format '{ext}'. Use PNG, JPG or WEBP.")
    ok, buf = cv2.imencode(ext, out, params)
    if not ok:
        raise IOError(f"Could not encode {path.name}")
    data = buf.tobytes()
    if ext == ".png":
        data = png_with_text(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def read_image(path: str | Path) -> np.ndarray:
    """cv2.imread that works with non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image '{path}'. Supported: JPG, PNG, WEBP, BMP.")
    return img
