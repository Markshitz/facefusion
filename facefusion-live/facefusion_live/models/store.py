"""Locating, verifying and (only on explicit request) downloading models."""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from facefusion_live.models.manifest import MODELS, ModelInfo, human_size

log = logging.getLogger(__name__)


class ModelMissingError(RuntimeError):
    def __init__(self, info: ModelInfo, path: Path):
        super().__init__(
            f"Model '{info.name}' ({human_size(info.size)}) is not installed at {path}. "
            f"Open Settings > Models to download it, or run: python -m facefusion_live models --download {info.name}")
        self.info = info


class ModelStore:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir)

    def path(self, name: str) -> Path:
        return self.dir / MODELS[name].file

    def is_installed(self, name: str) -> bool:
        p = self.path(name)
        return p.is_file() and p.stat().st_size == MODELS[name].size

    def require(self, name: str) -> Path:
        if name not in MODELS:
            raise KeyError(f"Unknown model '{name}'")
        p = self.path(name)
        if not self.is_installed(name):
            raise ModelMissingError(MODELS[name], p)
        return p

    def missing(self, names) -> list[ModelInfo]:
        return [MODELS[n] for n in names if not self.is_installed(n)]

    def verify(self, name: str) -> bool:
        return _sha256(self.path(name)) == MODELS[name].sha256

    def download(self, name: str, progress: Optional[Callable[[int, int], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None) -> Path:
        """Download with resume + checksum. Caller must have obtained user consent."""
        info = MODELS[name]
        dest = self.path(name)
        self.dir.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        start = part.stat().st_size if part.exists() else 0
        if start > info.size:
            part.unlink()
            start = 0
        req = urllib.request.Request(info.url, headers={"User-Agent": "facefusion-live"})
        if start:
            req.add_header("Range", f"bytes={start}-")
        log.info("Downloading %s (%s) from %s", name, human_size(info.size), info.url)
        with urllib.request.urlopen(req, timeout=60) as r:
            if start and r.status != 206:
                start = 0
            with open(part, "ab" if start else "wb") as f:
                done = start
                while True:
                    if cancel and cancel():
                        raise InterruptedError("Download cancelled")
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, info.size)
        if _sha256(part) != info.sha256:
            part.unlink(missing_ok=True)
            raise IOError(f"Checksum mismatch for {name}; the partial file was deleted. Retry the download.")
        os.replace(part, dest)
        return dest


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
