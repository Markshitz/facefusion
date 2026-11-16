"""Structured logging to console and a rotating file in the app directory."""
from __future__ import annotations

import logging
import logging.handlers

from facefusion_live.app.config import app_home

FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO, to_file: bool = True) -> None:
    root = logging.getLogger()
    if getattr(root, "_ffl_configured", False):
        return
    root.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(FMT))
    root.addHandler(h)
    if to_file:
        try:
            d = app_home() / "logs"
            d.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(d / "ffl.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(logging.Formatter(FMT))
            root.addHandler(fh)
        except OSError:
            pass
    root._ffl_configured = True  # type: ignore[attr-defined]
