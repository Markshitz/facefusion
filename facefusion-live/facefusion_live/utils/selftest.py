"""Headless GUI smoke test (used by CI / the Windows build)."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)


def run_selftest(image: str | None = None, shots_dir: str | None = None, report: str | None = None) -> int:
    if report:  # windowed builds have no console: mirror output to a file
        import builtins
        _print = builtins.print
        fh = open(report, "w", encoding="utf-8")

        def tee(*a, **k):
            _print(*a, **k)
            k.pop("file", None)
            _print(*a, **k, file=fh, flush=True)

        builtins.print = tee
    from PySide6.QtWidgets import QApplication

    from facefusion_live.app.config import load_settings
    from facefusion_live.app.gui import create_app
    from facefusion_live.ui.context import AppContext
    from facefusion_live.ui.main_window import MainWindow

    app = create_app()
    s = load_settings()
    s.consent_acknowledged = True
    ctx = AppContext(s)
    win = MainWindow(ctx)
    win.show()
    shots = Path(shots_dir or os.environ.get("FFL_SHOTS", "")) if (shots_dir or os.environ.get("FFL_SHOTS")) else None
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        t = time.time()
        while time.time() - t < 0.3:
            QApplication.processEvents()
        if shots:
            shots.mkdir(parents=True, exist_ok=True)
            win.grab().save(str(shots / f"tab{i}.png"))
    print(f"GUI OK: {win.tabs.count()} tabs built ({', '.join(win.tabs.tabText(i) for i in range(win.tabs.count()))})")
    rc = 0
    store = ctx.store
    try:
        eng = ctx.engine()
        print(f"Engine OK on {eng.provider}; swapper {eng.swapper_name()}")
        if image:
            from facefusion_live.core.safety import read_image
            img = read_image(image)
            faces = eng.analyze(img)
            if not faces:
                print("FAIL: no face detected in self-test image")
                rc = 1
            else:
                src = eng.make_source(img, faces[0])
                t = time.perf_counter()
                out = eng.process(img, [(faces[-1], src)])
                print(f"Swap OK: {len(faces)} faces, swap {1000 * (time.perf_counter() - t):.0f} ms, changed px "
                      f"{int((out != img).any(2).sum())}")
    except RuntimeError as e:
        print(f"Models not installed in {store.dir} — engine test skipped ({e})")
    win.close()
    app.processEvents()
    return rc
