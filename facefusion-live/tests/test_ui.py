"""Drive the real GUI (offscreen) through all three workflows."""
import time

import numpy as np
import pytest

from conftest import make_video, media, needs_models

pytest.importorskip("PySide6")


def wait(cond, timeout=120):
    from PySide6.QtWidgets import QApplication
    t = time.time()
    while not cond():
        QApplication.processEvents()
        time.sleep(0.02)
        if time.time() - t > timeout:
            raise TimeoutError
    QApplication.processEvents()


@pytest.fixture(scope="module")
def win():
    from facefusion_live.app.config import Settings
    from facefusion_live.app.gui import create_app
    from facefusion_live.ui.context import AppContext
    from facefusion_live.ui.main_window import MainWindow
    create_app()
    w = MainWindow(AppContext(Settings(consent_acknowledged=True, provider="cpu", swapper="inswapper_128")))
    w.ctx.save = lambda: None  # do not write user config during tests
    w.show()
    yield w
    w.close()


def test_tabs_exist(win):
    assert [win.tabs.tabText(i) for i in range(4)] == ["IMAGE SWAP", "VIDEO SWAP", "LIVE SWAP", "SETTINGS"]


@needs_models
def test_image_tab_workflow(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    t = win.image_tab
    t.src.file.set_path(media("lena.jpg"))
    wait(lambda: t.src.source is not None)
    t.tfile.set_path(media("t1.jpg"))
    wait(lambda: len(t.faces) == 6)
    assert not t.swap_btn.isEnabled()          # consent gate
    t.consent.setChecked(True)
    t.toggle_face(3)
    assert t.swap_btn.isEnabled() and 3 in t.assign
    t.swap_btn.click()
    wait(lambda: t.result is not None)
    assert (t.result != t.target).any()
    out = tmp_path / "res.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    t.save_btn.click()
    assert out.exists() and b"AI-generated" in out.read_bytes()


@needs_models
def test_video_tab_workflow(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from facefusion_live.core.safety import read_image
    from facefusion_live.video import ffmpeg as ff
    vid = make_video(tmp_path / "v.mp4", read_image(media("t1.jpg")), frames=12, fps=12, size=(640, 360))
    t = win.video_tab
    t.src.file.set_path(media("lena.jpg"))
    wait(lambda: t.src.source is not None)
    t.targets.addItem(vid)
    t.targets.setCurrentRow(0)
    assert t.info and t.info.has_audio
    t.scan_btn.click()
    wait(lambda: t.people)
    t.people_row.itemAt(0).widget().click()
    t.consent.setChecked(True)
    assert t.process_btn.isEnabled()
    t.preview_btn.click()
    wait(lambda: not t.busy and t._player is not None)
    t.process_btn.click()
    wait(lambda: not t.busy and t.result_path)
    out = tmp_path / "exported.mp4"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    t.export_btn.click()
    i = ff.probe(str(out))
    assert i.frames == 12 and i.has_audio


@needs_models
def test_live_tab_workflow(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from facefusion_live.core.safety import read_image
    vid = make_video(tmp_path / "cam.mp4", read_image(media("t1.jpg")), frames=20, fps=10, size=(640, 360), audio=False)
    t = win.live_tab
    t.src.file.set_path(media("lena.jpg"))
    wait(lambda: t.src.source is not None)
    t.cam.setCurrentIndex(t.cam.findData("file"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (vid, ""))
    assert not t.start_btn.isEnabled()
    t.consent.setChecked(True)
    t.start_btn.click()
    wait(lambda: t.pipe is not None)
    wait(lambda: t._seq > 2, 60)
    assert "FPS" in t.stats.text()
    t.start_btn.click()  # stop
    assert t.pipe is None and t.start_btn.text() == "Start Live Swap"


def test_settings_models_table(win):
    from facefusion_live.models.manifest import MODELS
    assert win.settings_tab.table.rowCount() == len(MODELS)
