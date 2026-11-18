"""Main window: Image / Video / Live / Settings tabs + shared Face Settings dock."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDockWidget, QLabel, QMainWindow, QMessageBox, QTabWidget

from facefusion_live import APP_NAME, __version__
from facefusion_live.ui.context import AppContext
from facefusion_live.ui.face_settings import FaceSettingsPanel
from facefusion_live.ui.image_tab import ImageTab
from facefusion_live.ui.live_tab import LiveTab
from facefusion_live.ui.settings_tab import SettingsTab
from facefusion_live.ui.video_tab import VideoTab

log = logging.getLogger(__name__)

RESPONSIBLE_USE = (
    "<b>Responsible use</b><br><br>"
    f"{APP_NAME} runs entirely on this computer. Nothing you load is uploaded.<br><br>"
    "By continuing you agree to:<ul>"
    "<li>Only use faces of people who have given you their <b>explicit consent</b>.</li>"
    "<li>Never create sexual or intimate imagery of real people, or content that deceives, "
    "harasses, defames or impersonates anyone.</li>"
    "<li>Never use the output to bypass identity verification or for fraud.</li>"
    "<li>Clearly disclose AI-generated media. Exports always carry a visible 'AI-generated' label.</li>"
    "</ul>Explicit content is detected and refused. Misuse may be illegal in your jurisdiction.")


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1400, 880)
        self.tabs = QTabWidget()
        self.image_tab = ImageTab(ctx)
        self.video_tab = VideoTab(ctx)
        self.live_tab = LiveTab(ctx)
        self.settings_tab = SettingsTab(ctx)
        self.tabs.addTab(self.image_tab, "IMAGE SWAP")
        self.tabs.addTab(self.video_tab, "VIDEO SWAP")
        self.tabs.addTab(self.live_tab, "LIVE SWAP")
        self.tabs.addTab(self.settings_tab, "SETTINGS")
        self.setCentralWidget(self.tabs)
        dock = QDockWidget("Face settings", self)
        dock.setWidget(FaceSettingsPanel(ctx))
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(300)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.provider_lbl = QLabel()
        self.statusBar().addPermanentWidget(self.provider_lbl)
        ctx.engineReset.connect(self._status)
        self._status()

    def _status(self):
        from facefusion_live.inference.session import provider_label, resolve_providers
        s = self.ctx.settings
        self.provider_lbl.setText(f"Inference: {provider_label(resolve_providers(s.provider, s.device_id))}  ")

    def first_run_checks(self):
        s = self.ctx.settings
        if not s.consent_acknowledged:
            box = QMessageBox(self)
            box.setWindowTitle("Before you start")
            box.setTextFormat(Qt.RichText)
            box.setText(RESPONSIBLE_USE)
            box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            box.button(QMessageBox.Ok).setText("I agree")
            if box.exec() != QMessageBox.Ok:
                QApplication.quit()
                return False
            s.consent_acknowledged = True
            self.ctx.save()
        store = self.ctx.store
        from facefusion_live.core.engine import FaceEngine
        req = FaceEngine(s, store).required_models() + ["fan_68_5", "open_nsfw", "gpen_bfr_256"]
        missing = [m.name for m in store.missing(req)]
        if missing:
            self.tabs.setCurrentWidget(self.settings_tab)
            from facefusion_live.ui.settings_tab import confirm_download
            if confirm_download(self, missing):
                self.settings_tab.download_selected(missing, confirmed=True)
        return True

    def closeEvent(self, e):
        self.live_tab.shutdown()
        self.video_tab.shutdown()
        e.accept()
