"""Qt application bootstrap."""
from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def create_app():
    from PySide6.QtWidgets import QApplication
    from facefusion_live import APP_NAME
    from facefusion_live.ui.style import DARK_QSS
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    return app


def run_gui() -> int:
    from facefusion_live.app.config import load_settings
    from facefusion_live.ui.context import AppContext
    from facefusion_live.ui.main_window import MainWindow

    app = create_app()

    def excepthook(t, v, tb):
        log.error("Unhandled error", exc_info=(t, v, tb))
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Unexpected error", f"{v}\n\nDetails were written to the log file.")

    sys.excepthook = excepthook
    win = MainWindow(AppContext(load_settings()))
    win.show()
    if not win.first_run_checks():
        return 0
    return app.exec()
