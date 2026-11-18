"""Application state shared by all tabs."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from facefusion_live.app.config import Settings, save_settings
from facefusion_live.core.engine import FaceEngine
from facefusion_live.models.manifest import human_size
from facefusion_live.models.store import ModelStore


class AppContext(QObject):
    engineReset = Signal()
    settingsChanged = Signal()

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self._engine: FaceEngine | None = None
        self._lock = threading.Lock()

    @property
    def store(self) -> ModelStore:
        return ModelStore(self.settings.resolved_models_dir())

    def engine(self) -> FaceEngine:
        """Create the engine on first use; raises a user-readable error if core models are missing."""
        with self._lock:
            if self._engine is None:
                eng = FaceEngine(self.settings, self.store)
                missing = eng.store.missing(eng.required_models())
                if missing:
                    raise RuntimeError("Required models are not installed: " + ", ".join(
                        f"{m.name} ({human_size(m.size)})" for m in missing) + ". Open the Settings tab > Models to download them.")
                self._engine = eng
            return self._engine

    def reset_engine(self) -> None:
        """Drop sessions (e.g. after changing GPU/CPU or detector)."""
        with self._lock:
            if self._engine is not None:
                self._engine.sessions.clear()
            self._engine = None
        self.engineReset.emit()

    def save(self) -> None:
        save_settings(self.settings)
        self.settingsChanged.emit()
