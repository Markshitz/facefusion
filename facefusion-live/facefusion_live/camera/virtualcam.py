"""Optional virtual-camera output via pyvirtualcam.

Backends: Windows -> OBS Virtual Camera (install OBS Studio 26+) or Unity Capture;
Linux -> v4l2loopback; macOS -> OBS. If no backend exists, start() raises with
setup instructions; the live preview keeps working.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

SETUP_HELP = (
    "Virtual camera backend not found.\n"
    "Windows: install OBS Studio (26 or newer) from https://obsproject.com, start it once, "
    "then choose 'OBS Virtual Camera' as the camera in Zoom/Teams/Discord.\n"
    "Linux: sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback exclusive_caps=1")


class VirtualCamera:
    def __init__(self):
        self.cam = None
        self.device: Optional[str] = None

    @staticmethod
    def available() -> bool:
        try:
            import pyvirtualcam  # noqa: F401
            return True
        except ImportError:
            return False

    def start(self, width: int, height: int, fps: float) -> str:
        try:
            import pyvirtualcam
        except ImportError as e:
            raise RuntimeError("pyvirtualcam is not installed. " + SETUP_HELP) from e
        try:
            self.cam = pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{e}\n\n{SETUP_HELP}") from e
        self.device = self.cam.device
        log.info("Virtual camera started: %s %dx%d@%.0f", self.device, width, height, fps)
        return self.device

    def send(self, frame: np.ndarray) -> None:
        if self.cam is None:
            return
        if frame.shape[1] != self.cam.width or frame.shape[0] != self.cam.height:
            import cv2
            frame = cv2.resize(frame, (self.cam.width, self.cam.height))
        self.cam.send(frame)

    def stop(self) -> None:
        if self.cam is not None:
            try:
                self.cam.close()
            finally:
                self.cam = None
