"""User configuration stored as JSON in the per-user app directory.

Nothing is hardcoded to a machine path: the app directory comes from the
platform convention and every path can be overridden via config or env vars
(FFL_HOME, FFL_MODELS_DIR).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)


def app_home() -> Path:
    env = os.environ.get("FFL_HOME")
    if env:
        return Path(env)
    exe_dir = Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False) and (exe_dir / "portable.txt").exists():
        return exe_dir / "data"  # portable mode: settings, logs and models stay next to the exe
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "FaceFusionLive"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "facefusion-live"


# Quality presets. Measured on CPU: the swap model is ~97% of the time; pixel boost x2 runs it 4x for ~10% more
# detail, while GPEN restoration gives most of the sharpness for ~0.3x of one swap pass.
PRESETS = {
    "best": dict(pixel_boost=2, enhancer="gpen_bfr_256", enhance_strength=0.8, color_match=0.3, forehead_blend=0.25),
    "balanced": dict(pixel_boost=1, enhancer="gpen_bfr_256", enhance_strength=0.7, color_match=0.3, forehead_blend=0.25),
    "fast": dict(pixel_boost=1, enhancer="none", color_match=0.3, forehead_blend=0.25),
}


@dataclass
class Settings:
    # inference
    provider: str = "auto"            # auto | cuda | directml | cpu
    device_id: int = 0
    models_dir: str = ""              # empty -> <app_home>/models
    detector: str = "retinaface_10g"  # retinaface_10g (SCRFD-10G) | scrfd_2.5g
    swapper: str = "auto"             # auto | inswapper_128 | inswapper_128_fp16
    det_size: int = 640
    det_threshold: float = 0.5
    # swap quality
    pixel_boost: int = 1              # 1, 2 or 4 -> crop 128/256/512 (see PRESETS)
    blend: float = 1.0                # 0..1 opacity of the swapped face
    mask_blur: float = 0.3            # feather amount of box mask
    color_match: float = 0.3          # 0..1 LAB statistic transfer strength
    forehead_blend: float = 0.25      # 0..0.5 fade of the crop top into the original forehead (hides seams)
    occlusion_mask: bool = False      # XSeg occluder mask
    region_mask: bool = False         # BiSeNet face-parsing mask
    mouth_preserve: bool = False      # keep the original mouth (see mouth_region)
    mouth_region: str = "inner"       # inner = teeth/tongue/opening inside the inner lip line; lips = whole lips
    landmarker: str = "2dfan4"        # 2dfan4 = 68 measured landmarks (precise lips/eyes); none = detector points
    offline_smoothing: bool = True    # video export: zero-lag smoothing using neighbouring frames
    enhancer: str = "gpen_bfr_256"    # none | gpen_bfr_256 | gpen_bfr_512 | gfpgan_1.4 | codeformer
    enhance_strength: float = 0.7
    codeformer_fidelity: float = 0.7
    # tracking / temporal
    tracking: bool = True
    detect_interval: int = 5          # full-frame re-detection every N frames
    smoothing: float = 0.5            # 0 = raw landmarks, 1 = heavy smoothing
    similarity_threshold: float = 0.35
    # live
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    live_watermark: bool = True
    adaptive: bool = True
    # export
    video_quality: int = 18           # CRF / CQ (lower = better)
    video_encoder: str = "auto"
    nsfw_check: bool = True
    telemetry: bool = False           # never used; explicit opt-in placeholder
    consent_acknowledged: bool = False
    recent: dict = field(default_factory=dict)

    def resolved_models_dir(self) -> Path:
        env = os.environ.get("FFL_MODELS_DIR")
        if env:
            return Path(env)
        return Path(self.models_dir) if self.models_dir else app_home() / "models"


def config_path() -> Path:
    return app_home() / "config.json"


def load_settings(path: Path | None = None) -> Settings:
    path = path or config_path()
    s = Settings()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {f.name for f in fields(Settings)}
            for k, v in data.items():
                if k in known:
                    setattr(s, k, v)
        except (OSError, ValueError) as e:
            log.warning("Ignoring unreadable config %s: %s", path, e)
    return s


def save_settings(s: Settings, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
    os.replace(tmp, path)
