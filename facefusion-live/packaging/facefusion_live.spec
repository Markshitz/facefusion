# PyInstaller spec — build with:  pyinstaller packaging/facefusion_live.spec --noconfirm
# Produces dist/FaceFusionLive/FaceFusionLive.exe (one-folder build; models are NOT bundled).
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
FFMPEG = ROOT / "packaging" / "ffmpeg"

binaries = collect_dynamic_libs("onnxruntime")
if os.environ.get("FFL_FLAVOR") == "cuda":  # CUDA/cuDNN DLLs from the nvidia-* wheels, same relative layout
    import importlib.util
    for pkg in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cudnn", "nvidia.cufft", "nvidia.curand", "nvidia.cuda_nvrtc"):
        if importlib.util.find_spec(pkg):
            binaries += collect_dynamic_libs(pkg)
datas = []
if FFMPEG.is_dir():
    for exe in ("ffmpeg.exe", "ffprobe.exe", "ffmpeg", "ffprobe"):
        if (FFMPEG / exe).exists():
            binaries.append((str(FFMPEG / exe), "ffmpeg"))
    for lic in FFMPEG.glob("LICENSE*"):
        datas.append((str(lic), "ffmpeg"))
datas += [(str(ROOT / "docs"), "docs"), (str(ROOT / "LICENSE"), ".")]

QT_EXCLUDES = ["PySide6.Qt3DAnimation", "PySide6.Qt3DCore", "PySide6.Qt3DExtras", "PySide6.Qt3DInput", "PySide6.Qt3DLogic",
               "PySide6.Qt3DRender", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
               "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQml", "PySide6.QtCharts", "PySide6.QtDataVisualization",
               "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors",
               "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtLocation", "PySide6.QtPositioning",
               "PySide6.QtSerialPort", "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtTextToSpeech", "PySide6.QtGraphs"]

a = Analysis([str(ROOT / "packaging" / "launcher.py")], pathex=[str(ROOT)], binaries=binaries, datas=datas,
             hiddenimports=collect_submodules("facefusion_live") + ["pygrabber.dshow_graph", "pyvirtualcam"],
             excludes=QT_EXCLUDES + ["tkinter", "matplotlib", "torch", "tensorflow", "scipy", "pandas", "IPython", "pytest"],
             noarchive=False)
pyz = PYZ(a.pure)
icon = ROOT / "packaging" / "icon.ico"
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FaceFusionLive", console=False,
          icon=str(icon) if icon.exists() else None, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="FaceFusionLive", upx=False)
