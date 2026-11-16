"""ONNX Runtime session factory with provider selection and fallback."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import onnxruntime as ort

log = logging.getLogger(__name__)

PROVIDER_ALIASES = {
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def available_providers() -> list[str]:
    return ort.get_available_providers()


_preloaded = False


def _preload_cuda() -> None:
    """onnxruntime-gpu >= 1.21 can load CUDA/cuDNN DLLs shipped as pip packages (used by the CUDA build)."""
    global _preloaded
    if not _preloaded and hasattr(ort, "preload_dlls"):
        _preloaded = True
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001
            log.warning("CUDA DLL preload failed: %s", e)


def resolve_providers(choice: str = "auto", device_id: int = 0) -> list:
    avail = available_providers()
    if "CUDAExecutionProvider" in avail and choice in ("auto", "cuda"):
        _preload_cuda()
    if choice == "auto":
        order = ["cuda", "directml", "coreml"]
    else:
        order = [choice]
    out: list = []
    for key in order:
        name = PROVIDER_ALIASES.get(key, key)
        if name in avail and name != "CPUExecutionProvider":
            if name == "CUDAExecutionProvider":
                out.append((name, {"device_id": device_id, "cudnn_conv_algo_search": "HEURISTIC",
                                   "arena_extend_strategy": "kSameAsRequested"}))
            elif name == "DmlExecutionProvider":
                out.append((name, {"device_id": device_id}))
            else:
                out.append(name)
            break
        if choice != "auto":
            log.warning("Execution provider '%s' is not available (have %s); falling back to CPU", key, avail)
    out.append("CPUExecutionProvider")
    return out


def provider_label(providers: list) -> str:
    first = providers[0]
    name = first[0] if isinstance(first, tuple) else first
    return name.replace("ExecutionProvider", "")


class Sessions:
    """Caches one InferenceSession per model file. Thread-safe creation."""

    def __init__(self, providers: list):
        self.providers = providers
        self._cache: dict[str, ort.InferenceSession] = {}
        self._lock = threading.Lock()
        # DirectML sessions are not safe for concurrent Run() calls.
        self.run_lock = threading.Lock() if provider_label(providers) == "Dml" else None

    def get(self, path: Path) -> ort.InferenceSession:
        key = str(path)
        with self._lock:
            s = self._cache.get(key)
            if s is None:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.log_severity_level = 3
                if provider_label(self.providers) == "Dml":
                    so.enable_mem_pattern = False
                    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                try:
                    s = ort.InferenceSession(key, so, providers=self.providers)
                except Exception as e:  # GPU init failure -> CPU fallback
                    if self.providers[0] == "CPUExecutionProvider":
                        raise
                    log.error("GPU session for %s failed (%s); using CPU", path.name, e)
                    s = ort.InferenceSession(key, so, providers=["CPUExecutionProvider"])
                log.info("Loaded %s on %s", path.name, s.get_providers()[0])
                self._cache[key] = s
            return s

    def run(self, sess: ort.InferenceSession, feeds: dict, outputs=None):
        if self.run_lock:
            with self.run_lock:
                return sess.run(outputs, feeds)
        return sess.run(outputs, feeds)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
