"""Tests that need no model files."""
import json
import threading
import time

import cv2
import numpy as np
import pytest

from conftest import FakeEngine, make_video
from facefusion_live.app.config import Settings, load_settings, save_settings
from facefusion_live.core.blend import box_mask, color_transfer, paste_back
from facefusion_live.core.face import Face, iou
from facefusion_live.core.mapping import FaceMapping
from facefusion_live.core.safety import draw_label, png_with_text, read_image, save_labeled_image
from facefusion_live.face_tracking.tracker import OneEuro
from facefusion_live.inference.session import resolve_providers
from facefusion_live.models.manifest import MODELS
from facefusion_live.models.store import ModelMissingError, ModelStore


def test_config_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    s = Settings(blend=0.4, enhancer="codeformer")
    save_settings(s, p)
    assert load_settings(p).blend == 0.4
    p.write_text(json.dumps({"blend": 0.2, "unknown_key": 1}))
    assert load_settings(p).blend == 0.2
    p.write_text("{corrupt")
    assert load_settings(p).blend == Settings().blend


def test_manifest_and_missing_model_message(tmp_path):
    assert all(len(m.sha256) == 64 and m.url.startswith("https://") for m in MODELS.values())
    st = ModelStore(tmp_path)
    with pytest.raises(ModelMissingError) as e:
        st.require("inswapper_128")
    assert "Settings > Models" in str(e.value) and "MB" in str(e.value)


def test_gpu_unavailable_falls_back_to_cpu():
    prov = resolve_providers("cuda")
    assert prov[-1] == "CPUExecutionProvider"
    assert resolve_providers("cpu") == ["CPUExecutionProvider"]


def test_iou():
    a = np.array([0, 0, 10, 10], np.float32)
    assert iou(a, a) == pytest.approx(1)
    assert iou(a, np.array([20, 20, 30, 30], np.float32)) == 0


def test_box_mask_feathered():
    m = box_mask(128, 0.3)
    assert m.shape == (128, 128) and m[64, 64] == pytest.approx(1, abs=1e-3) and m[0, 0] < 0.05


def test_paste_back_only_touches_roi():
    frame = np.zeros((400, 600, 3), np.uint8)
    crop = np.full((128, 128, 3), 200, np.float32)
    m = np.array([[1, 0, -100], [0, 1, -50]], np.float32)  # crop placed at (100,50)
    paste_back(frame, crop, np.ones((128, 128), np.float32), m)
    assert frame[50 + 64, 100 + 64].tolist() == [200, 200, 200]
    assert frame[300:, 400:].max() == 0


def test_color_transfer_moves_stats():
    src = np.full((64, 64, 3), 60, np.float32)
    src += np.random.rand(64, 64, 3).astype(np.float32) * 10
    ref = np.full((64, 64, 3), 180, np.uint8).astype(np.float32)
    out = color_transfer(src, ref, np.ones((64, 64), np.float32), 1.0)
    assert abs(out.mean() - 180) < 15


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    f = OneEuro(min_cutoff=1.0)
    raw, sm = [], []
    for i in range(200):
        x = np.array([100.0]) + rng.normal(0, 2, 1)
        raw.append(x[0])
        sm.append(f(x, 1 / 30)[0])
    assert np.std(np.diff(sm[20:])) < 0.5 * np.std(np.diff(raw[20:]))
    # fast real motion must still be followed with little lag
    g = OneEuro(min_cutoff=1.0)
    for i in range(60):
        y = g(np.array([i * 10.0]), 1 / 30)[0]
    assert abs(y - 590) < 30


def _face(x, emb=None, tid=-1):
    return Face(np.array([x, 0, x + 50, 50], np.float32), np.zeros((5, 2), np.float32), 0.9, emb, track_id=tid)


def test_mapping_modes():
    e1, e2, e3 = np.eye(3, 512)  # orthogonal identities
    src_a, src_b = object(), object()
    m = FaceMapping(mode="mapped", threshold=0.4)
    m.add(e1, src_a)
    m.add(e2, src_b)
    faces = [_face(0, e1, 1), _face(100, e2, 2), _face(200, e3, 3)]
    pairs = m.assign(faces)
    got = {f.track_id: s for f, s in pairs}
    assert got == {1: src_a, 2: src_b}          # stranger (track 3) untouched
    big = _face(0)
    big.bbox = np.array([0, 0, 200, 200], np.float32)
    m2 = FaceMapping(mode="largest", default=src_a)
    assert m2.assign([_face(300), big])[0][0] is big
    assert FaceMapping(mode="all", default=src_a).assign(faces).__len__() == 3
    assert FaceMapping(mode="all").assign(faces) == []


def test_label_and_png_metadata(tmp_path):
    img = np.full((480, 640, 3), 128, np.uint8)
    out = draw_label(img.copy())
    assert (out != img).any()
    p = save_labeled_image(img, tmp_path / "ü_out.png")
    data = p.read_bytes()
    assert b"tEXt" in data and b"AI-generated" in data
    assert read_image(p).shape == img.shape
    for ext in (".jpg", ".webp"):
        assert read_image(save_labeled_image(img, tmp_path / f"o{ext}")).shape == img.shape
    with pytest.raises(ValueError):
        save_labeled_image(img, tmp_path / "o.tiff")
    ok, buf = cv2.imencode(".png", img)
    assert cv2.imdecode(np.frombuffer(png_with_text(buf.tobytes()), np.uint8), 1) is not None


def test_read_image_errors(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"nope")
    with pytest.raises(ValueError):
        read_image(tmp_path / "bad.jpg")


# ---- video pipeline with a fake engine (no models) -----------------------------------------
@pytest.fixture(scope="module")
def plain_video(tmp_path_factory):
    img = np.random.default_rng(1).integers(0, 255, (360, 640, 3), np.uint8)
    return make_video(tmp_path_factory.mktemp("v") / "in.mp4", img, frames=40, fps=24, size=(640, 360))


def test_video_preserves_fps_frames_audio(plain_video, tmp_path):
    from facefusion_live.video import ffmpeg as ff
    from facefusion_live.video.processor import process_video
    eng = FakeEngine([_face(100)])
    out = process_video(eng, FaceMapping(mode="all", default=object()), plain_video, str(tmp_path / "o.mp4"))
    a, b = ff.probe(plain_video), ff.probe(out)
    assert b.fps == a.fps and b.frames == a.frames and b.has_audio and (b.width, b.height) == (a.width, a.height)
    for ext in (".mkv", ".mov", ".avi"):
        o = process_video(eng, FaceMapping(mode="all", default=object()), plain_video, str(tmp_path / f"o{ext}"))
        assert ff.probe(o).has_audio, ext


def test_video_cancel_cleans_up(plain_video, tmp_path):
    from facefusion_live.video.processor import Cancelled, process_video
    eng = FakeEngine()
    orig = eng.process
    eng.process = lambda f, p: (time.sleep(0.05), orig(f, p))[1]
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    out = tmp_path / "c.mp4"
    with pytest.raises(Cancelled):
        process_video(eng, FaceMapping(mode="all"), plain_video, str(out), cancel=cancel)
    assert not out.exists()


def test_video_segment_preview(plain_video, tmp_path):
    from facefusion_live.video import ffmpeg as ff
    from facefusion_live.video.processor import process_video
    out = process_video(FakeEngine(), FaceMapping(), plain_video, str(tmp_path / "p.mp4"), start=0.5, duration=0.5)
    assert 10 <= ff.probe(out).frames <= 14


def test_unreadable_video(tmp_path):
    from facefusion_live.video import ffmpeg as ff
    (tmp_path / "x.mp4").write_bytes(b"\x00" * 100)
    with pytest.raises(ff.FFmpegError):
        ff.probe(tmp_path / "x.mp4")


def test_live_pipeline_with_file_camera(plain_video):
    from facefusion_live.camera.live import LivePipeline
    eng = FakeEngine([_face(100)])
    lp = LivePipeline(eng, FaceMapping(mode="all", default=object()), plain_video)
    lp.start()
    time.sleep(1.5)
    r = lp.result()
    t = time.time()
    lp.stop()
    assert time.time() - t < 2, "stop must be prompt"
    assert r is not None and r[0].shape == (360, 640, 3)
    assert lp.stats.fps > 5 and lp.error is None and not lp.running


def test_camera_missing_is_clear_error():
    from facefusion_live.camera.capture import CaptureThread
    with pytest.raises(RuntimeError):
        CaptureThread("/nonexistent/cam.mp4").start()


def test_camera_disconnect_reported(plain_video, monkeypatch):
    from facefusion_live.camera import capture
    ct = capture.CaptureThread(plain_video, loop_file=False)
    ct.start()
    deadline = time.time() + 10
    while ct.error is None and time.time() < deadline:
        time.sleep(0.1)
    ct.stop()
    assert ct.error and "disconnected" in ct.error


def test_gpu_runtime_failure_falls_back_to_cpu(tmp_path):
    """A GPU provider rejecting an operator at run time (DirectML 'parameter is incorrect') must not crash:
    the model is rebuilt on CPU and the call retried."""
    import onnx
    from onnx import TensorProto, helper

    from facefusion_live.inference.session import SafeSession, Sessions
    g = helper.make_graph([helper.make_node("Add", ["x", "x"], ["y"])], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])])
    path = tmp_path / "add.onnx"
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8  # readable by every supported onnxruntime
    onnx.save(m, str(path))
    sessions = Sessions(["CPUExecutionProvider"])
    s = sessions.get(path)
    assert isinstance(s, SafeSession)

    class FailingGpu:
        def run(self, *_):
            raise RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Reshape_223 ... The parameter is incorrect.")

        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    s.inner = FailingGpu()
    x = np.array([[1, 2, 3]], np.float32)
    out = sessions.run(s, {"x": x})[0]
    assert out.tolist() == [[2, 4, 6]] and s.fell_back and not s.on_gpu
    assert sessions.run(s, {"x": x})[0].tolist() == [[2, 4, 6]]  # stays on CPU
