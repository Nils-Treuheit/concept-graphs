"""Contract tests for safe, fake-only RealSense recording."""

import logging
import json
import sys
from types import SimpleNamespace

import numpy as np

from conceptgraph.scripts.record_realsense_bag import make_recording_config


class FakeRS:
    class stream:
        depth = "depth"
        color = "color"

    class format:
        z16 = "z16"
        rgb8 = "rgb8"

    class config:
        def __init__(self):
            self.device = None
            self.streams = []
            self.record_path = None

        def enable_device(self, serial):
            self.device = serial

        def enable_record_to_file(self, path):
            self.record_path = path

        def enable_stream(self, *args):
            self.streams.append(args)


def test_recording_config_uses_matching_rgb8_and_z16_profiles_and_file(tmp_path):
    output = tmp_path / "capture.bag"

    config = make_recording_config(FakeRS, output, "serial-123", 1280, 720, 30)

    assert config.record_path == str(output)
    assert config.device == "serial-123"
    assert config.streams == [
        (FakeRS.stream.depth, 1280, 720, FakeRS.format.z16, 30),
        (FakeRS.stream.color, 1280, 720, FakeRS.format.rgb8, 30),
    ]


def test_recording_config_leaves_device_selection_to_pipeline_when_serial_is_none(tmp_path):
    config = make_recording_config(FakeRS, tmp_path / "capture.bag", None, 640, 360, 15)

    assert config.device is None
    assert [stream[1:] for stream in config.streams] == [
        (640, 360, FakeRS.format.z16, 15),
        (640, 360, FakeRS.format.rgb8, 15),
    ]


class _FakeStreamProfile:
    def width(self):
        return 1280

    def height(self):
        return 720

    def fps(self):
        return 30

    def as_video_stream_profile(self):
        return self


class _FakeProfile:
    def get_stream(self, _stream):
        return _FakeStreamProfile()

    def get_device(self):
        return SimpleNamespace(get_info=lambda _key: "fake-serial")


class _FakeFrame:
    def __init__(self, frame_number=1):
        self.frame_number = frame_number

    def get_frame_number(self):
        return self.frame_number

    def get_timestamp(self):
        return float(self.frame_number)

    def get_data(self):
        return np.zeros((720, 1280, 3), dtype=np.uint8)


class _FakeFrameSet:
    def get_color_frame(self):
        return _FakeFrame()

    def get_depth_frame(self):
        return _FakeFrame()


class _FakePipeline:
    def __init__(
        self, *, interrupt_once=False, stop_after_first_frame=False,
        interrupt_exception=KeyboardInterrupt,
    ):
        self.interrupt_once = interrupt_once
        self.interrupt_exception = interrupt_exception
        self.stop_after_first_frame = stop_after_first_frame
        self.frame_calls = 0
        self.started = 0
        self.stopped = 0

    def start(self, _config):
        self.started += 1
        record_path = getattr(_config, "record_path", None)
        if record_path is not None:
            from pathlib import Path

            path = Path(record_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake closed DB3")
        return _FakeProfile()

    def wait_for_frames(self, **_kwargs):
        if self.interrupt_once:
            self.interrupt_once = False
            raise self.interrupt_exception
        self.frame_calls += 1
        if self.stop_after_first_frame and self.frame_calls > 1:
            raise AssertionError("capture continued after the preview window closed")
        return _FakeFrameSet()

    def stop(self):
        self.stopped += 1


class _FakeConfig:
    def enable_device(self, _serial):
        pass

    def enable_stream(self, *_args):
        pass

    def enable_record_to_file(self, _path):
        pass


class _FakeOpenCV:
    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 0
    INTER_AREA = 0
    COLOR_RGB2BGR = 0
    FONT_HERSHEY_SIMPLEX = 0

    class _Writer:
        def isOpened(self):
            return True

        def write(self, _image):
            pass

        def release(self):
            pass

    def VideoWriter(self, *_args):
        return self._Writer()

    def VideoWriter_fourcc(self, *_args):
        return 0

    class error(Exception):
        pass

    def __init__(self, *, window_visible):
        self.window_visible = window_visible
        self.destroyed = []
        self.visibility_checks = 0

    def namedWindow(self, _name, _flags):
        pass

    def cvtColor(self, image, _code):
        return image

    def resize(self, image, _size, interpolation=None):
        return image

    def putText(self, image, *_args):
        return image

    def imshow(self, *_args):
        pass

    def waitKey(self, _delay):
        return -1

    def getWindowProperty(self, _name, _property):
        self.visibility_checks += 1
        return 1.0 if self.window_visible else 0.0

    def destroyWindow(self, name):
        self.destroyed.append(name)


class _FakeRecordingRS(FakeRS):
    class camera_info:
        serial_number = "serial_number"

    def __init__(self, pipeline):
        self.pipeline = lambda: pipeline


def _record_args(path):
    from argparse import Namespace

    return Namespace(
        output=str(path), serial=None, width=1280, height=720, fps=30,
        preview=True, duration_seconds=None,
    )


def _install_fake_rs(monkeypatch, pipeline):
    fake_rs = _FakeRecordingRS(pipeline)
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    from conceptgraph.scripts import record_realsense_bag as recorder

    def export_fake_recording(source, output_dir, **_kwargs):
        from pathlib import Path

        assert Path(source).is_file()
        archive = Path(output_dir)
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "review.mp4").write_bytes(b"fake review video")
        return {"verified": True, "verification": {"verified": True}}

    monkeypatch.setattr(recorder, "export_recording", export_fake_recording, raising=False)
    return fake_rs


def test_closing_preview_window_stops_and_finalizes_recording(monkeypatch, tmp_path, caplog):
    from conceptgraph.scripts import record_realsense_bag as recorder

    pipeline = _FakePipeline(stop_after_first_frame=True)
    _install_fake_rs(monkeypatch, pipeline)
    fake_cv2 = _FakeOpenCV(window_visible=False)
    monkeypatch.setattr(recorder, "cv2", fake_cv2)
    monkeypatch.setattr(recorder.os, "environ", {"DISPLAY": ":test"})

    with caplog.at_level(logging.INFO):
        result = recorder.record(_record_args(tmp_path / "closed-window.db3"))

    assert result == 0
    assert pipeline.started == 1
    assert pipeline.stopped == 1
    assert fake_cv2.visibility_checks == 1
    assert fake_cv2.destroyed == ["RealSense RGB-D recording"]
    assert "Recording finalized:" in caplog.text


def test_single_ctrl_c_finalizes_and_returns_normally(monkeypatch, tmp_path, caplog):
    from conceptgraph.scripts import record_realsense_bag as recorder

    class _CtrlC(Exception):
        """Test-safe stand-in for KeyboardInterrupt."""

    monkeypatch.setattr(recorder, "KeyboardInterrupt", _CtrlC, raising=False)
    class _CtrlCAfterFramePipeline(_FakePipeline):
        def __init__(self):
            super().__init__()
            self.did_interrupt = False

        def wait_for_frames(self, **kwargs):
            if self.frame_calls >= 1 and not self.did_interrupt:
                self.did_interrupt = True
                raise _CtrlC()
            return super().wait_for_frames(**kwargs)

    pipeline = _CtrlCAfterFramePipeline()
    _install_fake_rs(monkeypatch, pipeline)
    fake_cv2 = _FakeOpenCV(window_visible=True)
    monkeypatch.setattr(recorder, "cv2", fake_cv2)
    monkeypatch.setattr(recorder.os, "environ", {"DISPLAY": ":test"})

    with caplog.at_level(logging.INFO):
        result = recorder.record(_record_args(tmp_path / "ctrl-c.db3"))

    assert result == 0
    assert pipeline.started == 1
    assert pipeline.stopped == 1
    assert "Recording finalized:" in caplog.text


def test_manifest_and_temp_db3_survive_archive_export_failure(monkeypatch, tmp_path):
    """Archive-first capture retains source and session metadata if export fails."""
    import pytest
    from conceptgraph.scripts import record_realsense_bag as recorder

    pipeline = _FakePipeline(stop_after_first_frame=True)
    _install_fake_rs(monkeypatch, pipeline)
    monkeypatch.setattr(recorder, "cv2", _FakeOpenCV(window_visible=False))
    monkeypatch.setattr(recorder.os, "environ", {"DISPLAY": ":test"})

    class MetadataOnlySession(recorder.CaptureSession):
        def __init__(self, output_dir, fps, write_video=False):
            super().__init__(output_dir, fps=fps, write_video=write_video)
            assert write_video is False

    monkeypatch.setattr(
        recorder, "CaptureSession", MetadataOnlySession
    )
    output = tmp_path / "mp4-finalization-failure.db3"

    def failing_export(_source, _output_dir, **_kwargs):
        raise OSError("simulated archive export failure")

    monkeypatch.setattr(recorder, "export_recording", failing_export, raising=False)

    with pytest.raises(OSError, match="simulated archive export failure"):
        recorder.record(_record_args(output))

    assert (tmp_path / "capture_session.json").is_file()
    assert pipeline.stopped == 1
    temp_db3 = tmp_path / ".recording_tmp" / output.name
    assert temp_db3.is_file()
    assert json.loads((tmp_path / "capture_session.json").read_text())["recordings"]


def test_record_entrypoint_stops_then_closes_and_exports_verified_archive(
    monkeypatch, tmp_path
):
    """The production CLI path must wire SDK shutdown through archive finalization."""
    from pathlib import Path

    from conceptgraph.scripts import record_realsense_bag as recorder

    events = []

    class RecordingPipeline(_FakePipeline):
        def start(self, config):
            self.record_path = Path(config.record_path).resolve()
            return super().start(config)

        def stop(self):
            events.append("pipeline-stop")
            super().stop()

    class RecordingSession:
        def __init__(self, output_dir, fps, write_video=False):
            self.output_dir = Path(output_dir)
            self.fps = fps
            assert write_video is False
            self.recordings = []
            self.is_recording = False

        def start_recording(self, _frame, *, bag_path, video_path, recording_id=None):
            self.recordings.append({
                "recording_id": recording_id or "recording_001",
                "bag": Path(bag_path).name,
                "video": Path(video_path).name,
            })
            self.is_recording = True

        def write_frame(self, _frame):
            pass

        def stop_recording(self):
            events.append("session-finalize")
            self.is_recording = False

        def save(self):
            events.append("session-save")
            path = self.output_dir / "capture_session.json"
            path.write_text(json.dumps({"recordings": self.recordings}), encoding="utf-8")
            return str(path)

    pipeline = RecordingPipeline(stop_after_first_frame=True)
    _install_fake_rs(monkeypatch, pipeline)
    monkeypatch.setattr(recorder, "cv2", _FakeOpenCV(window_visible=False))
    monkeypatch.setattr(recorder.os, "environ", {"DISPLAY": ":test"})
    monkeypatch.setattr(recorder, "CaptureSession", RecordingSession)

    def export_recording(source, output_dir, **_kwargs):
        events.append("archive-export")
        assert Path(source) == pipeline.record_path
        assert Path(source).is_file()
        archive = Path(output_dir)
        assert archive == tmp_path / "recording_001_archive"
        archive.mkdir()
        (archive / "review.mp4").write_bytes(b"verified review video")
        return {"verified": True, "verification": {"verified": True}}

    monkeypatch.setattr(recorder, "export_recording", export_recording, raising=False)
    output = tmp_path / "recording_001.db3"
    from argparse import Namespace

    result = recorder.record(Namespace(
        output_dir=str(tmp_path), serial=None, width=1280, height=720, fps=30,
        preview=True, duration_seconds=None,
    ))

    assert result == 0
    assert events.index("pipeline-stop") < events.index("session-finalize") < events.index("archive-export")
    assert pipeline.record_path.name == output.name
    assert pipeline.record_path.parent != tmp_path
    assert pipeline.record_path.parent.name.startswith(".")
    assert not pipeline.record_path.exists()
    archive = tmp_path / "recording_001_archive"
    assert (archive / "review.mp4").is_file()
    saved = json.loads((tmp_path / "capture_session.json").read_text(encoding="utf-8"))
    recording = saved["recordings"][0]
    assert recording["recording_id"] == "recording_001"
    assert recording["archive"] == archive.name
    assert recording["video"] == f"{archive.name}/review.mp4"
    assert "bag" not in recording


def test_record_entrypoint_keeps_db3_when_archive_is_not_verified(monkeypatch, tmp_path):
    """An exporter result without both verification flags cannot discard source data."""
    import pytest
    from pathlib import Path

    from conceptgraph.scripts import record_realsense_bag as recorder

    class RecordingPipeline(_FakePipeline):
        def start(self, config):
            self.record_path = Path(config.record_path).resolve()
            return super().start(config)

    class RecordingSession:
        def __init__(self, output_dir, fps, write_video=False):
            self.output_dir = Path(output_dir)
            assert write_video is False
            self.recordings = []
            self.is_recording = False

        def start_recording(self, _frame, *, bag_path, video_path, recording_id=None):
            self.recordings.append({
                "recording_id": recording_id or "recording_001",
                "bag": Path(bag_path).name,
                "video": Path(video_path).name,
            })
            self.is_recording = True

        def write_frame(self, _frame):
            pass

        def stop_recording(self):
            self.is_recording = False

        def save(self):
            path = self.output_dir / "capture_session.json"
            path.write_text(json.dumps({"recordings": []}), encoding="utf-8")
            return str(path)

    pipeline = RecordingPipeline(stop_after_first_frame=True)
    _install_fake_rs(monkeypatch, pipeline)
    monkeypatch.setattr(recorder, "cv2", _FakeOpenCV(window_visible=False))
    monkeypatch.setattr(recorder.os, "environ", {"DISPLAY": ":test"})
    monkeypatch.setattr(recorder, "CaptureSession", RecordingSession)

    def unverified_export(_source, output_dir, **_kwargs):
        Path(output_dir).mkdir()
        return {"verified": False, "verification": {"verified": False}}

    monkeypatch.setattr(recorder, "export_recording", unverified_export, raising=False)
    output = tmp_path / "recording_001.db3"
    from argparse import Namespace

    with pytest.raises((RuntimeError, ValueError)):
        recorder.record(Namespace(
            output_dir=str(tmp_path), serial=None, width=1280, height=720, fps=30,
            preview=True, duration_seconds=None,
        ))

    assert pipeline.record_path.name == output.name
    assert pipeline.record_path.parent != tmp_path
    assert pipeline.record_path.parent.name.startswith(".")
    assert pipeline.record_path.read_bytes() == b"fake closed DB3"
