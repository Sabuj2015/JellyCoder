import io
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import pytest

from video_reducer import core


def make_media_info(**overrides: object) -> core.MediaInfo:
    defaults = dict(
        frames=120,
        duration=10.0,
        bitrate_kbps=4000.0,
        width=1920,
        height=1080,
        audio_codec="aac",
        audio_bitrate_kbps=192.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    defaults.update(overrides)
    return core.MediaInfo(**defaults)


# ensure_ffmpeg_available

def test_ensure_ffmpeg_available_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core.shutil, "which", lambda _: "ffmpeg")
    core.ensure_ffmpeg_available()


def test_ensure_ffmpeg_available_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError):
        core.ensure_ffmpeg_available()


# select_nvenc_encoder

class FakeCompletedProcess:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def fake_run_factory(stdout: str) -> Callable[..., FakeCompletedProcess]:
    def _runner(*_: object, **__: object) -> FakeCompletedProcess:
        return FakeCompletedProcess(stdout)
    return _runner


def test_select_nvenc_encoder_default(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "\n V..... hevc_nvenc\n V..... h264_nvenc\n"
    monkeypatch.setattr(core.subprocess, "run", fake_run_factory(output))
    selection = core.select_nvenc_encoder()
    assert selection.encoder == "h264_nvenc"
    assert selection.output_extension == ".mp4"


def test_select_nvenc_encoder_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "\n V..... h264_nvenc\n V..... hevc_nvenc\n"
    monkeypatch.setattr(core.subprocess, "run", fake_run_factory(output))
    selection = core.select_nvenc_encoder("hevc")
    assert selection.encoder == "hevc_nvenc"
    assert selection.output_extension == ".mkv"


def test_select_nvenc_encoder_invalid_preference() -> None:
    with pytest.raises(ValueError):
        core.select_nvenc_encoder("vp9")


def test_select_nvenc_encoder_preferred_missing(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    output = "\n V..... h264_nvenc\n"
    monkeypatch.setattr(core.subprocess, "run", fake_run_factory(output))
    with caplog.at_level(logging.WARNING):
        selection = core.select_nvenc_encoder("hevc")
    assert selection.encoder == "h264_nvenc"


def test_select_nvenc_encoder_preferred_warning_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    output = "\n V..... libx264\n"
    monkeypatch.setattr(core.subprocess, "run", fake_run_factory(output))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            core.select_nvenc_encoder("hevc")
    assert any("not available" in message for message in caplog.messages)


def test_select_nvenc_encoder_run_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _runner(*_: object, **__: object) -> FakeCompletedProcess:
        raise subprocess.CalledProcessError(1, ["ffmpeg"])

    monkeypatch.setattr(core.subprocess, "run", _runner)
    with pytest.raises(RuntimeError):
        core.select_nvenc_encoder()


def test_select_nvenc_encoder_no_nvenc(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "\n V..... libx264\n"
    monkeypatch.setattr(core.subprocess, "run", fake_run_factory(output))
    with pytest.raises(RuntimeError):
        core.select_nvenc_encoder()


# discover_videos

def test_discover_videos_with_ignore(tmp_path: Path) -> None:
    base = tmp_path / "videos"
    ignore = base / "ignore"
    nested = ignore / "nested"
    nested.mkdir(parents=True)
    allowed = base / "movie.mkv"
    allowed.touch()
    skipped = ignore / "skip.mp4"
    skipped.touch()
    nested_file = nested / "skip2.mkv"
    nested_file.touch()
    other = base / "clip.avi"
    other.touch()

    found = core.discover_videos(base, ignore)
    assert found == [allowed]


def test_discover_videos_without_ignore(tmp_path: Path) -> None:
    base = tmp_path / "videos"
    base.mkdir()
    file1 = base / "a.MKV"
    file1.touch()
    sub = base / "sub"
    sub.mkdir()
    file2 = sub / "b.mp4"
    file2.touch()

    found = core.discover_videos(base, None)
    assert found == [file1.resolve(), file2.resolve()]


def test_discover_videos_ignore_root(tmp_path: Path) -> None:
    base = tmp_path / "videos"
    base.mkdir()
    (base / "movie.mkv").touch()
    assert core.discover_videos(base, base) == []


def test_discover_videos_handles_nonexistent_ignore(tmp_path: Path) -> None:
    base = tmp_path / "videos"
    base.mkdir()
    file_path = base / "movie.mp4"
    file_path.touch()
    assert core.discover_videos(base, base / "missing") == [file_path]


# format_size

def test_format_size_units() -> None:
    assert core.format_size(500) == "500.00 B"
    assert core.format_size(2048) == "2.00 KB"
    huge = 1024 ** 5 * 5
    assert core.format_size(huge).endswith("PB")


# build_output_path

def test_build_output_path_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "video.mkv"
    result = core.build_output_path(src, tmp_path, True, None, ".mp4")
    assert result == src.with_suffix(".mp4")


def test_build_output_path_output_root(tmp_path: Path) -> None:
    src = tmp_path / "parent" / "video.mkv"
    src.parent.mkdir()
    output_root = tmp_path / "output"
    result = core.build_output_path(src, tmp_path, False, output_root, ".mp4")
    expected = output_root / "parent" / "video.mp4"
    assert result == expected


# probe_media_info

def test_probe_media_info_ffprobe_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing(*_: object, **__: object) -> FakeCompletedProcess:
        raise FileNotFoundError()

    monkeypatch.setattr(core.subprocess, "run", missing)
    info = core.probe_media_info(tmp_path / "missing.mkv")
    assert info.frames is None
    assert info.duration is None


def test_probe_media_info_called_process_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _runner(*_: object, **__: object) -> FakeCompletedProcess:
        raise subprocess.CalledProcessError(1, ["ffprobe"], stderr="boom")

    monkeypatch.setattr(core.subprocess, "run", _runner)
    info = core.probe_media_info(tmp_path / "bad.mkv")
    assert info.audio_codec is None


def test_probe_media_info_invalid_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(core.subprocess, "run", fake_run_factory("not-json"))
    info = core.probe_media_info(tmp_path / "broken.mkv")
    assert info.subtitle_codecs == []


def test_probe_media_info_parses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "nb_frames": None,
                "duration": "10.0",
                "avg_frame_rate": "30000/1001",
                "width": 1920,
                "height": 1080,
            },
            {
                "codec_type": "video",
                "codec_name": "mjpeg",
                "disposition": {"attached_pic": 1},
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "bit_rate": "192000",
            },
            {
                "codec_type": "subtitle",
                "codec_name": "srt",
            },
            {
                "codec_type": "data",
                "codec_name": "bin",
            },
        ],
        "format": {
            "bit_rate": "4000000",
            "duration": "10.0",
        },
    }
    monkeypatch.setattr(core.subprocess, "run", fake_run_factory(json.dumps(data)))
    info = core.probe_media_info(tmp_path / "good.mkv")
    assert info.frames == 299
    assert info.audio_codec == "aac"
    assert "mjpeg" in info.attached_pic_codecs
    assert info.subtitle_codecs == ["srt"]
    assert info.data_stream_codecs == ["bin"]


def test_probe_media_info_duration_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "nb_frames": "240",
                "width": 1280,
                "height": 720,
            }
        ],
        "format": {
            "bit_rate": "1000000",
            "duration": "8.0",
        },
    }
    monkeypatch.setattr(core.subprocess, "run", fake_run_factory(json.dumps(data)))
    info = core.probe_media_info(tmp_path / "fallback.mkv")
    assert info.duration == 8.0


# progress helpers

def test_format_progress_bar_bounds() -> None:
    bar = core._format_progress_bar(None)
    assert bar.startswith("[") and "0.0%" in bar
    over = core._format_progress_bar(2.0)
    assert "100.0%" in over


def test_render_progress_line_variants() -> None:
    line, ratio = core._render_progress_line(5, 10, None, None)
    assert "5/10" in line and ratio == 0.5
    line, ratio = core._render_progress_line(None, None, 5.0, 10.0)
    assert "time" in line and ratio == 0.5
    line, ratio = core._render_progress_line(3, None, None, None)
    assert "frame 3" in line and ratio is None
    line, ratio = core._render_progress_line(None, None, 2.0, None)
    assert "time 2.0" in line and ratio is None
    line, ratio = core._render_progress_line(None, None, None, None)
    assert "encoding" in line and ratio is None


def test_display_and_clear_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    line = core._display_progress("testing")
    assert "testing" in buffer.getvalue()
    core._clear_progress(line)
    assert buffer.getvalue().endswith("\r")
    core._clear_progress("")  # no crash on empty


# run_ffmpeg_with_progress

class FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)
        self.closed = False

    def __iter__(self) -> "FakeStdout":
        return self

    def __next__(self) -> str:
        return next(self._lines)

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, lines: list[str], wait_code: int, stderr_text: str) -> None:
        self.stdout = FakeStdout(lines)
        self.stderr = io.StringIO(stderr_text)
        self._wait_code = wait_code

    def wait(self) -> int:
        return self._wait_code


def test_run_ffmpeg_with_progress_success(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        "frame=1",
        "out_time_ms=500000",
        "progress=continue",
        "progress=end",
    ]
    process = FakeProcess(lines, 0, "")
    monkeypatch.setattr(core.subprocess, "Popen", lambda *args, **kwargs: process)
    times = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(time, "time", lambda: next(times, 3.0))
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    core.run_ffmpeg_with_progress(["ffmpeg"], 10, 10.0)
    assert "frame" in buffer.getvalue()
    assert process.stdout.closed


def test_run_ffmpeg_with_progress_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = ["frame=5", "progress=end"]
    process = FakeProcess(lines, 1, "fatal")
    monkeypatch.setattr(core.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(time, "time", lambda: 0.0)
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    with pytest.raises(subprocess.CalledProcessError) as exc:
        core.run_ffmpeg_with_progress(["ffmpeg"], None, None)
    assert "fatal" in exc.value.stderr


# encode_video

def test_encode_video_mp4_copy_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "input.mkv"
    src.write_bytes(b"x" * 200)
    dst = tmp_path / "output.mp4"

    info = make_media_info(
        subtitle_codecs=["srt"],
        video_codecs=["h264"],
    )

    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"y" * 100)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    with caplog.at_level(logging.WARNING):
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="hevc_nvenc",
            output_extension=".mp4",
            quality="Unknown",
        )
    assert dst.exists()
    assert src.exists()
    assert any("Unknown quality preset" in message for message in caplog.messages)
    recorded = commands[0]
    assert "-c:a" in recorded and "copy" in recorded
    assert "mov_text" in recorded


def test_encode_video_switches_to_mkv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "movie.mkv"
    src.write_bytes(b"a" * 100)
    dst = tmp_path / "out" / "movie.mp4"
    info = make_media_info(
        subtitle_codecs=["pgs"],
        video_codecs=["h264", "hevc"],
        attached_pic_codecs=["gif"],
        data_stream_codecs=["bin"],
        bitrate_kbps=4000.0,
        height=2160,
        frames=None,
        duration=12.0,
        audio_codec="dts",
        audio_bitrate_kbps=None,
    )

    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    def fake_run(cmd: list[str], *_: object) -> None:
        Path(cmd[-1]).write_bytes(b"z" * 200)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    with caplog.at_level(logging.WARNING):
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="720p",
        )
    final_path = dst.with_suffix(".mkv")
    assert final_path.exists()
    assert src.exists()
    assert any("MP4 container not suitable" in message for message in caplog.messages)
    assert final_path.stat().st_size > src.stat().st_size


def test_encode_video_transcodes_audio_when_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "audio.mkv"
    src.write_bytes(b"m" * 150)
    dst = tmp_path / "audio.mp4"
    info = make_media_info(
        audio_codec="flac",
        audio_bitrate_kbps=512.0,
        subtitle_codecs=[],
        bitrate_kbps=None,
        frames=None,
        duration=8.0,
    )

    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"t" * 60)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    with caplog.at_level(logging.INFO):
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="auto",
        )
    assert dst.exists()
    assert any("transcoded to AAC" in message for message in caplog.messages)
    command = commands[0]
    b_index = command.index("-b:v")
    assert command[b_index + 1] == "0"


def test_encode_video_overwrite_with_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "clip.mkv"
    src.write_bytes(b"s" * 150)
    dst = src.with_suffix(".mp4")
    temp_dst = dst.with_name(f"_{dst.name}")
    temp_dst.write_bytes(b"temp")
    info = make_media_info(frames=50, duration=5.0)

    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    call_count = {"count": 0}
    original_unlink = Path.unlink

    def fake_run(cmd: list[str], *_: object) -> None:
        call_count["count"] += 1
        target = Path(cmd[-1])
        target.write_bytes(b"p" * 80)
        if call_count["count"] == 1:
            raise subprocess.CalledProcessError(1, cmd, output="progress", stderr="err")

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    def noisy_unlink(self: Path) -> None:
        if self == temp_dst and not getattr(noisy_unlink, "raised", False):
            noisy_unlink.raised = True
            raise OSError("busy")
        original_unlink(self)

    noisy_unlink.raised = False  # type: ignore[attr-defined]

    monkeypatch.setattr(Path, "unlink", noisy_unlink)

    core.encode_video(
        src=src,
        dst=dst,
        overwrite=True,
        encoder="hevc_nvenc",
        output_extension=".mp4",
        quality="auto",
    )
    assert call_count["count"] == 2
    assert not src.exists()
    assert dst.exists()


def test_encode_video_all_attempts_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "broken.mkv"
    src.write_bytes(b"s" * 100)
    dst = tmp_path / "broken.mp4"
    info = make_media_info()

    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    def fake_run(cmd: list[str], *_: object) -> None:
        raise subprocess.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    with pytest.raises(RuntimeError) as exc:
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="hevc_nvenc",
            output_extension=".mp4",
            quality="auto",
        )
    assert "ffmpeg failed" in str(exc.value)


# process_videos

def test_process_videos_handles_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    base = tmp_path / "input"
    base.mkdir()
    ok = base / "ok.mkv"
    ok.touch()
    fail = base / "fail.mkv"
    fail.touch()
    output_root = tmp_path / "out"

    calls: list[tuple[Path, Path]] = []

    def fake_encode(*, src: Path, dst: Path, **_: object) -> None:
        calls.append((src, dst))
        if src.name.startswith("fail"):
            raise ValueError("boom")

    monkeypatch.setattr(core, "encode_video", fake_encode)

    with caplog.at_level(logging.ERROR):
        core.process_videos(
            videos=[ok, fail],
            base_input=base,
            overwrite=False,
            output_root=output_root,
            encoder="h264_nvenc",
            output_extension=".mp4",
            max_workers=2,
            quality="auto",
        )
    assert len(calls) == 2
    assert any("Failed" in message for message in caplog.messages)


# reduce_videos

def test_reduce_videos_invalid_input(tmp_path: Path) -> None:
    config = core.ReducerConfig(input_path=tmp_path / "missing")
    with pytest.raises(ValueError):
        core.reduce_videos(config)


def test_reduce_videos_no_videos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "input"
    base.mkdir()
    config = core.ReducerConfig(input_path=base)

    monkeypatch.setattr(core, "ensure_ffmpeg_available", lambda: None)
    monkeypatch.setattr(core, "select_nvenc_encoder", lambda *_: core.NvencSelection("h264_nvenc", ".mp4"))
    monkeypatch.setattr(core, "discover_videos", lambda *_: [])
    called = {"process": False}
    monkeypatch.setattr(core, "process_videos", lambda **_: called.__setitem__("process", True))

    core.reduce_videos(config)
    assert not called["process"]


def test_reduce_videos_runs_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "input"
    base.mkdir()
    video = base / "movie.mkv"
    video.touch()
    config = core.ReducerConfig(input_path=base, overwrite=False, max_workers=3, quality="1080p")

    monkeypatch.setattr(core, "ensure_ffmpeg_available", lambda: None)
    monkeypatch.setattr(core, "select_nvenc_encoder", lambda *_: core.NvencSelection("hevc_nvenc", ".mkv"))
    monkeypatch.setattr(core, "discover_videos", lambda *_: [video])

    recorded = {}

    def capture_process(**kwargs: object) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(core, "process_videos", capture_process)

    core.reduce_videos(config)
    assert recorded["videos"] == [video]
    assert recorded["max_workers"] == 3
    assert recorded["quality"] == "1080p"
    assert recorded["output_extension"] == ".mkv"
    assert recorded["output_root"].exists()


def test_reduce_videos_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "input"
    base.mkdir()
    video = base / "movie.mkv"
    video.touch()
    config = core.ReducerConfig(input_path=base, overwrite=True)

    monkeypatch.setattr(core, "ensure_ffmpeg_available", lambda: None)
    monkeypatch.setattr(core, "select_nvenc_encoder", lambda *_: core.NvencSelection("h264_nvenc", ".mp4"))
    monkeypatch.setattr(core, "discover_videos", lambda *_: [video])

    captured = {}

    def capture_process(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(core, "process_videos", capture_process)

    core.reduce_videos(config)
    assert captured["output_root"] is None
    assert captured["overwrite"] is True
