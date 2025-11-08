import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from video_reducer import core


class DummyCompletedProcess:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def runner_with_output(stdout: str):
    def _run(*_: object, **__: object) -> DummyCompletedProcess:
        return DummyCompletedProcess(stdout)
    return _run


def test_select_nvenc_encoder_skips_noise_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "\n  ---- encoders ----\n A..... thing\n V\n S..... other\n V..... hevc_nvenc\n"
    monkeypatch.setattr(core.subprocess, "run", runner_with_output(output))
    selection = core.select_nvenc_encoder()
    assert selection.encoder == "hevc_nvenc"
    assert selection.output_extension == ".mkv"


def test_probe_media_info_handles_value_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "nb_frames": "NaN",
                "duration": "oops",
                "avg_frame_rate": "bad/value",
                "width": 1920,
                "height": 1080,
            },
            {
                "codec_type": "video",
                "codec_name": "h265",
                "nb_frames": "missing",
                "duration": "5",
                "avg_frame_rate": "120/0",
            },
            {
                "codec_type": "video",
                "codec_name": "av1",
                "nb_frames": "missing",
                "duration": "4",
                "avg_frame_rate": "24/1",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "bit_rate": "bad",
            },
            {
                "codec_type": "subtitle",
                "codec_name": "srt",
            },
            {
                "codec_type": "video",
                "codec_name": "mjpeg",
                "disposition": {"attached_pic": 1},
            },
            {
                "codec_type": "data",
                "codec_name": "bin",
            },
            {
                "codec_type": "video",
                "codec_name": "vp8",
                "duration": 3,
                "avg_frame_rate": "bad/value",
            },
        ],
        "format": {
            "bit_rate": "invalid",
            "duration": "??",
        },
    }
    monkeypatch.setattr(core.subprocess, "run", runner_with_output(json.dumps(data)))
    info = core.probe_media_info(tmp_path / "values.mkv")
    assert info.frames == 96
    assert info.duration == 4.0
    assert info.audio_bitrate_kbps is None
    assert info.bitrate_kbps is None
    assert "srt" in info.subtitle_codecs
    assert "mjpeg" in info.attached_pic_codecs
    assert "bin" in info.data_stream_codecs


def test_probe_media_info_handles_sparse_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": None,
                "nb_frames": "42",
                "duration": "6",
                "avg_frame_rate": None,
                "width": "not-int",
                "height": "not-int",
                "disposition": {"attached_pic": 1},
            },
            {
                "codec_type": "video",
                "codec_name": "vp9",
                "nb_frames": "none",
                "duration": "7",
                "avg_frame_rate": "120/0",
                "width": 640,
                "height": 360,
            },
            {
                "codec_type": "audio",
                "codec_name": 123,
                "bit_rate": None,
            },
            {
                "codec_type": "subtitle",
                "codec_name": None,
            },
            {
                "codec_type": "data",
                "codec_name": None,
            },
        ],
        "format": {
            "bit_rate": None,
            "duration": None,
        },
    }
    monkeypatch.setattr(core.subprocess, "run", runner_with_output(json.dumps(data)))
    info = core.probe_media_info(tmp_path / "sparse.mkv")
    assert info.frames == 42
    assert info.width == 640 and info.height == 360
    assert info.audio_codec is None
    assert info.bitrate_kbps is None


def test_probe_media_info_format_duration_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "av1",
                "nb_frames": None,
                "duration": None,
                "avg_frame_rate": "30/1",
            }
        ],
        "format": {
            "bit_rate": "bad",
            "duration": "oops",
        },
    }
    monkeypatch.setattr(core.subprocess, "run", runner_with_output(json.dumps(data)))
    info = core.probe_media_info(tmp_path / "format.mkv")
    assert info.duration is None


def test_probe_media_info_zero_fps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = {
        "streams": [
            {
                "codec_type": "video",
                "duration": 8,
                "avg_frame_rate": "0/3",
            },
            {
                "codec_type": "video",
                "duration": 4,
                "avg_frame_rate": None,
            },
        ],
        "format": {
            "bit_rate": 1_000_000,
        },
    }
    monkeypatch.setattr(core.subprocess, "run", runner_with_output(json.dumps(data)))
    info = core.probe_media_info(tmp_path / "zero.mkv")
    assert info.frames is None
    assert info.duration == 4.0
    assert info.bitrate_kbps == 1000.0


class _IteratorStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)
        self.closed = False

    def __iter__(self) -> "_IteratorStdout":
        return self

    def __next__(self) -> str:
        return next(self._lines)

    def close(self) -> None:
        self.closed = True


class _StubProcess:
    def __init__(self, lines: list[str]) -> None:
        self.stdout = _IteratorStdout(lines)
        self.stderr = io.StringIO("")

    def wait(self) -> int:
        return 0


def test_run_ffmpeg_with_progress_tolerates_bad_values(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        "frame=abc",
        "out_time_ms=oops",
        "progress=continue",
        "progress=end",
    ]
    process = _StubProcess(lines)
    monkeypatch.setattr(core.subprocess, "Popen", lambda *args, **kwargs: process)
    times = iter([0.0, 0.5, 1.1, 1.6])
    monkeypatch.setattr(time, "time", lambda: next(times, 2.0))
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    core.run_ffmpeg_with_progress(["ffmpeg"], total_frames=None, total_duration=None)
    assert process.stdout.closed


def test_run_ffmpeg_with_progress_handles_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FalseyStdout:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self) -> "FalseyStdout":
            return self

        def __next__(self) -> str:
            raise StopIteration

        def __bool__(self) -> bool:
            return False

        def close(self) -> None:
            self.closed = True


    class FalseyStderr:
        def __bool__(self) -> bool:
            return False

        def read(self) -> str:
            raise AssertionError("stderr should not be read")

        def close(self) -> None:
            raise AssertionError("stderr should not be closed")


    class QuietProcess:
        def __init__(self) -> None:
            self.stdout = FalseyStdout()
            self.stderr = FalseyStderr()

        def wait(self) -> int:
            return 0


    monkeypatch.setattr(core.subprocess, "Popen", lambda *args, **kwargs: QuietProcess())
    monkeypatch.setattr(time, "time", lambda: 0.0)
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    core.run_ffmpeg_with_progress(["ffmpeg"], total_frames=None, total_duration=None)
    assert buffer.getvalue() == "\n"


def test_encode_video_resets_target_bitrate_when_ratio_vanishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "tiny.mkv"
    src.write_bytes(b"0" * 100)
    dst = tmp_path / "tiny.mp4"
    info = core.MediaInfo(
        frames=10,
        duration=1.0,
        bitrate_kbps=1.0,
        width=640,
        height=480,
        audio_codec="aac",
        audio_bitrate_kbps=192.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)
    monkeypatch.setattr(core, "MIN_TARGET_BITRATE_KBPS", 0)

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"1" * 50)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    core.encode_video(
        src=src,
        dst=dst,
        overwrite=False,
        encoder="hevc_nvenc",
        output_extension=".mp4",
        quality="auto",
    )
    encoded = commands[0]
    idx = encoded.index("-b:v")
    assert encoded[idx + 1] == "0"


def test_encode_video_scaling_adjusts_bitrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "scale.mkv"
    src.write_bytes(b"x" * 500)
    dst = tmp_path / "scale.mp4"
    info = core.MediaInfo(
        frames=300,
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
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"y" * 200)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    with caplog.at_level(logging.DEBUG):
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="720p",
        )
    encoded = commands[0]
    assert "-vf" in encoded and "scale=-2:720" in encoded
    b_index = encoded.index("-b:v")
    assert encoded[b_index + 1].endswith("k")
    assert any("Adjusting target bitrate" in msg for msg in caplog.messages)


def test_encode_video_skips_when_destination_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "skip.mkv"
    src.write_bytes(b"q" * 100)
    dst = tmp_path / "skip.mp4"
    dst.write_bytes(b"existing")
    info = core.MediaInfo(
        frames=0,
        duration=None,
        bitrate_kbps=None,
        width=None,
        height=None,
        audio_codec=None,
        audio_bitrate_kbps=None,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    called = {"ran": False}

    def fail_run(*_: object, **__: object) -> None:
        called["ran"] = True

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fail_run)

    with caplog.at_level(logging.INFO):
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="auto",
        )
    assert called["ran"] is False
    assert any("Skipping existing file" in msg for msg in caplog.messages)


def test_encode_video_height_unknown_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "unknown.mkv"
    src.write_bytes(b"x" * 400)
    dst = tmp_path / "unknown.mp4"
    info = core.MediaInfo(
        frames=120,
        duration=5.0,
        bitrate_kbps=6000.0,
        width=1920,
        height=None,
        audio_codec="aac",
        audio_bitrate_kbps=128.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    def fake_run(cmd: list[str], *_: object) -> None:
        Path(cmd[-1]).write_bytes(b"z" * 100)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    with caplog.at_level(logging.INFO):
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="480p",
        )
    assert any("source height unknown" in msg for msg in caplog.messages)


def test_encode_video_height_no_scale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "flat.mkv"
    src.write_bytes(b"m" * 300)
    dst = tmp_path / "flat.mp4"
    info = core.MediaInfo(
        frames=90,
        duration=3.0,
        bitrate_kbps=1200.0,
        width=854,
        height=360,
        audio_codec="aac",
        audio_bitrate_kbps=96.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    def fake_run(cmd: list[str], *_: object) -> None:
        Path(cmd[-1]).write_bytes(b"n" * 100)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    with caplog.at_level(logging.INFO):
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="720p",
        )
    assert any("already <= target" in msg for msg in caplog.messages)


def test_encode_video_handles_missing_audio_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "mute.mkv"
    src.write_bytes(b"r" * 120)
    dst = tmp_path / "mute.mp4"
    info = core.MediaInfo(
        frames=None,
        duration=4.0,
        bitrate_kbps=1000.0,
        width=1280,
        height=720,
        audio_codec=None,
        audio_bitrate_kbps=None,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    def fake_run(cmd: list[str], *_: object) -> None:
        Path(cmd[-1]).write_bytes(b"s" * 60)

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
    assert any("Audio stream will be transcoded" in msg for msg in caplog.messages)


def test_encode_video_missing_output_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "vanish.mkv"
    src.write_bytes(b"d" * 200)
    dst = tmp_path / "vanish.mp4"
    info = core.MediaInfo(
        frames=60,
        duration=2.0,
        bitrate_kbps=2000.0,
        width=1280,
        height=720,
        audio_codec="aac",
        audio_bitrate_kbps=160.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    def fake_run(cmd: list[str], *_: object) -> None:
        target = Path(cmd[-1])
        if target.exists():
            target.unlink()

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    original_stat = Path.stat

    def missing_stat(self: Path) -> os.stat_result:  # type: ignore[override]
        if self.name.startswith("_") or self.name.endswith(".mp4"):
            raise FileNotFoundError
        return original_stat(self)

    monkeypatch.setattr(Path, "stat", missing_stat)

    with pytest.raises(RuntimeError) as exc:
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=True,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="auto",
        )
    assert "Expected output file missing" in str(exc.value)


def test_encode_video_final_attempt_runtime_without_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "broken.mkv"
    src.write_bytes(b"e" * 200)
    dst = tmp_path / "broken.mp4"
    info = core.MediaInfo(
        frames=120,
        duration=5.0,
        bitrate_kbps=3000.0,
        width=1920,
        height=1080,
        audio_codec="aac",
        audio_bitrate_kbps=128.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    def always_fail(cmd: list[str], *_: object) -> None:
        raise subprocess.CalledProcessError(1, cmd, output="trace", stderr="nope")

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", always_fail)

    with pytest.raises(RuntimeError) as exc:
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="auto",
        )
    message = str(exc.value)
    assert "STDOUT" in message
    assert "trace" in message


def test_encode_video_final_attempt_runtime_with_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "loud.mkv"
    src.write_bytes(b"e" * 220)
    dst = tmp_path / "loud.mp4"
    info = core.MediaInfo(
        frames=150,
        duration=6.0,
        bitrate_kbps=3200.0,
        width=1920,
        height=1080,
        audio_codec="aac",
        audio_bitrate_kbps=160.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    class WithStdoutError(subprocess.CalledProcessError):
        def __init__(self, cmd: list[str]) -> None:
            super().__init__(1, cmd, stderr="fail")
            self.stdout = "failing output"

    call_state = {"attempt": 0}

    def always_fail(cmd: list[str], *_: object) -> None:
        call_state["attempt"] += 1
        Path(cmd[-1]).write_bytes(b"fail")
        raise WithStdoutError(cmd)

    original_unlink = Path.unlink

    def flaky_unlink(self: Path) -> None:  # type: ignore[override]
        if self == dst and call_state["attempt"] >= 3:
            raise OSError("locked")
        original_unlink(self)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", always_fail)
    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    with pytest.raises(RuntimeError) as exc:
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="auto",
        )
    assert "failing output" in str(exc.value)


def test_encode_video_optional_nvenc_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "flags.mkv"
    src.write_bytes(b"f" * 140)
    dst = tmp_path / "flags.mp4"
    info = core.MediaInfo(
        frames=10,
        duration=1.0,
        bitrate_kbps=800.0,
        width=640,
        height=480,
        audio_codec="aac",
        audio_bitrate_kbps=96.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)
    monkeypatch.setattr(core, "NVENC_PROFILE_MAP", {})
    monkeypatch.setattr(core, "NVENC_TUNE", "")
    monkeypatch.setattr(core, "NVENC_LOOKAHEAD", "")
    monkeypatch.setattr(core, "NVENC_SPATIAL_AQ", "")
    monkeypatch.setattr(core, "NVENC_TEMPORAL_AQ", "")

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"g" * 80)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    core.encode_video(
        src=src,
        dst=dst,
        overwrite=False,
        encoder="mystery_nvenc",
        output_extension=".mp4",
        quality="auto",
    )
    encoded = commands[0]
    assert "-profile:v" not in encoded
    assert "-tune" not in encoded
    assert "-rc-lookahead" not in encoded
    assert "-spatial_aq" not in encoded
    assert "-temporal_aq" not in encoded


def test_encode_video_non_mp4_skips_mp4_logic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "direct.mkv"
    src.write_bytes(b"h" * 180)
    dst = tmp_path / "out" / "direct.mkv"
    info = core.MediaInfo(
        frames=60,
        duration=2.0,
        bitrate_kbps=2000.0,
        width=1280,
        height=720,
        audio_codec="aac",
        audio_bitrate_kbps=128.0,
        subtitle_codecs=["srt"],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"i" * 90)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    core.encode_video(
        src=src,
        dst=dst,
        overwrite=False,
        encoder="h264_nvenc",
        output_extension=".mkv",
        quality="auto",
    )
    encoded = commands[0]
    assert "-movflags" not in encoded
    assert encoded[-1].endswith(".mkv")


def test_encode_video_scaling_adjustment_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "steady.mkv"
    src.write_bytes(b"j" * 240)
    dst = tmp_path / "steady.mp4"
    info = core.MediaInfo(
        frames=180,
        duration=6.0,
        bitrate_kbps=4000.0,
        width=1920,
        height=1080,
        audio_codec="aac",
        audio_bitrate_kbps=160.0,
        subtitle_codecs=[],
        video_codecs=["hevc"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)
    monkeypatch.setattr(core, "HEVC_BITRATE_RATIO", 0.1)
    monkeypatch.setattr(core, "MIN_TARGET_BITRATE_KBPS", 400)

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"k" * 120)

    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    with caplog.at_level(logging.DEBUG):
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="hevc_nvenc",
            output_extension=".mp4",
            quality="720p",
        )
    command = commands[0]
    bitrate_index = command.index("-b:v")
    assert command[bitrate_index + 1] == "400k"
    assert not any("Adjusting target bitrate" in message for message in caplog.messages)


def test_encode_video_logs_duration_when_frames_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "timed.mkv"
    src.write_bytes(b"l" * 260)
    dst = tmp_path / "timed.mp4"
    info = core.MediaInfo(
        frames=None,
        duration=9.0,
        bitrate_kbps=2500.0,
        width=1280,
        height=720,
        audio_codec="aac",
        audio_bitrate_kbps=128.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    def fake_run(cmd: list[str], *_: object) -> None:
        Path(cmd[-1]).write_bytes(b"m" * 140)

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
    assert any("Estimated duration" in message for message in caplog.messages)


def test_encode_video_fallback_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "retry.mkv"
    src.write_bytes(b"n" * 320)
    dst = tmp_path / "retry.mp4"
    info = core.MediaInfo(
        frames=None,
        duration=8.0,
        bitrate_kbps=2800.0,
        width=1920,
        height=1080,
        audio_codec="aac",
        audio_bitrate_kbps=160.0,
        subtitle_codecs=[],
        video_codecs=["h264"],
        attached_pic_codecs=[],
        data_stream_codecs=[],
    )
    monkeypatch.setattr(core, "probe_media_info", lambda path: info)

    call_state = {"count": 0}

    class WithStdoutError(subprocess.CalledProcessError):
        def __init__(self, cmd: list[str]) -> None:
            super().__init__(1, cmd, stderr="broken")
            self.stdout = "partial"

    def fake_run(cmd: list[str], *_: object) -> None:
        call_state["count"] += 1
        target = Path(cmd[-1])
        target.write_bytes(b"p" * 100)
        if call_state["count"] == 1:
            raise WithStdoutError(cmd)

    original_unlink = Path.unlink

    def flaky_unlink(self: Path) -> None:  # type: ignore[override]
        if self == dst and call_state["count"] == 1:
            raise OSError("locked")
        original_unlink(self)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    monkeypatch.setattr(core, "run_ffmpeg_with_progress", fake_run)

    with caplog.at_level(logging.WARNING):
        core.encode_video(
            src=src,
            dst=dst,
            overwrite=False,
            encoder="h264_nvenc",
            output_extension=".mp4",
            quality="auto",
        )
    assert call_state["count"] == 2
    assert dst.exists()
    assert any("attempting fallback" in message for message in caplog.messages)