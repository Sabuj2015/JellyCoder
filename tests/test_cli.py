import argparse
import runpy
from pathlib import Path

import pytest

from video_reducer import cli


def test_prompt_for_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inputs = iter(["", "nonexistent", str(tmp_path)])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with caplog.at_level("WARNING"):
        result = cli._prompt_for_directory()
    assert result == tmp_path.resolve()
    assert any("not a valid directory" in message for message in caplog.messages)


def test_prompt_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["", "", "yes"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert cli._prompt_overwrite(default=True) is True
    assert cli._prompt_overwrite(default=False) is False
    assert cli._prompt_overwrite(default=False) is True


def test_prompt_quality(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    inputs = iter(["bad", "1080p"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with caplog.at_level("WARNING"):
        result = cli._prompt_quality()
    assert result == "1080p"
    assert any("Invalid quality selection" in message for message in caplog.messages)


def test_prompt_quality_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter([""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert cli._prompt_quality(default="720p") == "720p"


def test_prompt_backend(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    inputs = iter(["bad", "qsv"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with caplog.at_level("WARNING"):
        result = cli._prompt_backend()
    assert result == "qsv"
    assert any("Invalid backend selection" in message for message in caplog.messages)


def test_prompt_backend_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter([""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert cli._prompt_backend(default="amf") == "amf"


def test_parse_args_defaults() -> None:
    args = cli.parse_args([])
    assert args.input is None
    assert args.max_workers == 1
    assert args.codec == "auto"


def test_parse_args_full(tmp_path: Path) -> None:
    args = cli.parse_args([
        str(tmp_path),
        "--overwrite",
        "--output",
        str(tmp_path / "out"),
        "--max-workers",
        "3",
        "--log-level",
        "debug",
        "--codec",
        "h264",
        "--quality",
        "720p",
        "--backend",
        "qsv",
    ])
    assert args.input == str(tmp_path)
    assert args.overwrite is True
    assert args.output == tmp_path / "out"
    assert args.max_workers == 3
    assert args.log_level == "debug"
    assert args.codec == "h264"
    assert args.quality == "720p"
    assert args.backend == "qsv"


def test_main_with_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    namespace = argparse.Namespace(
        input=str(tmp_path),
        overwrite=True,
        output=tmp_path / "out",
        max_workers=2,
        log_level="info",
        codec="h264",
        quality="1080p",
        backend="nvenc",
    )
    captured = {}

    monkeypatch.setattr(cli, "parse_args", lambda argv=None: namespace)
    monkeypatch.setattr(cli, "reduce_videos", lambda config: captured.update({"config": config}))

    cli.main([])
    cfg = captured["config"]
    assert cfg.input_path == tmp_path.resolve()
    assert cfg.overwrite is True
    assert cfg.output_root == tmp_path / "out"
    assert cfg.max_workers == 2
    assert cfg.preferred_codec == "h264"
    assert cfg.quality == "1080p"
    assert cfg.encoder_backend == "nvenc"


def test_main_interactive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    namespace = argparse.Namespace(
        input=None,
        overwrite=False,
        output=None,
        max_workers=1,
        log_level="info",
        codec="auto",
        quality=None,
        backend="auto",
    )
    captured = {}

    monkeypatch.setattr(cli, "parse_args", lambda argv=None: namespace)
    monkeypatch.setattr(cli, "_prompt_for_directory", lambda: tmp_path)
    monkeypatch.setattr(cli, "_prompt_overwrite", lambda default: True)
    monkeypatch.setattr(cli, "_prompt_quality", lambda default: "480p")
    monkeypatch.setattr(cli, "_prompt_backend", lambda default: "qsv")
    monkeypatch.setattr(cli, "reduce_videos", lambda config: captured.update({"config": config}))

    cli.main([])
    cfg = captured["config"]
    assert cfg.input_path == tmp_path.resolve()
    assert cfg.overwrite is True
    assert cfg.output_root is None
    assert cfg.preferred_codec is None
    assert cfg.quality == "480p"
    assert cfg.encoder_backend == "qsv"


def test_package_main_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}
    import importlib

    importlib.import_module("video_reducer.__main__")
    monkeypatch.setattr(cli, "main", lambda argv=None: called.update({"ran": True}))
    runpy.run_module("video_reducer.__main__", run_name="__main__")
    assert called == {"ran": True}
