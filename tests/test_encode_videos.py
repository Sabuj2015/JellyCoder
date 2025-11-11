import importlib
import runpy


def test_encode_videos_main(monkeypatch):
    called = {}
    import jelly_coder.cli as cli

    monkeypatch.setattr(cli, "main", lambda argv=None: called.update({"ran": True, "argv": argv}))

    module = importlib.import_module("encode_videos")
    module.main(["--help"])
    assert called == {"ran": True, "argv": ["--help"]}


def test_encode_videos_module_execution(monkeypatch):
    called = {}
    import jelly_coder.cli as cli

    monkeypatch.setattr(cli, "main", lambda argv=None: called.update({"ran": True}))
    runpy.run_module("encode_videos", run_name="__main__")
    assert called == {"ran": True}
