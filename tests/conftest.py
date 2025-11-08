import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def restore_std_streams() -> Generator[None, None, None]:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@pytest.fixture
def temp_cwd(tmp_path: Path) -> Generator[Path, None, None]:
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(previous)