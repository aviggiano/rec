from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_smoke() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "rec_pipeline.cli", "--help"],  # noqa: S607
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "rec" in completed.stdout
    assert "run" in completed.stdout
