import json
import os
from pathlib import Path
import subprocess
import sys


def test_cli_runs_under_cp1252_and_writes_utf8_artifact(tmp_path):
    artifact = tmp_path / "evaluation.json"
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    completed = subprocess.run(
        [sys.executable, "-m", "research_copilot.evaluation", "--output", str(artifact)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("cp1252", errors="replace")
    # ASCII-only JSON works regardless of the console code page.
    console = json.loads(completed.stdout.decode("ascii"))
    artifact_text = artifact.read_text(encoding="utf-8")
    assert "有依据回答与原文引用" in artifact_text
    assert json.loads(artifact_text) == console
    assert console["passed"] == console["total"] == 6
