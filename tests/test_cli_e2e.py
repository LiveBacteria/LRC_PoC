"""End-to-end CLI tests for milestone workflows."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str) -> dict[str, object]:
    command = [sys.executable, "-m", "lrc_poc.cli", *args]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def test_cli_run_once() -> None:
    payload = _run_cli(
        "run-once",
        "--text",
        "grief",
        "--mode",
        "0",
        "--max-entries",
        "4000",
        "--limit-per-pos",
        "200",
    )
    assert payload["winner"]
    assert len(payload["top_k"]) >= 1


def test_cli_recurse() -> None:
    payload = _run_cli(
        "recurse",
        "--text",
        "grief",
        "--iterations",
        "4",
        "--mode",
        "0",
        "--max-entries",
        "4000",
        "--limit-per-pos",
        "200",
    )
    assert payload["iterations"] == 4
    assert len(payload["trajectory"]) == 4


def test_cli_map(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("grief\njoy\nrage\n", encoding="utf-8")
    out_dir = tmp_path / "map_output"

    payload = _run_cli(
        "map",
        "--seed-file",
        str(seed_file),
        "--iterations",
        "3",
        "--mode",
        "0",
        "--out",
        str(out_dir),
        "--max-entries",
        "4000",
        "--limit-per-pos",
        "200",
    )
    assert payload["seed_count"] == 3
    for file_path in payload["output_files"]:
        assert Path(file_path).exists()
