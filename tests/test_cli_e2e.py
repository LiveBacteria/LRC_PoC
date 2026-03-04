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
        "the cat jumped over the dog",
        "--max-entries",
        "1500",
        "--limit-per-pos",
        "80",
    )
    assert payload["result_type"] == "sentence_cycle"
    assert payload["use_domain_heuristics"] is False
    assert payload["definition_cycle"]["expanded_sentence"]
    assert payload["definition_cycle"]["reduced_sentence"]


def test_cli_run_once_supports_domain_heuristics_flag() -> None:
    payload = _run_cli(
        "run-once",
        "--text",
        "grief",
        "--use-domain-heuristics",
        "--max-entries",
        "1200",
        "--limit-per-pos",
        "60",
    )
    assert payload["use_domain_heuristics"] is True


def test_cli_run_once_can_emit_reduction_diagnostics() -> None:
    payload = _run_cli(
        "run-once",
        "--text",
        "the cat jumped over the dog",
        "--max-entries",
        "1200",
        "--limit-per-pos",
        "60",
        "--emit-reduction-diagnostics",
    )
    assert payload["result_type"] == "sentence_cycle"
    assert isinstance(payload["reduction_diagnostics"], dict)
    assert isinstance(payload["reducer_config"], dict)


def test_cli_recurse() -> None:
    payload = _run_cli(
        "recurse",
        "--text",
        "a positive emotional state with pleasure and contentment",
        "--iterations",
        "3",
        "--max-entries",
        "1500",
        "--limit-per-pos",
        "80",
    )
    assert payload["iterations"] == 3
    assert len(payload["trajectory"]) == 3


def test_cli_map(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("grief\njoy\nrage\n", encoding="utf-8")
    out_dir = tmp_path / "map_output"

    payload = _run_cli(
        "map",
        "--seed-file",
        str(seed_file),
        "--iterations",
        "2",
        "--out",
        str(out_dir),
        "--max-entries",
        "1500",
        "--limit-per-pos",
        "80",
    )
    assert payload["seed_count"] == 3
    for file_path in payload["output_files"]:
        assert Path(file_path).exists()


def test_cli_investigate_reduction(tmp_path: Path) -> None:
    thread_payload = {
        "id": "thread-1",
        "title": "Thread 1",
        "created_at": "2026-03-03T00:00:00",
        "updated_at": "2026-03-03T00:00:00",
        "milestone_hint": "m2",
        "messages": [
            {"role": "user", "content": "test case"},
            {
                "role": "assistant",
                "payload": {
                    "kind": "m2",
                    "input_text": "The cat jumped over the dog.",
                    "definition_style": "literal_first",
                    "cycle_count": 1,
                    "operation_sequence": ["expand", "expand", "reduce"],
                    "recursive_reduce_max_steps": 8,
                    "final_output": "",
                    "cycles": [],
                },
            },
        ],
    }
    thread_path = tmp_path / "thread.json"
    thread_path.write_text(json.dumps(thread_payload), encoding="utf-8")
    payload = _run_cli(
        "investigate-reduction",
        "--thread-json",
        str(thread_path),
        "--max-entries",
        "1200",
        "--limit-per-pos",
        "60",
    )
    assert payload["incident_case_count"] == 1
    assert isinstance(payload["summary_diagnostics"], dict)
    assert isinstance(payload["summary_root_cause_ranked"], list)
    assert isinstance(payload["cases"], list)
