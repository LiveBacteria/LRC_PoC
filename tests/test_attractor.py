"""Tests for attractor mapping."""

from __future__ import annotations

import json
import time
from pathlib import Path
import csv

from lrc_poc.attractor import map_attractors
from lrc_poc.lexicon import build_wordnet_lexicon


def test_map_attractors_writes_artifacts(tmp_path: Path, sample_lexicon) -> None:
    out_dir = tmp_path / "attractor_run"
    result = map_attractors(
        lexicon=sample_lexicon,
        seeds=("grief", "loss"),
        iterations=3,
        output_dir=out_dir,
    )
    assert result.seed_count == 2
    assert len(result.output_files) == 3
    for file_path in result.output_files:
        assert Path(file_path).exists()

    basin_payload = json.loads(Path(result.output_files[2]).read_text(encoding="utf-8"))
    assert isinstance(basin_payload, dict)

    summary_file = Path(result.output_files[1])
    with summary_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader)
    assert first_row["use_domain_heuristics"] in {"True", "False"}
    assert first_row["ambiguity_penalty_enabled"] in {"True", "False"}


def test_map_attractors_is_repeatable(tmp_path: Path, sample_lexicon) -> None:
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    result_a = map_attractors(
        lexicon=sample_lexicon,
        seeds=("grief", "loss"),
        iterations=3,
        output_dir=out_a,
    )
    result_b = map_attractors(
        lexicon=sample_lexicon,
        seeds=("grief", "loss"),
        iterations=3,
        output_dir=out_b,
    )
    assert result_a.basin_summary == result_b.basin_summary


def test_attractor_runtime_sanity(tmp_path: Path) -> None:
    lexicon = build_wordnet_lexicon(limit_per_pos=30, max_entries=1000)
    start = time.monotonic()
    _ = map_attractors(
        lexicon=lexicon,
        seed_count=20,
        iterations=3,
        output_dir=tmp_path / "runtime",
    )
    elapsed = time.monotonic() - start
    assert elapsed < 120
