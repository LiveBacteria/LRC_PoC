"""Smoke tests for dashboard dependencies and data flow."""

from __future__ import annotations

from pathlib import Path

from lrc_poc.attractor import map_attractors
from lrc_poc.dashboard import app as dashboard_app


def test_dashboard_pipeline_smoke(tmp_path: Path) -> None:
    pipeline = dashboard_app.prepare_pipeline(max_entries=4000, limit_per_pos=200)
    result = pipeline.run_once("grief", mode=0, top_k=5)
    assert result.winner.word
    assert len(result.top_k) >= 1

    map_result = map_attractors(
        lexicon=pipeline.lexicon,
        seeds=("grief", "joy", "rage"),
        iterations=3,
        output_dir=tmp_path / "dashboard_map",
        mode_used=0,
        pipeline=pipeline,
    )
    assert map_result.seed_count == 3
    for output_file in map_result.output_files:
        assert Path(output_file).exists()
