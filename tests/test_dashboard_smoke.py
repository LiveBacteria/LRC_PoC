"""Smoke tests for dashboard dependencies and data flow."""

from __future__ import annotations

from pathlib import Path

from lrc_poc.attractor import map_attractors
from lrc_poc.dashboard import app as dashboard_app
from lrc_poc.pipeline import LRCPipeline


def test_dashboard_pipeline_smoke(tmp_path: Path, sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    result = pipeline.run_once("grief", mode=0, top_k=5)
    assert result.winner.word
    assert len(result.top_k) >= 1

    map_result = map_attractors(
        lexicon=sample_lexicon,
        seeds=("grief", "joy", "rage"),
        iterations=2,
        output_dir=tmp_path / "dashboard_map",
        mode_used=0,
        pipeline=pipeline,
    )
    assert map_result.seed_count == 3
    for output_file in map_result.output_files:
        assert Path(output_file).exists()


def test_sentence_mode_style_matrix_smoke(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    payload = dashboard_app._run_sentence_mode_style_matrix(
        text="The cat jumped over the dog.",
        pipeline=pipeline,
        milestone_key="m1",
        cycle_count=1,
        operation_sequence=("expand", "reduce"),
        recursive_reduce_max_steps=8,
    )
    assert payload["kind"] == "matrix"
    assert payload["styles_tested"] == ["literal_first", "cloud_compact"]
    assert len(payload["results"]) == 2
    assert all(item["definition_style"] != "literal_raw" for item in payload["results"])
