"""Milestone 3 attractor mapping utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .lexicon import sample_seed_words
from .models import AttractorRunResult, IterationResult, LexiconEntry
from .recurse import run_recursion, summarize_iterations
from .score import SemanticScorer


def _trajectory_rows(seed: str, iterations: tuple[IterationResult, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in iterations:
        rows.append(
            {
                "seed": seed,
                "iteration": item.iteration,
                "input_text": item.input_text,
                "winner_word": item.winner_word,
                "winner_score": item.winner_score,
                "confidence": item.confidence,
                "drift_from_origin": item.drift_from_origin,
                "entropy": item.entropy,
                "fixed_point": item.fixed_point,
                "cycle_detected": item.cycle_detected,
                "cycle_length": item.cycle_length,
                "top_k_words": "|".join(item.top_k_words),
            }
        )
    return rows


def _summary_row(seed: str, iterations: tuple[IterationResult, ...]) -> dict[str, object]:
    last = iterations[-1]
    summary = summarize_iterations(iterations)
    return {
        "seed": seed,
        "final_word": last.winner_word,
        "fixed_points": int(summary["fixed_points"]),
        "cycles": int(summary["cycles"]),
        "average_drift": summary["average_drift"],
        "average_entropy": summary["average_entropy"],
    }


def map_attractors(
    lexicon: tuple[LexiconEntry, ...] | list[LexiconEntry],
    seeds: tuple[str, ...] | list[str] | None = None,
    seed_count: int = 200,
    iterations: int = 10,
    output_dir: str | Path = "artifacts",
    scorer: SemanticScorer | None = None,
    mode_used: int = 0,
    fallback_reason: str = "",
) -> AttractorRunResult:
    """Run recursive mapping over many seeds and export result artifacts."""
    if not lexicon:
        raise ValueError("lexicon is empty")
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if seed_count <= 0 and not seeds:
        raise ValueError("seed_count must be > 0")

    local_scorer = scorer or SemanticScorer()
    seed_terms = tuple(seeds) if seeds else sample_seed_words(max_seeds=seed_count)
    if not seed_terms:
        raise ValueError("no seeds available for attractor mapping")

    trajectory_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    basin_summary: dict[str, int] = {}

    for seed in seed_terms:
        iterations_result = run_recursion(
            source_text=seed,
            lexicon=lexicon,
            iterations=iterations,
            scorer=local_scorer,
            mode_used=mode_used,
            fallback_reason=fallback_reason,
        )
        trajectory_records.extend(_trajectory_rows(seed, iterations_result))
        row = _summary_row(seed, iterations_result)
        summary_records.append(row)
        basin_summary[row["final_word"]] = basin_summary.get(row["final_word"], 0) + 1

    trajectories_df = pd.DataFrame(trajectory_records)
    summary_df = pd.DataFrame(summary_records)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trajectories_file = output_path / "attractor_trajectories.csv"
    summary_file = output_path / "attractor_summary.csv"
    basin_file = output_path / "attractor_basins.json"

    trajectories_df.to_csv(trajectories_file, index=False)
    summary_df.to_csv(summary_file, index=False)
    basin_file.write_text(json.dumps(basin_summary, indent=2), encoding="utf-8")

    fixed_points = int(summary_df["fixed_points"].gt(0).sum())
    cycles = int(summary_df["cycles"].gt(0).sum())
    avg_drift = float(summary_df["average_drift"].mean()) if not summary_df.empty else 0.0
    avg_entropy = float(summary_df["average_entropy"].mean()) if not summary_df.empty else 0.0

    return AttractorRunResult(
        seed_count=len(seed_terms),
        iterations=iterations,
        fixed_points=fixed_points,
        cycles=cycles,
        average_drift=avg_drift,
        average_entropy=avg_entropy,
        basin_summary=basin_summary,
        output_files=(
            str(trajectories_file),
            str(summary_file),
            str(basin_file),
        ),
    )
