"""Command-line interface for LRC milestone workflows."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from .attractor import map_attractors
from .lexicon import build_wordnet_lexicon
from .pipeline import LRCPipeline
from .recurse import lexical_entropy_profile, run_pipeline_recursion, summarize_iterations
from .sentence_ops import run_sentence_definition_cycle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LRC Milestone 3 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3, 4])
    shared.add_argument("--top-k", type=int, default=10)
    shared.add_argument("--max-candidates", type=int, default=300)
    shared.add_argument("--max-entries", type=int, default=80000)
    shared.add_argument("--limit-per-pos", type=int, default=0)

    run_once_parser = subparsers.add_parser("run-once", parents=[shared])
    run_once_parser.add_argument("--text", required=True)
    run_once_parser.add_argument(
        "--definition-style",
        choices=["literal_first", "semantic_relational", "literal_raw"],
        default="literal_first",
    )
    run_once_parser.add_argument(
        "--semantic-fallback-reduction",
        action="store_true",
        help="Allow semantic fallback when exact definition-to-term match is unavailable.",
    )
    run_once_parser.add_argument(
        "--include-global-compression",
        action="store_true",
        help="Also compute global lexical compression for sentence input.",
    )

    recurse_parser = subparsers.add_parser("recurse", parents=[shared])
    recurse_parser.add_argument("--text", required=True)
    recurse_parser.add_argument("--iterations", type=int, default=10)

    map_parser = subparsers.add_parser("map", parents=[shared])
    map_parser.add_argument("--seed-file", type=str, default="")
    map_parser.add_argument("--seed-count", type=int, default=200)
    map_parser.add_argument("--iterations", type=int, default=10)
    map_parser.add_argument("--out", type=str, default="artifacts")

    return parser


def _load_seeds(seed_file: str) -> list[str] | None:
    if not seed_file:
        return None
    path = Path(seed_file)
    if not path.exists():
        raise FileNotFoundError(f"Seed file does not exist: {path}")
    seeds = [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()]
    return [item for item in seeds if item]


def _load_lexicon(max_entries: int, limit_per_pos: int) -> tuple:
    limit = limit_per_pos if limit_per_pos > 0 else None
    max_items = max_entries if max_entries > 0 else None
    return build_wordnet_lexicon(limit_per_pos=limit, max_entries=max_items)


def _print_payload(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2))


def _run_sentence_cycle_compat(
    *,
    text: str,
    lexicon,
    pipeline: LRCPipeline,
    mode: int,
    definition_style: str,
    semantic_fallback_reduction: bool,
) -> dict[str, object]:
    signature = inspect.signature(run_sentence_definition_cycle)
    kwargs = {
        "text": text,
        "lexicon": lexicon,
        "pipeline": pipeline,
        "mode": mode,
        "definition_style": definition_style,
        "use_semantic_fallback": semantic_fallback_reduction,
    }
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return run_sentence_definition_cycle(**supported)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    lexicon = _load_lexicon(max_entries=args.max_entries, limit_per_pos=args.limit_per_pos)
    pipeline = LRCPipeline(lexicon=lexicon)

    if args.command == "run-once":
        sentence_cycle = _run_sentence_cycle_compat(
            text=args.text,
            lexicon=lexicon,
            pipeline=pipeline,
            mode=args.mode,
            definition_style=args.definition_style,
            semantic_fallback_reduction=args.semantic_fallback_reduction,
        )
        token_count = len([piece for piece in args.text.strip().split() if piece])
        sentence_input = token_count > 1

        if sentence_input and not args.include_global_compression:
            _print_payload(
                {
                    "mode": args.mode,
                    "result_type": "sentence_cycle",
                    "definition_cycle": sentence_cycle,
                }
            )
            return

        result = pipeline.run_once(
            text=args.text,
            mode=args.mode,
            top_k=args.top_k,
            max_candidates=args.max_candidates,
        )
        _print_payload(
            {
                "mode": args.mode,
                "result_type": "global_compression",
                "winner": result.winner.word,
                "confidence": result.confidence,
                "fallback_reason": result.fallback_reason,
                "definition_cycle": sentence_cycle,
                "top_k": [
                    {
                        "word": item.word,
                        "score": item.score_breakdown.total,
                        "definition": item.definition,
                    }
                    for item in result.top_k
                ],
            }
        )
        return

    if args.command == "recurse":
        iterations = run_pipeline_recursion(
            source_text=args.text,
            pipeline=pipeline,
            iterations=args.iterations,
            mode=args.mode,
        )
        _print_payload(
            {
                "mode": args.mode,
                "iterations": args.iterations,
                "summary": summarize_iterations(iterations),
                "candidate_uncertainty_profile": lexical_entropy_profile(iterations),
                "trajectory": [
                    {
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
                    }
                    for item in iterations
                ],
            }
        )
        return

    if args.command == "map":
        seeds = _load_seeds(args.seed_file) if args.seed_file else None
        result = map_attractors(
            lexicon=lexicon,
            seeds=seeds,
            seed_count=args.seed_count,
            iterations=args.iterations,
            output_dir=args.out,
            mode_used=args.mode,
            pipeline=pipeline,
        )
        _print_payload(
            {
                "mode": args.mode,
                "seed_count": result.seed_count,
                "iterations": result.iterations,
                "fixed_points": result.fixed_points,
                "cycles": result.cycles,
                "average_drift": result.average_drift,
                "average_entropy": result.average_entropy,
                "basin_summary": result.basin_summary,
                "output_files": list(result.output_files),
            }
        )
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
