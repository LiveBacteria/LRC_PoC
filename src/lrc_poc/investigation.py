"""Replay and root-cause tooling for sentence reduction incidents."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .sentence_ops import DEFAULT_RECURSIVE_REDUCTION_LIMIT, run_sentence_operation_sequence


def _merge_reduction_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, dict):
                target = merged.setdefault(key, {})
                if not isinstance(target, dict):
                    continue
                for bucket_key, bucket_value in value.items():
                    target[str(bucket_key)] = int(target.get(str(bucket_key), 0)) + int(bucket_value)
            elif isinstance(value, (int, float)):
                merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def _as_int(mapping: dict[str, Any], key: str) -> int:
    return int(mapping.get(key, 0))


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 3)


def _span_replacement_probability(diagnostics: dict[str, Any], *, max_spans: int = 24) -> dict[str, float | int]:
    considered = diagnostics.get("windows_considered_by_span", {})
    accepted = diagnostics.get("accepted_replacements_by_span", {})
    if not isinstance(considered, dict) or not isinstance(accepted, dict):
        return {}
    span_rows: list[tuple[str, int, int]] = []
    for span_key in {*considered.keys(), *accepted.keys()}:
        seen = int(considered.get(span_key, 0))
        won = int(accepted.get(span_key, 0))
        span_rows.append((str(span_key), seen, won))
    span_rows.sort(key=lambda row: (-row[1], int(row[0])))

    probabilities: dict[str, float | int] = {}
    for span_key, seen, won in span_rows[:max_spans]:
        probabilities[span_key] = round((won / seen), 6) if seen > 0 else 0.0
    omitted = len(span_rows) - len(probabilities)
    if omitted > 0:
        probabilities["_omitted_span_count"] = omitted
    return probabilities


def _rank_root_causes(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    windows_considered = _as_int(diagnostics, "windows_considered")
    exact_matches = _as_int(diagnostics, "exact_definition_matches")
    semantic_attempted = _as_int(diagnostics, "semantic_windows_attempted")
    no_candidates = _as_int(diagnostics, "windows_no_candidates")
    rejected_score = _as_int(diagnostics, "windows_rejected_score")
    rejected_margin = _as_int(diagnostics, "windows_rejected_margin")
    pruned_limit = _as_int(diagnostics, "windows_pruned_limit")
    fallback_preserve = _as_int(diagnostics, "fallback_preserve_count")
    accepted = _as_int(diagnostics, "accepted_replacements")

    exact_index_miss = max(0, windows_considered - exact_matches)
    threshold_reject = rejected_score + rejected_margin

    replacement_denominator = max(1, accepted + fallback_preserve)
    ranked = [
        {
            "cause": "exact_index_miss",
            "count": exact_index_miss,
            "percentage": _percent(exact_index_miss, max(1, windows_considered)),
            "denominator": "windows_considered",
        },
        {
            "cause": "semantic_candidate_miss",
            "count": no_candidates,
            "percentage": _percent(no_candidates, max(1, semantic_attempted)),
            "denominator": "semantic_windows_attempted",
        },
        {
            "cause": "threshold_reject",
            "count": threshold_reject,
            "percentage": _percent(threshold_reject, max(1, semantic_attempted)),
            "denominator": "semantic_windows_attempted",
        },
        {
            "cause": "prune_limit_reject",
            "count": pruned_limit,
            "percentage": _percent(pruned_limit, max(1, windows_considered)),
            "denominator": "windows_considered",
        },
        {
            "cause": "fallback_preserve",
            "count": fallback_preserve,
            "percentage": _percent(fallback_preserve, replacement_denominator),
            "denominator": "accepted_plus_fallback",
        },
    ]
    return sorted(ranked, key=lambda row: (-float(row["percentage"]), -int(row["count"]), row["cause"]))


def _extract_cases(thread_payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = thread_payload.get("messages", [])
    if not isinstance(messages, list):
        return []

    cases: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        payload = message.get("payload")
        if not isinstance(payload, dict) or str(payload.get("kind")) != "m2":
            continue

        operation_sequence = payload.get("operation_sequence", [])
        if not isinstance(operation_sequence, list):
            operation_sequence = []
        sequence = tuple(str(item).strip() for item in operation_sequence if str(item).strip())
        if not sequence:
            continue

        cycle_count = int(payload.get("cycle_count", 1) or 1)
        definition_style = str(payload.get("definition_style", "literal_first") or "literal_first")
        recursive_reduce_max_steps = int(
            payload.get("recursive_reduce_max_steps", DEFAULT_RECURSIVE_REDUCTION_LIMIT)
            or DEFAULT_RECURSIVE_REDUCTION_LIMIT
        )
        input_text = str(payload.get("input_text", "") or "")
        expected_final_output = str(payload.get("final_output", "") or "")

        cycles = payload.get("cycles", [])
        if not input_text and isinstance(cycles, list) and cycles:
            first_cycle = cycles[0]
            if isinstance(first_cycle, dict):
                input_text = str(first_cycle.get("input", "") or "")

        if not input_text:
            continue

        cases.append(
            {
                "message_index": index,
                "input_text": input_text,
                "operation_sequence": sequence,
                "cycle_count": cycle_count,
                "definition_style": definition_style,
                "recursive_reduce_max_steps": recursive_reduce_max_steps,
                "expected_final_output": expected_final_output,
            }
        )
    return cases


def _collect_reduce_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for step in steps:
        operation = str(step.get("operation", ""))
        if operation == "reduce":
            reports.append(
                {
                    "step": int(step.get("step", 0)),
                    "operation": operation,
                    "replacement_count": int(step.get("replacement_count", 0)),
                    "reduction_diagnostics": dict(step.get("reduction_diagnostics", {})),
                }
            )
            continue

        if operation != "recursive_reduce":
            continue

        recursive_reports: list[dict[str, Any]] = []
        recursive_iterations = step.get("recursive_iterations", [])
        if isinstance(recursive_iterations, list):
            for item in recursive_iterations:
                if not isinstance(item, dict):
                    continue
                recursive_reports.append(
                    {
                        "depth": int(item.get("depth", 0)),
                        "replacement_count": int(item.get("replacement_count", 0)),
                        "reduction_diagnostics": dict(item.get("reduction_diagnostics", {})),
                    }
                )

        reports.append(
            {
                "step": int(step.get("step", 0)),
                "operation": operation,
                "replacement_count": int(step.get("replacement_count", 0)),
                "recursive_iteration_count": int(step.get("recursive_iteration_count", 0)),
                "recursive_stop_reason": str(step.get("recursive_stop_reason", "")),
                "reduction_diagnostics": dict(
                    step.get("recursive_reduction_diagnostics_totals", step.get("reduction_diagnostics", {}))
                ),
                "recursive_iterations": recursive_reports,
            }
        )
    return reports


def build_reduction_incident_report(
    *,
    thread_json_path: str | Path,
    lexicon,
    pipeline,
    max_cases: int = 0,
) -> dict[str, Any]:
    path = Path(thread_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = _extract_cases(payload)
    if max_cases > 0:
        cases = cases[:max_cases]

    incident_cases: list[dict[str, Any]] = []
    report_started = time.perf_counter()
    for case in cases:
        case_started = time.perf_counter()
        current_text = str(case["input_text"])
        cycle_reports: list[dict[str, Any]] = []
        case_cycle_diagnostics: list[dict[str, Any]] = []

        for cycle_index in range(1, int(case["cycle_count"]) + 1):
            cycle_started = time.perf_counter()
            sequence_result = run_sentence_operation_sequence(
                text=current_text,
                lexicon=lexicon,
                pipeline=pipeline,
                mode=0,
                definition_style=str(case["definition_style"]),
                use_semantic_fallback=False,
                operation_sequence=tuple(case["operation_sequence"]),
                recursive_reduce_max_steps=int(case["recursive_reduce_max_steps"]),
            )
            reduce_steps = _collect_reduce_steps(list(sequence_result.get("steps", [])))
            cycle_diagnostics = dict(sequence_result.get("reduction_diagnostics", {}))
            case_cycle_diagnostics.append(cycle_diagnostics)
            cycle_reports.append(
                {
                    "cycle": cycle_index,
                    "input_text": current_text,
                    "final_output": str(sequence_result.get("final_text", "")),
                    "reduction_diagnostics": cycle_diagnostics,
                    "reduce_steps": reduce_steps,
                    "elapsed_seconds": round(time.perf_counter() - cycle_started, 6),
                }
            )
            current_text = str(sequence_result.get("final_text", ""))

        aggregate_diagnostics = _merge_reduction_diagnostics(case_cycle_diagnostics)
        accepted = _as_int(aggregate_diagnostics, "accepted_replacements")
        incident_cases.append(
            {
                "message_index": int(case["message_index"]),
                "input_text": str(case["input_text"]),
                "expected_final_output": str(case["expected_final_output"]),
                "replay_final_output": current_text,
                "operation_sequence": list(case["operation_sequence"]),
                "cycle_count": int(case["cycle_count"]),
                "definition_style": str(case["definition_style"]),
                "recursive_reduce_max_steps": int(case["recursive_reduce_max_steps"]),
                "status": "stalled_no_replacements" if accepted == 0 else "replacements_found",
                "aggregate_diagnostics": aggregate_diagnostics,
                "root_cause_ranked": _rank_root_causes(aggregate_diagnostics),
                "replacement_probability_by_span": _span_replacement_probability(aggregate_diagnostics),
                "elapsed_seconds": round(time.perf_counter() - case_started, 6),
                "cycles": cycle_reports,
            }
        )

    summary_diagnostics = _merge_reduction_diagnostics(
        [dict(item.get("aggregate_diagnostics", {})) for item in incident_cases]
    )
    stalled_count = sum(1 for item in incident_cases if str(item.get("status")) == "stalled_no_replacements")
    return {
        "source_thread_path": str(path),
        "incident_case_count": len(incident_cases),
        "stalled_case_count": stalled_count,
        "summary_diagnostics": summary_diagnostics,
        "summary_root_cause_ranked": _rank_root_causes(summary_diagnostics),
        "summary_replacement_probability_by_span": _span_replacement_probability(summary_diagnostics),
        "total_elapsed_seconds": round(time.perf_counter() - report_started, 6),
        "cases": incident_cases,
    }
