"""Semantic accuracy regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lrc_poc.expand import expand_text
from lrc_poc.lexicon import build_wordnet_lexicon
from lrc_poc.reduce import reduce_cloud
from lrc_poc.score import SemanticScorer


@pytest.fixture(scope="module")
def benchmark_lexicon():
    return build_wordnet_lexicon(limit_per_pos=None, max_entries=None)


def _load_gold_cases() -> list[dict[str, object]]:
    path = Path(__file__).parent / "data" / "gold_cases.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(payload.get("cases", []))


def _load_baseline() -> dict[str, float]:
    path = Path(__file__).parent / "data" / "deterministic_baseline.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_gold_case_hit_rates(benchmark_lexicon) -> None:
    cases = _load_gold_cases()
    scorer = SemanticScorer()
    top1_hits = 0
    top3_hits = 0
    top10_hits = 0
    confidences: list[float] = []

    for case in cases:
        text = str(case["input"])
        expected = {item.lower() for item in case["expected"]}
        cloud = expand_text(text)
        result = reduce_cloud(cloud=cloud, lexicon=benchmark_lexicon, scorer=scorer, top_k=10)

        top_words = [item.word.lower() for item in result.top_k]
        top1_hits += int(top_words[0] in expected)
        top3_hits += int(any(word in expected for word in top_words[:3]))
        top10_hits += int(any(word in expected for word in top_words[:10]))
        confidences.append(result.confidence)

    total = max(len(cases), 1)
    metrics = {
        "top1_hit_rate": top1_hits / total,
        "top3_hit_rate": top3_hits / total,
        "top10_hit_rate": top10_hits / total,
        "avg_confidence": sum(confidences) / len(confidences),
    }
    baseline = _load_baseline()

    assert metrics["top1_hit_rate"] >= baseline["min_top1_hit_rate"]
    assert metrics["top3_hit_rate"] >= baseline["min_top3_hit_rate"]
    assert metrics["top10_hit_rate"] >= baseline["min_top10_hit_rate"]
    assert metrics["avg_confidence"] >= baseline["min_avg_confidence"]
