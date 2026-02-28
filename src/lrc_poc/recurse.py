"""Recursive expansion/reduction analysis."""

from __future__ import annotations

from collections import Counter
import math
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .expand import safe_expand_text
from .models import IterationResult, LexiconEntry, ReductionResult, SemanticCloud
from .reduce import reduce_cloud
from .score import SemanticScorer


def _text_similarity(left: str, right: str) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        matrix = vectorizer.fit_transform([left, right])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def _normalized_entropy(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    positive = [max(value, 1e-9) for value in values]
    total = sum(positive)
    probs = [value / total for value in positive]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def run_recursion(
    source_text: str,
    lexicon: tuple[LexiconEntry, ...] | list[LexiconEntry],
    iterations: int = 10,
    scorer: SemanticScorer | None = None,
    mode_used: int = 0,
    fallback_reason: str = "",
    expand_fn: Callable[[str], SemanticCloud] = safe_expand_text,
    reduce_fn: Callable[..., ReductionResult] = reduce_cloud,
) -> tuple[IterationResult, ...]:
    """Run recursive expand->reduce iterations and track dynamics."""
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")

    local_scorer = scorer or SemanticScorer()
    origin = source_text.strip().lower()
    current = origin
    seen_words: dict[str, int] = {}
    results: list[IterationResult] = []

    for index in range(iterations):
        cloud = expand_fn(current)
        reduction = reduce_fn(
            cloud=cloud,
            lexicon=lexicon,
            scorer=local_scorer,
            top_k=10,
            mode_used=mode_used,
            fallback_reason=fallback_reason,
        )
        winner = reduction.winner.word
        scores = tuple(item.score_breakdown.total for item in reduction.top_k)
        entropy = _normalized_entropy(scores)
        drift = 1.0 - _text_similarity(origin, winner)

        cycle_detected = winner in seen_words
        cycle_length = (index - seen_words[winner]) if cycle_detected else None
        fixed_point = winner == current
        seen_words[winner] = index

        results.append(
            IterationResult(
                iteration=index + 1,
                input_text=current,
                winner_word=winner,
                winner_score=reduction.winner.score_breakdown.total,
                confidence=reduction.confidence,
                drift_from_origin=drift,
                entropy=entropy,
                fixed_point=fixed_point,
                cycle_detected=cycle_detected,
                cycle_length=cycle_length,
                top_k_words=tuple(item.word for item in reduction.top_k),
            )
        )
        current = winner

    return tuple(results)


def summarize_iterations(iterations: tuple[IterationResult, ...]) -> dict[str, float]:
    """Aggregate recursion metrics for reporting and mapping."""
    if not iterations:
        return {
            "fixed_points": 0.0,
            "cycles": 0.0,
            "average_drift": 0.0,
            "average_entropy": 0.0,
        }

    fixed_points = sum(1 for item in iterations if item.fixed_point)
    cycles = sum(1 for item in iterations if item.cycle_detected)
    average_drift = sum(item.drift_from_origin for item in iterations) / len(iterations)
    average_entropy = sum(item.entropy for item in iterations) / len(iterations)

    return {
        "fixed_points": float(fixed_points),
        "cycles": float(cycles),
        "average_drift": average_drift,
        "average_entropy": average_entropy,
    }


def lexical_entropy_profile(iterations: tuple[IterationResult, ...]) -> dict[str, float]:
    """Compute entropy profile over winner trajectories."""
    winners = [item.winner_word for item in iterations]
    if not winners:
        return {"unique_winners": 0.0, "winner_entropy": 0.0}

    counts = Counter(winners)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0

    return {
        "unique_winners": float(len(counts)),
        "winner_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
    }
