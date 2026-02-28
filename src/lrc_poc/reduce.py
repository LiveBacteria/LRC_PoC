"""Semantic reduction logic."""

from __future__ import annotations

from .candidates import generate_candidates
from .models import CandidateResult, LexiconEntry, ReductionResult, SemanticCloud
from .score import SemanticScorer


def reduce_cloud(
    cloud: SemanticCloud,
    lexicon: tuple[LexiconEntry, ...] | list[LexiconEntry],
    scorer: SemanticScorer | None = None,
    top_k: int = 10,
    max_candidates: int = 300,
    mode_used: int = 0,
    fallback_reason: str = "",
    candidate_pool: tuple[LexiconEntry, ...] | list[LexiconEntry] | None = None,
) -> ReductionResult:
    """Reduce a semantic cloud to top-k lexical candidates."""
    if not lexicon:
        raise ValueError("lexicon is empty")

    local_scorer = scorer or SemanticScorer()
    local_candidates = (
        tuple(candidate_pool)
        if candidate_pool is not None
        else generate_candidates(cloud=cloud, lexicon=lexicon, max_candidates=max_candidates)
    )
    if not local_candidates:
        raise ValueError("candidate generation returned no items")

    ranked: list[CandidateResult] = []
    for candidate in local_candidates:
        breakdown = local_scorer.score_candidate(cloud=cloud, candidate=candidate)
        ranked.append(
            CandidateResult(
                word=candidate.word,
                definition=candidate.definition,
                part_of_speech=candidate.part_of_speech,
                synset_id=candidate.synset_id,
                score_breakdown=breakdown,
                explanation=local_scorer.explanation(cloud, candidate, breakdown),
            )
        )

    ranked.sort(key=lambda item: (-item.score_breakdown.total, item.word))
    deduped: list[CandidateResult] = []
    seen: set[str] = set()
    for item in ranked:
        if item.word in seen:
            continue
        seen.add(item.word)
        deduped.append(item)
        if len(deduped) >= max(top_k, 1):
            break

    winner = deduped[0]
    second_best = deduped[1].score_breakdown.total if len(deduped) > 1 else None
    confidence = local_scorer.confidence_from_margin(
        winner.score_breakdown.total,
        second_best,
    )

    return ReductionResult(
        winner=winner,
        top_k=tuple(deduped),
        confidence=confidence,
        candidate_count=len(local_candidates),
        mode_used=mode_used,
        fallback_reason=fallback_reason,
    )
