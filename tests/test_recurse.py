"""Tests for recursion metrics and dynamics."""

from __future__ import annotations

from lrc_poc.models import CandidateResult, ReductionResult, ScoreBreakdown
from lrc_poc.recurse import lexical_entropy_profile, run_recursion, summarize_iterations


def _result_for_word(word: str) -> ReductionResult:
    breakdown = ScoreBreakdown(
        definition_similarity=1.0,
        synonym_relation_score=1.0,
        hypernym_alignment=1.0,
        keyword_overlap=1.0,
        pos_match=1.0,
        concision_bonus=1.0,
        ambiguity_penalty=0.0,
        total=1.0,
    )
    winner = CandidateResult(
        word=word,
        definition=f"definition for {word}",
        part_of_speech="n",
        synset_id=f"{word}.n.01",
        score_breakdown=breakdown,
        explanation="test",
    )
    return ReductionResult(
        winner=winner,
        top_k=(winner,),
        confidence=0.9,
        candidate_count=1,
        mode_used=0,
    )


def test_run_recursion_detects_fixed_point(sample_lexicon) -> None:
    def expand_fn(text: str):
        from lrc_poc.models import SemanticCloud

        return SemanticCloud(
            source_text=text,
            key_terms=(text,),
            definitions=(text,),
            synonyms=(text,),
            broader_concepts=(),
            related_concepts=(),
            constraints=(),
            negative_constraints=(),
            composed_cloud_text=text,
            preferred_pos="n",
        )

    def reduce_fn(**kwargs):
        cloud = kwargs["cloud"]
        return _result_for_word(cloud.source_text)

    iterations = run_recursion(
        source_text="grief",
        lexicon=sample_lexicon,
        iterations=3,
        expand_fn=expand_fn,
        reduce_fn=reduce_fn,
    )
    assert iterations[0].fixed_point
    assert all(item.winner_word == "grief" for item in iterations)


def test_run_recursion_detects_cycle(sample_lexicon) -> None:
    def expand_fn(text: str):
        from lrc_poc.models import SemanticCloud

        return SemanticCloud(
            source_text=text,
            key_terms=(text,),
            definitions=(text,),
            synonyms=(text,),
            broader_concepts=(),
            related_concepts=(),
            constraints=(),
            negative_constraints=(),
            composed_cloud_text=text,
            preferred_pos="n",
        )

    mapping = {"a": "b", "b": "a"}

    def reduce_fn(**kwargs):
        cloud = kwargs["cloud"]
        return _result_for_word(mapping[cloud.source_text])

    iterations = run_recursion(
        source_text="a",
        lexicon=sample_lexicon,
        iterations=4,
        expand_fn=expand_fn,
        reduce_fn=reduce_fn,
    )
    assert any(item.cycle_detected for item in iterations)
    cycle_items = [item for item in iterations if item.cycle_detected]
    assert cycle_items[0].cycle_length == 2


def test_run_recursion_is_deterministic(sample_lexicon) -> None:
    first = run_recursion("grief", sample_lexicon, iterations=3)
    second = run_recursion("grief", sample_lexicon, iterations=3)
    assert [item.winner_word for item in first] == [item.winner_word for item in second]


def test_recursion_summary_helpers(sample_lexicon) -> None:
    iterations = run_recursion("grief", sample_lexicon, iterations=3)
    summary = summarize_iterations(iterations)
    profile = lexical_entropy_profile(iterations)
    assert summary["average_drift"] >= 0.0
    assert summary["average_entropy"] >= 0.0
    assert profile["unique_winners"] >= 1.0
