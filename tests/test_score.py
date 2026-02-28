"""Tests for candidate scoring logic."""

from __future__ import annotations

from lrc_poc.score import SemanticScorer


def test_relevant_candidate_scores_higher(sample_cloud, sample_lexicon) -> None:
    scorer = SemanticScorer()
    relevant = sample_lexicon[1]
    irrelevant = sample_lexicon[2]

    relevant_score = scorer.score_candidate(sample_cloud, relevant)
    irrelevant_score = scorer.score_candidate(sample_cloud, irrelevant)

    assert relevant_score.total > irrelevant_score.total


def test_concision_bonus_prefers_shorter_words(sample_cloud) -> None:
    from lrc_poc.models import LexiconEntry

    scorer = SemanticScorer()
    short = LexiconEntry(
        word="grief",
        lemma="grief",
        part_of_speech="n",
        definition="deep sorrow from loss",
        sense_count=1,
    )
    long = LexiconEntry(
        word="deep emotional sorrow",
        lemma="deep emotional sorrow",
        part_of_speech="n",
        definition="deep sorrow from loss",
        sense_count=1,
    )

    short_score = scorer.score_candidate(sample_cloud, short)
    long_score = scorer.score_candidate(sample_cloud, long)
    assert short_score.concision_bonus > long_score.concision_bonus


def test_confidence_from_margin_increases_with_gap() -> None:
    scorer = SemanticScorer()
    low_gap = scorer.confidence_from_margin(0.51, 0.50)
    high_gap = scorer.confidence_from_margin(0.70, 0.40)
    assert high_gap > low_gap
