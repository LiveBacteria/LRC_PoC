"""Tests for semantic reduction stage."""

from __future__ import annotations

import pytest

from lrc_poc.reduce import reduce_cloud
from lrc_poc.score import SemanticScorer


def test_reduce_cloud_returns_top_k(sample_cloud, sample_lexicon) -> None:
    result = reduce_cloud(sample_cloud, sample_lexicon, scorer=SemanticScorer(), top_k=2)
    assert result.winner.word in {"sorrow", "bereavement"}
    assert len(result.top_k) == 2
    assert result.top_k[0].score_breakdown.total >= result.top_k[1].score_breakdown.total


def test_reduce_cloud_returns_confidence(sample_cloud, sample_lexicon) -> None:
    result = reduce_cloud(sample_cloud, sample_lexicon, scorer=SemanticScorer(), top_k=3)
    assert 0.0 <= result.confidence <= 1.0
    assert result.candidate_count >= len(result.top_k)


def test_reduce_cloud_empty_lexicon_rejected(sample_cloud) -> None:
    with pytest.raises(ValueError):
        reduce_cloud(sample_cloud, ())
