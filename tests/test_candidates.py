"""Tests for candidate generation."""

from __future__ import annotations

from lrc_poc.candidates import generate_candidates


def test_generate_candidates_returns_ranked_subset(sample_cloud, sample_lexicon) -> None:
    candidates = generate_candidates(sample_cloud, sample_lexicon, max_candidates=2)
    assert len(candidates) == 2
    assert candidates[0].word in {"sorrow", "bereavement"}


def test_generate_candidates_bounded_output(sample_cloud, sample_lexicon) -> None:
    candidates = generate_candidates(sample_cloud, sample_lexicon, max_candidates=1)
    assert len(candidates) == 1


def test_generate_candidates_handles_sparse_cloud(sample_lexicon) -> None:
    from lrc_poc.models import SemanticCloud

    cloud = SemanticCloud(
        source_text="zzz",
        key_terms=("zzz",),
        definitions=("zzz",),
        synonyms=(),
        broader_concepts=(),
        related_concepts=(),
        constraints=(),
        negative_constraints=(),
        composed_cloud_text="source: zzz",
        preferred_pos="n",
    )
    candidates = generate_candidates(cloud, sample_lexicon, max_candidates=2)
    assert len(candidates) == 2
