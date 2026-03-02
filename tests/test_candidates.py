"""Tests for candidate generation."""

from __future__ import annotations

import lrc_poc.candidates as candidates_module
from lrc_poc.candidates import generate_candidates
from lrc_poc.models import LexiconEntry, SemanticCloud


def test_generate_candidates_returns_ranked_subset(sample_cloud, sample_lexicon) -> None:
    candidates = generate_candidates(sample_cloud, sample_lexicon, max_candidates=2)
    assert len(candidates) == 2
    assert candidates[0].word in {"sorrow", "bereavement"}


def test_generate_candidates_bounded_output(sample_cloud, sample_lexicon) -> None:
    candidates = generate_candidates(sample_cloud, sample_lexicon, max_candidates=1)
    assert len(candidates) == 1


def test_generate_candidates_handles_sparse_cloud(sample_lexicon) -> None:
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


def test_domain_heuristics_default_off(monkeypatch) -> None:
    monkeypatch.setattr(candidates_module, "wordnet_synsets", lambda *_args, **_kwargs: [])

    cloud = SemanticCloud(
        source_text="a feeling tied to death and loss",
        key_terms=("death", "loss", "feeling"),
        definitions=("a feeling tied to death and loss",),
        synonyms=(),
        broader_concepts=(),
        related_concepts=(),
        constraints=(),
        negative_constraints=(),
        composed_cloud_text="a feeling tied to death and loss",
        preferred_pos="n",
    )
    lexicon = (
        LexiconEntry(
            word="bereavement",
            lemma="bereavement",
            part_of_speech="n",
            definition="social custom for a family event",
            sense_count=1,
        ),
        LexiconEntry(
            word="instrument",
            lemma="instrument",
            part_of_speech="n",
            definition="a tool used after a major loss",
            sense_count=1,
        ),
    )

    candidates = generate_candidates(cloud, lexicon, max_candidates=2)
    assert candidates[0].word == "instrument"


def test_domain_heuristics_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(candidates_module, "wordnet_synsets", lambda *_args, **_kwargs: [])

    cloud = SemanticCloud(
        source_text="a feeling tied to death and loss",
        key_terms=("death", "loss", "feeling"),
        definitions=("a feeling tied to death and loss",),
        synonyms=(),
        broader_concepts=(),
        related_concepts=(),
        constraints=(),
        negative_constraints=(),
        composed_cloud_text="a feeling tied to death and loss",
        preferred_pos="n",
    )
    lexicon = (
        LexiconEntry(
            word="bereavement",
            lemma="bereavement",
            part_of_speech="n",
            definition="social custom for a family event",
            sense_count=1,
        ),
        LexiconEntry(
            word="instrument",
            lemma="instrument",
            part_of_speech="n",
            definition="a tool used after a major loss",
            sense_count=1,
        ),
    )

    candidates = generate_candidates(
        cloud,
        lexicon,
        max_candidates=2,
        use_domain_heuristics=True,
    )
    assert candidates[0].word == "bereavement"
