"""Shared fixtures for LRC tests."""

from __future__ import annotations

import pytest

from lrc_poc.models import LexiconEntry, SemanticCloud


@pytest.fixture()
def sample_cloud() -> SemanticCloud:
    return SemanticCloud(
        source_text="grief",
        key_terms=("grief", "loss", "death"),
        definitions=(
            "intense sorrow caused by loss",
            "a feeling tied to emotional suffering after death",
        ),
        synonyms=("grief", "sorrow", "bereavement", "mourning"),
        broader_concepts=("emotion", "sadness"),
        related_concepts=("lament", "heartache"),
        constraints=("preserve original semantic intent",),
        negative_constraints=("joy",),
        composed_cloud_text=(
            "source: grief definitions: intense sorrow caused by loss "
            "feeling tied to emotional suffering after death"
        ),
        preferred_pos="n",
    )


@pytest.fixture()
def sample_lexicon() -> tuple[LexiconEntry, ...]:
    return (
        LexiconEntry(
            word="sorrow",
            lemma="sorrow",
            part_of_speech="n",
            definition="a feeling of deep distress caused by loss",
            examples=("her sorrow was overwhelming",),
            synonyms=("sorrow", "grief"),
            hypernyms=("sadness", "emotion"),
            hyponyms=("heartache",),
            synset_id="sorrow.n.01",
            sense_count=3,
        ),
        LexiconEntry(
            word="bereavement",
            lemma="bereavement",
            part_of_speech="n",
            definition="state of sorrow over the death of a loved one",
            examples=("the family was in bereavement",),
            synonyms=("bereavement", "mourning", "grief"),
            hypernyms=("sadness", "emotion"),
            hyponyms=("widowhood",),
            synset_id="bereavement.n.01",
            sense_count=1,
        ),
        LexiconEntry(
            word="happiness",
            lemma="happiness",
            part_of_speech="n",
            definition="a positive emotional state",
            examples=("she radiated happiness",),
            synonyms=("happiness", "joy"),
            hypernyms=("emotion",),
            hyponyms=("delight",),
            synset_id="happiness.n.01",
            sense_count=2,
        ),
    )
