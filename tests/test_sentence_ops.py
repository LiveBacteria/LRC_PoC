"""Tests for sentence-level definition expansion/reduction operations."""

from __future__ import annotations

from lrc_poc.pipeline import LRCPipeline
from lrc_poc.sentence_ops import (
    compose_expanded_sentence,
    expand_sentence_to_definitions,
    run_sentence_definition_cycle,
)


def test_expand_sentence_to_definitions_basic() -> None:
    mappings = expand_sentence_to_definitions("The cat jumped over the dog.")
    tokens = [item.token.lower() for item in mappings]
    assert "cat" in tokens
    assert "dog" in tokens
    assert all(item.definition for item in mappings)


def test_compose_expanded_sentence_replaces_words() -> None:
    mappings = expand_sentence_to_definitions("cat")
    expanded = compose_expanded_sentence("cat", mappings)
    assert expanded != "cat"
    assert isinstance(expanded, str)


def test_run_sentence_definition_cycle(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    cycle = run_sentence_definition_cycle(
        text="grief",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
    )
    assert cycle["expanded_sentence"]
    assert cycle["reduced_sentence"]
    assert isinstance(cycle["mappings"], list)
