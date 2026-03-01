"""Tests for sentence-level definition expansion/reduction operations."""

from __future__ import annotations

from lrc_poc.pipeline import LRCPipeline
from lrc_poc.sentence_ops import (
    compose_expanded_sentence,
    expand_sentence_to_definitions,
    reduce_definitions_to_terms,
    run_sentence_definition_cycle,
)


def test_expand_sentence_to_definitions_basic() -> None:
    mappings = expand_sentence_to_definitions("The cat jumped over the dog.")
    tokens = [item.token.lower() for item in mappings]
    assert "cat" in tokens
    assert "dog" in tokens
    assert all(item.definition for item in mappings)
    cat_defs = [item.definition for item in mappings if item.token.lower() == "cat"]
    assert cat_defs
    assert ";" not in cat_defs[0]


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
    assert isinstance(cycle["token_mappings"], list)
    assert isinstance(cycle["reduction_segments"], list)


def test_definition_styles_change_expansion_output(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    text = "The cat jumped over the dog."
    literal = run_sentence_definition_cycle(
        text=text,
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="literal_first",
    )
    relational = run_sentence_definition_cycle(
        text=text,
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="semantic_relational",
    )
    raw = run_sentence_definition_cycle(
        text=text,
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="literal_raw",
    )
    assert literal["expanded_sentence"] != relational["expanded_sentence"]
    assert raw["expanded_sentence"] != literal["expanded_sentence"]
    assert ";" in raw["expanded_sentence"]
    assert ";" not in literal["expanded_sentence"]


def test_definition_cycle_preserves_original_tokens_on_exact_matches(sample_lexicon) -> None:
    from lrc_poc.models import LexiconEntry

    mappings = expand_sentence_to_definitions("cat jumped over dog")
    custom_lexicon = tuple(
        LexiconEntry(
            word=item.token.lower(),
            lemma=item.token.lower(),
            part_of_speech="n",
            definition=item.definition,
            sense_count=1,
        )
        for item in mappings
    )
    pipeline = LRCPipeline(lexicon=custom_lexicon, provider=None)
    reduced, segments = reduce_definitions_to_terms(
        mappings=mappings,
        lexicon=custom_lexicon,
        pipeline=pipeline,
        mode=0,
    )
    assert " ".join(reduced) == "cat jumped over dog"
    assert len(segments) == len(mappings)
