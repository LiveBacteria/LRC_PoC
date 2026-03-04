"""Tests for sentence-level definition expansion/reduction operations."""

from __future__ import annotations

import pytest

from lrc_poc.pipeline import LRCPipeline
from lrc_poc.sentence_ops import (
    compose_expanded_sentence,
    expand_sentence_to_definitions,
    parse_operation_sequence,
    reduce_definitions_to_terms,
    run_sentence_definition_cycle,
    run_sentence_operation_sequence,
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
    assert isinstance(cycle["mappings"], list)
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
    cloud = run_sentence_definition_cycle(
        text=text,
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="cloud_compact",
    )
    raw = run_sentence_definition_cycle(
        text=text,
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="literal_raw",
    )
    assert literal["expanded_sentence"] != cloud["expanded_sentence"]
    assert raw["expanded_sentence"] != literal["expanded_sentence"]
    assert "synonyms " not in cloud["expanded_sentence"].lower()
    assert "broader " not in cloud["expanded_sentence"].lower()
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
    assert reduced
    assert all(term and " " not in term for term in reduced)
    assert len(segments) <= len(mappings)


def test_parse_operation_sequence_supports_custom_text() -> None:
    sequence = parse_operation_sequence("expand -> expand, reduce")
    assert sequence == ("expand", "expand", "reduce")


def test_parse_operation_sequence_supports_recursive_reduce_aliases() -> None:
    sequence = parse_operation_sequence("expand -> reduce_recursive")
    assert sequence == ("expand", "recursive_reduce")


def test_parse_operation_sequence_rejects_unknown_operations() -> None:
    with pytest.raises(ValueError):
        parse_operation_sequence("expand, mutate, reduce")


def test_run_sentence_operation_sequence_respects_operation_order(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    sequence = ("expand", "expand", "reduce")
    result = run_sentence_operation_sequence(
        text="grief",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        operation_sequence=sequence,
    )
    assert tuple(step["operation"] for step in result["steps"]) == sequence
    assert result["final_text"] == result["steps"][-1]["output_text"]


def test_run_sentence_operation_sequence_uses_implicit_expand_for_reduce(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    result = run_sentence_operation_sequence(
        text="grief",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        operation_sequence=("reduce", "reduce"),
    )
    assert result["steps"][0]["implicit_expand"] is True
    assert result["steps"][1]["implicit_expand"] is True


def test_run_sentence_operation_sequence_recursive_reduce_reports_stop_reason(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    result = run_sentence_operation_sequence(
        text="grief",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        operation_sequence=("expand", "recursive_reduce"),
        recursive_reduce_max_steps=6,
    )
    recursive_step = result["steps"][-1]
    assert recursive_step["operation"] == "recursive_reduce"
    assert recursive_step["recursive_iteration_count"] >= 1
    assert recursive_step["recursive_iteration_count"] <= 6
    assert recursive_step["recursive_stop_reason"] in {"no_change", "cycle_detected", "max_steps_reached"}


def test_run_sentence_definition_cycle_matches_default_sequence(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    legacy = run_sentence_definition_cycle(
        text="grief",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
    )
    sequenced = run_sentence_operation_sequence(
        text="grief",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        operation_sequence=("expand", "reduce"),
    )
    assert legacy["expanded_sentence"] == sequenced["last_expanded_sentence"]
    assert legacy["reduced_sentence"] == sequenced["last_reduced_sentence"]


def test_expand_reduce_does_not_append_expansion_tail(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    text = "The cat jumped over the dog."
    result = run_sentence_operation_sequence(
        text=text,
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="literal_first",
        operation_sequence=("expand", "reduce"),
    )
    reduced = str(result["final_text"]).lower()
    assert not reduced.startswith("the cat jumped over the dog ")
    assert "the cat jumped over the dog soft" not in reduced


def test_reduce_windows_are_greedy_and_dynamic_not_fixed_span(sample_lexicon) -> None:
    from lrc_poc.models import LexiconEntry

    mappings = expand_sentence_to_definitions("alpha beta gamma delta")
    combo_definition = " ".join(item.definition for item in mappings)
    lexicon = (
        LexiconEntry(
            word="combo",
            lemma="combo",
            part_of_speech="n",
            definition=combo_definition,
            sense_count=1,
        ),
    )
    pipeline = LRCPipeline(lexicon=lexicon, provider=None)
    reduced_terms, segments = reduce_definitions_to_terms(
        mappings=mappings,
        lexicon=lexicon,
        pipeline=pipeline,
        mode=0,
    )
    assert reduced_terms == ("combo",)
    assert len(segments) == 1
    assert segments[0]["end_index"] - segments[0]["start_index"] == len(mappings)
    assert segments[0]["method"] == "exact_definition_match"


def test_reduce_replacements_only_use_spans_of_two_or_more(sample_lexicon) -> None:
    from lrc_poc.models import LexiconEntry

    mappings = expand_sentence_to_definitions("alpha beta gamma")
    combo_definition = " ".join(item.definition for item in mappings[:2])
    lexicon = (
        LexiconEntry(
            word="pair",
            lemma="pair",
            part_of_speech="n",
            definition=combo_definition,
            sense_count=1,
        ),
    )
    pipeline = LRCPipeline(lexicon=lexicon, provider=None)
    result = run_sentence_operation_sequence(
        text="alpha beta gamma",
        lexicon=lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="literal_first",
        operation_sequence=("expand", "reduce"),
    )
    reduce_step = result["steps"][-1]
    for segment in reduce_step["reduction_segments"]:
        if segment["method"] == "no_match":
            continue
        assert (segment["end_index"] - segment["start_index"]) >= 2


def test_expand_on_expand_increases_cloud_mass(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    result = run_sentence_operation_sequence(
        text="cat",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="cloud_compact",
        operation_sequence=("expand", "expand"),
    )
    first_expand = str(result["steps"][0]["output_text"])
    second_expand = str(result["steps"][1]["output_text"])
    assert len(second_expand.split()) > len(first_expand.split())


def test_recursive_reduce_stops_and_additional_reduce_is_noop(sample_lexicon) -> None:
    from lrc_poc.models import LexiconEntry

    mappings = expand_sentence_to_definitions("alpha beta")
    combo_definition = " ".join(item.definition for item in mappings)
    lexicon = (
        LexiconEntry(
            word="alphabeta",
            lemma="alphabeta",
            part_of_speech="n",
            definition=combo_definition,
            sense_count=1,
        ),
    )
    pipeline = LRCPipeline(lexicon=lexicon, provider=None)
    result = run_sentence_operation_sequence(
        text="alpha beta",
        lexicon=lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="literal_first",
        operation_sequence=("expand", "recursive_reduce"),
        recursive_reduce_max_steps=10,
    )
    recursive_step = result["steps"][-1]
    assert recursive_step["operation"] == "recursive_reduce"
    assert recursive_step["recursive_stop_reason"] in {"no_change", "cycle_detected", "max_steps_reached"}

    post = run_sentence_operation_sequence(
        text=str(result["final_text"]),
        lexicon=lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="literal_first",
        operation_sequence=("reduce",),
    )
    assert post["steps"][0]["replacement_count"] == 0
    assert str(post["final_text"]) == str(result["final_text"])


def test_expand_reduce_output_never_includes_kind_of_filler(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    result = run_sentence_operation_sequence(
        text="The cat jumped over the dog.",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="cloud_compact",
        operation_sequence=("expand", "reduce"),
    )
    assert "kind of" not in str(result["final_text"]).lower()


def test_reduce_step_emits_reduction_diagnostics(sample_lexicon) -> None:
    from lrc_poc.models import LexiconEntry

    mappings = expand_sentence_to_definitions("alpha beta gamma")
    pair_definition = " ".join(item.definition for item in mappings[:2])
    lexicon = (
        LexiconEntry(
            word="pair",
            lemma="pair",
            part_of_speech="n",
            definition=pair_definition,
            sense_count=1,
        ),
    )
    pipeline = LRCPipeline(lexicon=lexicon, provider=None)
    result = run_sentence_operation_sequence(
        text="alpha beta gamma",
        lexicon=lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="literal_first",
        operation_sequence=("expand", "reduce"),
    )
    reduce_step = result["steps"][-1]
    diagnostics = reduce_step["reduction_diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["windows_considered"] >= 1
    assert diagnostics["accepted_replacements"] >= 1
    assert diagnostics["fallback_preserve_count"] >= 0
    assert isinstance(diagnostics["windows_considered_by_span"], dict)
    assert isinstance(diagnostics["accepted_replacements_by_span"], dict)


def test_recursive_reduce_emits_aggregate_diagnostics(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    result = run_sentence_operation_sequence(
        text="The cat jumped over the dog.",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="cloud_compact",
        operation_sequence=("expand", "recursive_reduce"),
        recursive_reduce_max_steps=6,
    )
    recursive_step = result["steps"][-1]
    assert recursive_step["operation"] == "recursive_reduce"
    assert isinstance(recursive_step["reduction_diagnostics"], dict)
    assert isinstance(recursive_step["recursive_reduction_diagnostics_totals"], dict)
    assert recursive_step["recursive_reduction_diagnostics_totals"]["windows_considered"] >= 1


def test_definition_cycle_exposes_reducer_config_and_diagnostics(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    cycle = run_sentence_definition_cycle(
        text="The cat jumped over the dog.",
        lexicon=sample_lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style="literal_first",
    )
    assert isinstance(cycle["reducer_config"], dict)
    assert isinstance(cycle["reduction_diagnostics"], dict)
    assert cycle["reducer_config"]["max_candidates_per_window"] >= 1
