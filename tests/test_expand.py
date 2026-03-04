"""Tests for semantic expansion."""

from __future__ import annotations

import pytest

from lrc_poc.expand import expand_text


def test_expand_text_schema_completeness() -> None:
    cloud = expand_text("grief")
    assert cloud.source_text == "grief"
    assert len(cloud.key_terms) >= 1
    assert len(cloud.definitions) >= 1
    assert isinstance(cloud.synonyms, tuple)
    assert isinstance(cloud.broader_concepts, tuple)
    assert isinstance(cloud.related_concepts, tuple)
    assert isinstance(cloud.constraints, tuple)
    assert isinstance(cloud.negative_constraints, tuple)
    assert cloud.composed_cloud_text


def test_expand_text_normalizes_input() -> None:
    cloud = expand_text("  Grief   ")
    assert cloud.source_text == "grief"


def test_expand_text_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        expand_text("   ")
