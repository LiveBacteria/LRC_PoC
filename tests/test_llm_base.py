"""Tests for shared LLM utility behavior."""

from __future__ import annotations

from lrc_poc.llm.base import normalize_str_list


def test_normalize_str_list_handles_dict_items() -> None:
    values = [{"value": "Grief"}, {"text": "Sorrow"}, {"word": "Bereavement"}]
    normalized = normalize_str_list(values)
    assert normalized == ["grief", "sorrow", "bereavement"]


def test_normalize_str_list_handles_dict_root() -> None:
    normalized = normalize_str_list({"candidate": "deep emotional loss"})
    assert normalized == ["deep emotional loss"]
