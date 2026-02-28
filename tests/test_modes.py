"""Tests for mode-selectable pipeline behavior and fallback paths."""

from __future__ import annotations

import pytest

from lrc_poc.llm.base import BaseLLMProvider, LLMProviderConfig
from lrc_poc.pipeline import LRCPipeline


class FakeProvider(BaseLLMProvider):
    def __init__(self, configured: bool = True) -> None:
        key = "fake-key" if configured else ""
        super().__init__(
            LLMProviderConfig(provider_name="fake", model_name="fake-model", api_key=key)
        )

    def expand_cloud(self, source_text: str) -> dict[str, list[str] | str]:
        return {
            "definitions": [f"{source_text} enriched meaning"],
            "synonyms": [source_text, "emotion"],
            "broader_concepts": ["concept"],
            "related_concepts": ["association"],
            "constraints": ["preserve nuance"],
            "negative_constraints": ["unrelated"],
        }

    def propose_candidates(self, cloud_text: str, top_n: int = 20) -> list[str]:
        return ["happiness", "sorrow", "bereavement"][:top_n]

    def rerank_candidates(self, cloud_text: str, candidates: list[str]) -> list[str]:
        return ["happiness"] + [candidate for candidate in candidates if candidate != "happiness"]


def test_mode_0_deterministic(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    result = pipeline.run_once("grief", mode=0)
    assert result.mode_used == 0
    assert result.winner.word in {"sorrow", "bereavement", "happiness"}
    assert result.fallback_reason == ""


def test_mode_1_falls_back_without_provider(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=FakeProvider(configured=False))
    result = pipeline.run_once("grief", mode=1)
    assert result.mode_used == 1
    assert result.fallback_reason


def test_mode_2_uses_candidate_proposals(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=FakeProvider(configured=True))
    result = pipeline.run_once("grief", mode=2)
    top_words = [item.word for item in result.top_k]
    assert "happiness" in top_words


def test_mode_3_reranks_winner(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=FakeProvider(configured=True))
    result = pipeline.run_once("grief", mode=3)
    assert result.winner.word == "happiness"


def test_mode_4_selectable_with_fallback(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=FakeProvider(configured=False))
    result = pipeline.run_once("grief", mode=4)
    assert result.mode_used == 4
    assert result.winner.word in {"sorrow", "bereavement", "happiness"}
    assert "fallback" in result.fallback_reason.lower()


def test_invalid_mode_rejected(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    with pytest.raises(ValueError):
        pipeline.run_once("grief", mode=99)


def test_schema_parity_across_modes(sample_lexicon) -> None:
    modes = [0, 1, 2, 3, 4]
    providers = {
        0: None,
        1: FakeProvider(configured=False),
        2: FakeProvider(configured=True),
        3: FakeProvider(configured=True),
        4: FakeProvider(configured=False),
    }
    results = []
    for mode in modes:
        pipeline = LRCPipeline(lexicon=sample_lexicon, provider=providers[mode])
        results.append(pipeline.run_once("grief", mode=mode))

    for result in results:
        assert result.winner.word
        assert len(result.top_k) >= 1
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.fallback_reason, str)
