"""Tests for deterministic-only pipeline behavior."""

from __future__ import annotations

import pytest

from lrc_poc.pipeline import LRCPipeline


def test_mode_0_deterministic(sample_lexicon) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    result = pipeline.run_once("grief", mode=0)
    assert result.mode_used == 0
    assert result.winner.word in {"sorrow", "bereavement", "happiness"}
    assert result.fallback_reason == ""


@pytest.mark.parametrize("invalid_mode", [1, 2, 3, 4, 99])
def test_nonzero_modes_rejected(sample_lexicon, invalid_mode: int) -> None:
    pipeline = LRCPipeline(lexicon=sample_lexicon, provider=None)
    with pytest.raises(ValueError, match="deterministic mode 0"):
        pipeline.run_once("grief", mode=invalid_mode)
