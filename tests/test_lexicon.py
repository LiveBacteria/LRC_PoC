"""Tests for WordNet lexicon construction."""

from __future__ import annotations

import pytest

from lrc_poc.lexicon import LexiconResourceError, build_wordnet_lexicon, wordnet_synsets


def test_build_wordnet_lexicon_contains_required_fields() -> None:
    lexicon = build_wordnet_lexicon(limit_per_pos=5, max_entries=20)
    assert len(lexicon) > 0
    first = lexicon[0]
    assert first.word
    assert first.lemma
    assert first.part_of_speech
    assert first.definition
    assert isinstance(first.synonyms, tuple)


def test_build_wordnet_lexicon_is_deduplicated() -> None:
    lexicon = build_wordnet_lexicon(limit_per_pos=3, max_entries=30)
    keys = {(entry.word, entry.part_of_speech, entry.definition) for entry in lexicon}
    assert len(keys) == len(lexicon)


def test_build_wordnet_lexicon_populates_sense_count() -> None:
    lexicon = build_wordnet_lexicon(limit_per_pos=40, max_entries=300)
    assert any(entry.sense_count > 1 for entry in lexicon)

    single_token_entries = [
        entry
        for entry in lexicon
        if " " not in entry.lemma and entry.part_of_speech in {"n", "v", "a", "r"}
    ]
    assert len(single_token_entries) >= 20

    for entry in single_token_entries[:20]:
        expected = len(wordnet_synsets(entry.lemma, pos=entry.part_of_speech))
        assert expected >= 1
        assert entry.sense_count == expected


def test_missing_wordnet_raises_domain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import lrc_poc.lexicon as lexicon_module

    def _boom(*_args: object, **_kwargs: object) -> list[object]:
        raise LookupError("missing wordnet")

    monkeypatch.setattr(lexicon_module.wn, "synsets", _boom)
    with pytest.raises(LexiconResourceError):
        lexicon_module.ensure_wordnet_available(download=False)
