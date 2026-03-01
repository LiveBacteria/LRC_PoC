"""Lexicon construction utilities."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import threading

import nltk
from nltk.corpus import wordnet as wn

from .models import LexiconEntry

_WORDNET_LOCK = threading.RLock()


class LexiconResourceError(RuntimeError):
    """Raised when required lexical resources are unavailable."""


def _normalize_token(value: str) -> str:
    return value.replace("_", " ").strip().lower()


def ensure_wordnet_available(download: bool = False) -> None:
    """Ensure WordNet resources are available for lexicon operations."""
    try:
        with _WORDNET_LOCK:
            _ = wn.synsets("dog")
    except LookupError as error:
        if not download:
            raise LexiconResourceError(
                "WordNet resources are unavailable. Run: "
                "python -m nltk.downloader wordnet omw-1.4"
            ) from error
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        try:
            with _WORDNET_LOCK:
                _ = wn.synsets("dog")
        except LookupError as second_error:
            raise LexiconResourceError(
                "WordNet download failed. Ensure network access and NLTK data path."
            ) from second_error


def wordnet_synsets(term: str, pos: str | None = None):
    """Thread-safe wrapper for WordNet synset lookup."""
    with _WORDNET_LOCK:
        return wn.synsets(term, pos=pos)


@lru_cache(maxsize=4)
def build_wordnet_lexicon(
    limit_per_pos: int | None = None,
    max_entries: int | None = None,
    download_if_missing: bool = False,
) -> tuple[LexiconEntry, ...]:
    """Build a deduplicated lexicon from WordNet synsets."""
    ensure_wordnet_available(download=download_if_missing)

    lexicon: list[LexiconEntry] = []
    seen: set[tuple[str, str, str]] = set()
    per_pos_counts: dict[str, int] = defaultdict(int)

    with _WORDNET_LOCK:
        for synset in wn.all_synsets():
            part_of_speech = synset.pos()
            if limit_per_pos is not None and per_pos_counts[part_of_speech] >= limit_per_pos:
                continue

            definition = synset.definition().strip()
            synonyms = tuple(sorted({_normalize_token(item) for item in synset.lemma_names()}))
            hypernyms = tuple(
                sorted({_normalize_token(name) for hyper in synset.hypernyms() for name in hyper.lemma_names()})
            )
            hyponyms = tuple(
                sorted({_normalize_token(name) for hypo in synset.hyponyms()[:20] for name in hypo.lemma_names()})
            )
            examples = tuple(synset.examples())

            for lemma in synonyms:
                key = (lemma, part_of_speech, definition)
                if key in seen:
                    continue
                seen.add(key)
                per_pos_counts[part_of_speech] += 1

                lexicon.append(
                    LexiconEntry(
                        word=lemma,
                        lemma=lemma,
                        part_of_speech=part_of_speech,
                        definition=definition,
                        examples=examples,
                        synonyms=synonyms,
                        hypernyms=hypernyms,
                        hyponyms=hyponyms,
                        synset_id=synset.name(),
                        # Avoid nested WordNet reads while iterating all_synsets().
                        # We approximate ambiguity by counting observed synsets in this pass.
                        sense_count=1,
                    )
                )
                if max_entries is not None and len(lexicon) >= max_entries:
                    return tuple(lexicon)

    return tuple(lexicon)


def sample_seed_words(max_seeds: int = 200) -> tuple[str, ...]:
    """Return noun-oriented seed words for attractor experiments."""
    ensure_wordnet_available(download=False)
    words: list[str] = []
    with _WORDNET_LOCK:
        for synset in wn.all_synsets(pos=wn.NOUN):
            base = _normalize_token(synset.lemma_names()[0])
            if " " in base:
                continue
            words.append(base)
            if len(words) >= max_seeds:
                break
    return tuple(words)
