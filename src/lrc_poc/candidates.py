"""Candidate generation for lexical reduction."""

from __future__ import annotations

import re

from .models import LexiconEntry, SemanticCloud

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _definition_tokens(definition: str) -> set[str]:
    return {match.group(0).lower() for match in WORD_PATTERN.finditer(definition)}


def generate_candidates(
    cloud: SemanticCloud,
    lexicon: tuple[LexiconEntry, ...] | list[LexiconEntry],
    max_candidates: int = 300,
) -> tuple[LexiconEntry, ...]:
    """Generate a bounded candidate pool based on lexical neighborhood."""
    if not lexicon:
        return ()

    key_terms = set(cloud.key_terms)
    synonym_terms = set(cloud.synonyms)
    broader_terms = set(cloud.broader_concepts)
    related_terms = set(cloud.related_concepts)
    cloud_terms = key_terms | synonym_terms | broader_terms | related_terms

    scored: list[tuple[int, LexiconEntry]] = []
    for entry in lexicon:
        score = 0
        if entry.word in key_terms:
            score += 8
        score += len(set(entry.synonyms) & synonym_terms) * 3
        score += len(set(entry.hypernyms) & broader_terms) * 2
        score += len(set(entry.hyponyms) & related_terms) * 1

        definition_overlap = len(_definition_tokens(entry.definition) & cloud_terms)
        score += min(definition_overlap, 5)

        if entry.part_of_speech == cloud.preferred_pos:
            score += 2

        if score > 0:
            scored.append((score, entry))

    if not scored:
        # Fallback for sparse clouds and unknown terms.
        fallback = sorted(lexicon, key=lambda item: (len(item.word.split()), item.word))
        return tuple(fallback[:max_candidates])

    scored.sort(key=lambda item: (-item[0], len(item[1].word.split()), item[1].word))

    deduped: list[LexiconEntry] = []
    seen_words: set[str] = set()
    for _, entry in scored:
        if entry.word in seen_words:
            continue
        seen_words.add(entry.word)
        deduped.append(entry)
        if len(deduped) >= max_candidates:
            break

    return tuple(deduped)
