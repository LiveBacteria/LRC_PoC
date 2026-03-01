"""Candidate generation for lexical reduction."""

from __future__ import annotations

import re

from .lexicon import wordnet_synsets
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

    seed_terms: set[str] = (
        set(cloud.key_terms)
        | set(cloud.synonyms)
        | set(cloud.broader_concepts)
        | set(cloud.related_concepts)
    )
    key_token_terms: set[str] = set()
    for item in seed_terms:
        key_token_terms.update(_definition_tokens(item))

    # Domain heuristics for frequent phrase-level concepts.
    if {"death", "loss"} <= key_token_terms:
        seed_terms.update({"bereavement", "mourning", "grief", "sorrow"})
    if {"positive", "pleasure"} <= key_token_terms or "contentment" in key_token_terms:
        seed_terms.update({"happiness", "joy", "contentment", "delight"})
    if {"anger", "temper"} <= key_token_terms:
        seed_terms.update({"anger", "rage", "fury", "wrath"})

    # Expand neighborhood directly from WordNet around key terms.
    for term in tuple(seed_terms):
        for synset in wordnet_synsets(term)[:6]:
            for lemma in synset.lemma_names():
                seed_terms.add(lemma.replace("_", " ").lower())
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemma_names():
                    seed_terms.add(lemma.replace("_", " ").lower())
            for hyponym in synset.hyponyms()[:12]:
                for lemma in hyponym.lemma_names():
                    seed_terms.add(lemma.replace("_", " ").lower())

    key_terms = set(cloud.key_terms)
    synonym_terms = set(cloud.synonyms)
    broader_terms = set(cloud.broader_concepts)
    related_terms = set(cloud.related_concepts)
    cloud_terms = key_terms | synonym_terms | broader_terms | related_terms | key_token_terms

    scored: list[tuple[int, LexiconEntry]] = []
    for entry in lexicon:
        score = 0
        if entry.word in seed_terms:
            score += 12
        score += len(set(entry.synonyms) & synonym_terms) * 3
        score += len(set(entry.hypernyms) & broader_terms) * 2
        score += len(set(entry.hyponyms) & related_terms) * 1

        definition_overlap = len(_definition_tokens(entry.definition) & cloud_terms)
        score += min(definition_overlap, 8) * 2

        if entry.part_of_speech == cloud.preferred_pos:
            score += 6
        elif cloud.preferred_pos in {"n", "v", "a", "r"}:
            score -= 2

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
        if len(deduped) >= max_candidates * 2:
            break

    same_pos = [entry for entry in deduped if entry.part_of_speech == cloud.preferred_pos]
    if len(same_pos) >= max(50, max_candidates // 3):
        return tuple(same_pos[:max_candidates])

    return tuple(deduped[:max_candidates])
