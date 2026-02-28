"""Sentence-level definition expansion and reduction operators."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

from nltk.corpus import wordnet as wn

from .lexicon import ensure_wordnet_available

if TYPE_CHECKING:
    from .models import LexiconEntry
    from .pipeline import LRCPipeline

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']*")
NON_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']*|[^A-Za-z]+")


@dataclass(frozen=True)
class DefinitionMapping:
    token: str
    definition: str


def _normalize_definition(definition: str) -> str:
    normalized = re.sub(r"\s+", " ", definition.strip().lower())
    return normalized


def _definition_for_token(token: str) -> str:
    synsets = wn.synsets(token.lower())
    if not synsets:
        return token.lower()
    definition = synsets[0].definition().strip()
    return definition if definition else token.lower()


def expand_sentence_to_definitions(text: str) -> tuple[DefinitionMapping, ...]:
    """Replace each lexical token in a sentence with its literal WordNet definition."""
    ensure_wordnet_available(download=False)
    mappings: list[DefinitionMapping] = []
    for match in WORD_PATTERN.finditer(text):
        token = match.group(0)
        mappings.append(
            DefinitionMapping(
                token=token,
                definition=_definition_for_token(token),
            )
        )
    return tuple(mappings)


def compose_expanded_sentence(text: str, mappings: tuple[DefinitionMapping, ...]) -> str:
    """Return sentence text with lexical tokens replaced by definitions."""
    if not mappings:
        return text

    mapping_iter = iter(mappings)
    pieces: list[str] = []
    for chunk in NON_WORD_PATTERN.findall(text):
        if WORD_PATTERN.fullmatch(chunk):
            mapped = next(mapping_iter, None)
            pieces.append(mapped.definition if mapped else chunk)
        else:
            pieces.append(chunk)
    return "".join(pieces)


def _build_definition_index(
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
) -> dict[str, str]:
    index: dict[str, str] = {}
    for entry in lexicon:
        definition = _normalize_definition(entry.definition)
        if definition not in index or len(entry.word) < len(index[definition]):
            index[definition] = entry.word
    return index


def reduce_definitions_to_terms(
    mappings: tuple[DefinitionMapping, ...],
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
    mode: int,
) -> tuple[str, ...]:
    """Reduce literal definitions back to terms using exact definition matches first."""
    definition_index = _build_definition_index(lexicon)
    reduced: list[str] = []
    for mapping in mappings:
        normalized = _normalize_definition(mapping.definition)
        exact = definition_index.get(normalized)
        if exact:
            # Preserve original token on exact definition matches produced by
            # the literal expansion operator. This keeps expand/reduce cycles coherent.
            reduced.append(mapping.token.lower())
            continue
        # Fallback: semantic reduction over the definition slice.
        result = pipeline.run_once(mapping.definition, mode=mode, top_k=5, max_candidates=200)
        reduced.append(result.winner.word)
    return tuple(reduced)


def run_sentence_definition_cycle(
    text: str,
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
    mode: int,
) -> dict[str, object]:
    """Execute one expand/reduce sentence cycle aligned with definition-replacement semantics."""
    mappings = expand_sentence_to_definitions(text)
    expanded_sentence = compose_expanded_sentence(text, mappings)
    reduced_terms = reduce_definitions_to_terms(mappings, lexicon=lexicon, pipeline=pipeline, mode=mode)
    reduced_sentence = " ".join(reduced_terms)
    return {
        "expanded_sentence": expanded_sentence,
        "reduced_sentence": reduced_sentence,
        "mappings": [
            {
                "token": mapping.token,
                "definition": mapping.definition,
                "reduced_term": reduced_terms[index],
            }
            for index, mapping in enumerate(mappings)
        ],
    }
