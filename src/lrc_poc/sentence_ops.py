"""Sentence-level definition expansion and reduction operators."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

from .lexicon import ensure_wordnet_available, wordnet_synsets

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


def _clip_definition(definition: str) -> str:
    clipped = definition.strip()
    if ";" in clipped:
        clipped = clipped.split(";", 1)[0].strip()
    if " -- " in clipped:
        clipped = clipped.split(" -- ", 1)[0].strip()
    return clipped


def _literal_definition_for_token(token: str, raw: bool = False) -> str:
    synsets = wordnet_synsets(token.lower())
    if not synsets:
        return token.lower()
    definition = synsets[0].definition().strip()
    if not definition:
        return token.lower()
    return definition if raw else _clip_definition(definition)


def _semantic_definition_for_token(
    token: str,
    pipeline: "LRCPipeline | None",
    mode: int,
) -> str:
    if pipeline is not None and mode in {1, 2, 3, 4}:
        provider = getattr(pipeline, "provider", None)
        if provider is not None and provider.is_configured():
            try:
                payload = provider.expand_cloud(token.lower())
                definitions = payload.get("definitions", [])
                if isinstance(definitions, list) and definitions:
                    candidate = _clip_definition(str(definitions[0]))
                    if candidate:
                        return candidate
            except Exception:
                pass
    return _literal_definition_for_token(token, raw=False)


def _definition_for_token(
    token: str,
    definition_style: str,
    pipeline: "LRCPipeline | None",
    mode: int,
) -> str:
    if definition_style == "semantic_relational":
        return _semantic_definition_for_token(token=token, pipeline=pipeline, mode=mode)
    if definition_style == "literal_raw":
        return _literal_definition_for_token(token=token, raw=True)
    return _literal_definition_for_token(token=token, raw=False)


def expand_sentence_to_definitions(
    text: str,
    definition_style: str = "literal_first",
    pipeline: "LRCPipeline | None" = None,
    mode: int = 0,
) -> tuple[DefinitionMapping, ...]:
    """Replace each lexical token in a sentence with a configured definition style."""
    ensure_wordnet_available(download=False)
    mappings: list[DefinitionMapping] = []
    for match in WORD_PATTERN.finditer(text):
        token = match.group(0)
        mappings.append(
            DefinitionMapping(
                token=token,
                definition=_definition_for_token(
                    token=token,
                    definition_style=definition_style,
                    pipeline=pipeline,
                    mode=mode,
                ),
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
    use_semantic_fallback: bool = False,
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
        if use_semantic_fallback:
            # Optional semantic fallback over the definition slice.
            result = pipeline.run_once(mapping.definition, mode=mode, top_k=5, max_candidates=200)
            reduced.append(result.winner.word)
        else:
            reduced.append(mapping.token.lower())
    return tuple(reduced)


def run_sentence_definition_cycle(
    text: str,
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
    mode: int,
    definition_style: str = "literal_first",
    use_semantic_fallback: bool = False,
) -> dict[str, object]:
    """Execute one expand/reduce sentence cycle aligned with definition-replacement semantics."""
    mappings = expand_sentence_to_definitions(
        text=text,
        definition_style=definition_style,
        pipeline=pipeline,
        mode=mode,
    )
    expanded_sentence = compose_expanded_sentence(text, mappings)
    reduced_terms = reduce_definitions_to_terms(
        mappings,
        lexicon=lexicon,
        pipeline=pipeline,
        mode=mode,
        use_semantic_fallback=use_semantic_fallback,
    )
    reduced_sentence = " ".join(reduced_terms)
    return {
        "expanded_sentence": expanded_sentence,
        "reduced_sentence": reduced_sentence,
        "definition_style": definition_style,
        "semantic_fallback_reduction": use_semantic_fallback,
        "mappings": [
            {
                "token": mapping.token,
                "definition": mapping.definition,
                "reduced_term": reduced_terms[index],
            }
            for index, mapping in enumerate(mappings)
        ],
    }
