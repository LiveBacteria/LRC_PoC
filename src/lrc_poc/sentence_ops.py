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
_COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}
_FUNCTION_WORDS = {
    "above",
    "across",
    "after",
    "around",
    "before",
    "below",
    "between",
    "into",
    "near",
    "off",
    "onto",
    "over",
    "through",
    "under",
    "upon",
}


@dataclass(frozen=True)
class DefinitionMapping:
    token: str
    definition: str


def _normalize_definition(definition: str) -> str:
    normalized = re.sub(r"\s+", " ", definition.strip().lower())
    return normalized


def _tokenize_text(text: str) -> set[str]:
    tokens = {piece.lower() for piece in WORD_PATTERN.findall(text)}
    return {piece for piece in tokens if piece and piece not in _COMMON_STOPWORDS}


def _clip_definition(definition: str, *, max_words: int = 10) -> str:
    clipped = definition.strip()
    clipped = re.sub(r"\([^)]*\)", "", clipped)
    clipped = re.sub(r"^\s*[-–—]\s*", "", clipped)
    for delimiter in (";", ":", " -- ", ". "):
        if delimiter in clipped:
            clipped = clipped.split(delimiter, 1)[0].strip()
    words = clipped.split()
    if len(words) > max_words:
        clipped = " ".join(words[:max_words])
    return clipped.strip()


def _synset_signature_tokens(synset) -> set[str]:
    lemmas = [name.replace("_", " ") for name in synset.lemma_names()]
    hypernym_lemmas: list[str] = []
    for hypernym in synset.hypernyms()[:3]:
        hypernym_lemmas.extend(name.replace("_", " ") for name in hypernym.lemma_names())
    text = " ".join([synset.definition(), *synset.examples(), *lemmas, *hypernym_lemmas])
    return _tokenize_text(text)


def _select_synset_for_token(token: str, context_tokens: set[str]):
    synsets = list(wordnet_synsets(token.lower()))
    if not synsets:
        return None

    token_lower = token.lower()
    preferred_synsets = synsets
    if token_lower.endswith("ed") or token_lower.endswith("ing"):
        verb_synsets = [item for item in synsets if item.pos() == "v"]
        if verb_synsets:
            preferred_synsets = verb_synsets
    elif token_lower in _FUNCTION_WORDS:
        modifier_synsets = [item for item in synsets if item.pos() in {"r", "a", "s"}]
        if modifier_synsets:
            preferred_synsets = modifier_synsets

    context = set(context_tokens)
    context.discard(token_lower)
    if not context:
        return preferred_synsets[0]

    best = preferred_synsets[0]
    best_score = -10_000
    for index, synset in enumerate(preferred_synsets[:10]):
        signature = _synset_signature_tokens(synset)
        overlap = len(context & signature)
        score = overlap * 10 - index
        if score > best_score:
            best = synset
            best_score = score
    return best


def _literal_definition_for_token(
    token: str,
    synset,
    *,
    raw: bool = False,
) -> str:
    if synset is None:
        return token.lower()
    definition = synset.definition().strip()
    if not definition:
        return token.lower()
    return definition if raw else _clip_definition(definition, max_words=10)


def _semantic_definition_for_token(
    token: str,
    synset,
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
                    candidate = _clip_definition(str(definitions[0]), max_words=8)
                    if candidate:
                        return candidate
            except Exception:
                pass

    if synset is None:
        return token.lower()
    hypernyms = synset.hypernyms()
    if hypernyms and hypernyms[0].lemma_names():
        hypernym_name = hypernyms[0].lemma_names()[0].replace("_", " ").lower()
        return f"kind of {hypernym_name}"
    return _clip_definition(synset.definition(), max_words=8)


def _definition_for_token(
    token: str,
    synset,
    definition_style: str,
    pipeline: "LRCPipeline | None",
    mode: int,
) -> str:
    if definition_style == "semantic_relational":
        return _semantic_definition_for_token(
            token=token,
            synset=synset,
            pipeline=pipeline,
            mode=mode,
        )
    if definition_style == "literal_raw":
        return _literal_definition_for_token(token=token, synset=synset, raw=True)
    return _literal_definition_for_token(token=token, synset=synset, raw=False)


def expand_sentence_to_definitions(
    text: str,
    definition_style: str = "literal_first",
    pipeline: "LRCPipeline | None" = None,
    mode: int = 0,
) -> tuple[DefinitionMapping, ...]:
    """Replace each lexical token in a sentence with a configured definition style."""
    ensure_wordnet_available(download=False)
    mappings: list[DefinitionMapping] = []
    context_tokens = _tokenize_text(text)
    for match in WORD_PATTERN.finditer(text):
        token = match.group(0)
        synset = _select_synset_for_token(token=token, context_tokens=context_tokens)
        mappings.append(
            DefinitionMapping(
                token=token,
                definition=_definition_for_token(
                    token=token,
                    synset=synset,
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
) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    """Reduce expanded definition strings to lexical strings, optionally across spans."""
    definition_index = _build_definition_index(lexicon)
    reduced_phrases: list[str] = []
    segments: list[dict[str, object]] = []

    index = 0
    max_span = 3
    while index < len(mappings):
        selected_span_len = 1
        selected_phrase = mappings[index].token.lower()
        selected_method = "token_preserve"
        selected_confidence: float | None = None
        selected_definition_text = mappings[index].definition

        for span_len in range(min(max_span, len(mappings) - index), 0, -1):
            span = mappings[index : index + span_len]
            span_definition = " ".join(item.definition for item in span).strip()
            normalized = _normalize_definition(span_definition)

            exact = definition_index.get(normalized)
            if exact:
                selected_span_len = span_len
                selected_phrase = exact if span_len > 1 else span[0].token.lower()
                selected_method = "exact_definition_match"
                selected_confidence = 1.0
                selected_definition_text = span_definition
                break

            if use_semantic_fallback and span_len > 1:
                result = pipeline.run_once(span_definition, mode=mode, top_k=5, max_candidates=250)
                if result.confidence >= 0.45:
                    selected_span_len = span_len
                    selected_phrase = result.winner.word
                    selected_method = "semantic_span_compression"
                    selected_confidence = result.confidence
                    selected_definition_text = span_definition
                    break

        if selected_span_len == 1 and selected_method == "token_preserve":
            normalized = _normalize_definition(mappings[index].definition)
            exact = definition_index.get(normalized)
            if exact:
                selected_method = "exact_token_roundtrip"
                selected_confidence = 1.0
            elif use_semantic_fallback:
                result = pipeline.run_once(
                    mappings[index].definition,
                    mode=mode,
                    top_k=5,
                    max_candidates=200,
                )
                if result.confidence >= 0.35:
                    selected_phrase = result.winner.word
                    selected_method = "semantic_token_compression"
                    selected_confidence = result.confidence
                else:
                    selected_method = "token_preserve_low_confidence"
                    selected_confidence = result.confidence

        token_span = mappings[index : index + selected_span_len]
        reduced_phrases.append(selected_phrase)
        segments.append(
            {
                "start_index": index,
                "end_index": index + selected_span_len,
                "source_tokens": " ".join(item.token for item in token_span),
                "source_definition_text": selected_definition_text,
                "reduced_phrase": selected_phrase,
                "method": selected_method,
                "confidence": selected_confidence,
            }
        )
        index += selected_span_len

    return tuple(reduced_phrases), tuple(segments)


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
    reduced_terms, reduction_segments = reduce_definitions_to_terms(
        mappings,
        lexicon=lexicon,
        pipeline=pipeline,
        mode=mode,
        use_semantic_fallback=use_semantic_fallback,
    )
    reduced_sentence = " ".join(reduced_terms)
    token_level_reduced = [mapping.token.lower() for mapping in mappings]
    for segment in reduction_segments:
        start = int(segment["start_index"])
        end = int(segment["end_index"])
        reduced_phrase = str(segment["reduced_phrase"])
        for idx in range(start, min(end, len(token_level_reduced))):
            token_level_reduced[idx] = reduced_phrase

    legacy_mappings = [
        {
            "token": mapping.token,
            "definition": mapping.definition,
            "reduced_term": token_level_reduced[index],
        }
        for index, mapping in enumerate(mappings)
    ]

    return {
        "expanded_sentence": expanded_sentence,
        "reduced_sentence": reduced_sentence,
        "definition_style": definition_style,
        "semantic_fallback_reduction": use_semantic_fallback,
        "mappings": legacy_mappings,
        "token_mappings": [
            {
                "token": mapping.token,
                "definition": mapping.definition,
            }
            for mapping in mappings
        ],
        "reduction_segments": list(reduction_segments),
    }
