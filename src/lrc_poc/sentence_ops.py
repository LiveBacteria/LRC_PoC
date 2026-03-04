"""Sentence-level expand/reduce operators for M1 and M2 workflows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Sequence

from .candidates import generate_candidates
from .lexicon import ensure_wordnet_available, wordnet_synsets
from .models import SemanticCloud

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
_ALLOWED_DEFINITION_STYLES = ("literal_first", "cloud_compact", "literal_raw")


@dataclass(frozen=True)
class ReducerConfig:
    tau_score: float = 0.38
    tau_margin: float = 0.02
    max_candidates_per_window: int = 30
    max_semantic_window_tokens: int = 3
    max_semantic_window_chars: int = 180


DEFAULT_REDUCER_CONFIG = ReducerConfig()

_REDUCTION_DIAGNOSTIC_SCALAR_KEYS = (
    "total_token_units",
    "windows_considered",
    "exact_definition_matches",
    "windows_pruned_limit",
    "semantic_windows_attempted",
    "windows_no_candidates",
    "windows_rejected_score",
    "windows_rejected_margin",
    "accepted_replacements",
    "fallback_preserve_count",
)


def _empty_reduction_diagnostics() -> dict[str, object]:
    diagnostics: dict[str, object] = {key: 0 for key in _REDUCTION_DIAGNOSTIC_SCALAR_KEYS}
    diagnostics["windows_considered_by_span"] = {}
    diagnostics["accepted_replacements_by_span"] = {}
    return diagnostics


def _increment_span_bucket(bucket: dict[str, int], span_len: int) -> None:
    key = str(span_len)
    bucket[key] = int(bucket.get(key, 0)) + 1


def _merge_reduction_diagnostics(items: Sequence[dict[str, object]]) -> dict[str, object]:
    merged = _empty_reduction_diagnostics()
    for item in items:
        if not item:
            continue
        for key in _REDUCTION_DIAGNOSTIC_SCALAR_KEYS:
            merged[key] = int(merged[key]) + int(item.get(key, 0))
        for bucket_key in ("windows_considered_by_span", "accepted_replacements_by_span"):
            merged_bucket = merged[bucket_key]
            source_bucket = item.get(bucket_key, {})
            if not isinstance(merged_bucket, dict) or not isinstance(source_bucket, dict):
                continue
            for span_len, count in source_bucket.items():
                merged_bucket[str(span_len)] = int(merged_bucket.get(str(span_len), 0)) + int(count)
    return merged


def _serialize_reducer_config(config: ReducerConfig) -> dict[str, float | int]:
    return {
        "tau_score": float(config.tau_score),
        "tau_margin": float(config.tau_margin),
        "max_candidates_per_window": int(config.max_candidates_per_window),
        "max_semantic_window_tokens": int(config.max_semantic_window_tokens),
        "max_semantic_window_chars": int(config.max_semantic_window_chars),
    }


@dataclass(frozen=True)
class DefinitionMapping:
    token: str
    definition: str


@dataclass(frozen=True)
class TokenCloud:
    raw_definition: str
    short_definition: str
    definition_terms: tuple[str, ...]
    synonyms: tuple[str, ...]
    broader_concepts: tuple[str, ...]
    related_concepts: tuple[str, ...]
    preferred_pos: str


@dataclass(frozen=True)
class TokenUnit:
    token: str
    lowered: str
    cloud: TokenCloud


SEQUENCE_OPERATION_EXPAND = "expand"
SEQUENCE_OPERATION_REDUCE = "reduce"
SEQUENCE_OPERATION_RECURSIVE_REDUCE = "recursive_reduce"
DEFAULT_OPERATION_SEQUENCE: tuple[str, str] = (
    SEQUENCE_OPERATION_EXPAND,
    SEQUENCE_OPERATION_REDUCE,
)
DEFAULT_RECURSIVE_REDUCTION_LIMIT = 12
_SEQUENCE_OPERATION_ALIASES = {
    "expand": SEQUENCE_OPERATION_EXPAND,
    "expansion": SEQUENCE_OPERATION_EXPAND,
    "exp": SEQUENCE_OPERATION_EXPAND,
    "e": SEQUENCE_OPERATION_EXPAND,
    "reduce": SEQUENCE_OPERATION_REDUCE,
    "reduction": SEQUENCE_OPERATION_REDUCE,
    "compress": SEQUENCE_OPERATION_REDUCE,
    "contraction": SEQUENCE_OPERATION_REDUCE,
    "contract": SEQUENCE_OPERATION_REDUCE,
    "r": SEQUENCE_OPERATION_REDUCE,
    "recursive_reduce": SEQUENCE_OPERATION_RECURSIVE_REDUCE,
    "recursive_reduction": SEQUENCE_OPERATION_RECURSIVE_REDUCE,
    "reduce_recursive": SEQUENCE_OPERATION_RECURSIVE_REDUCE,
    "recursive_reduce_until_stable": SEQUENCE_OPERATION_RECURSIVE_REDUCE,
    "rr": SEQUENCE_OPERATION_RECURSIVE_REDUCE,
}


def _normalize_sequence_operation(operation: str) -> str:
    cleaned = re.sub(r"[\s\-]+", "_", str(operation).strip().lower()).strip("_")
    normalized = _SEQUENCE_OPERATION_ALIASES.get(cleaned)
    if normalized:
        return normalized
    allowed = ", ".join(sorted(set(_SEQUENCE_OPERATION_ALIASES.values())))
    raise ValueError(f"Unsupported sequence operation '{operation}'. Allowed operations: {allowed}")


def normalize_operation_sequence(operation_sequence: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalize sequence operations to canonical names."""
    normalized = tuple(_normalize_sequence_operation(item) for item in operation_sequence)
    if not normalized:
        raise ValueError("Operation sequence cannot be empty")
    return normalized


def parse_operation_sequence(sequence_text: str) -> tuple[str, ...]:
    """Parse comma/arrow/space-separated sequence text into canonical operations."""
    if not sequence_text or not sequence_text.strip():
        raise ValueError("Custom sequence is empty")
    chunks = re.split(r"\s*(?:,|;|\||->|\n)\s*", sequence_text.strip())
    raw_tokens: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        raw_tokens.extend(part for part in chunk.split() if part)
    return normalize_operation_sequence(raw_tokens)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_for_index(text: str) -> str:
    lowered = _normalize_text(text)
    return re.sub(r"[^a-z0-9\s'-]+", "", lowered)


def _tokenize_terms(text: str) -> set[str]:
    terms = {match.group(0).lower() for match in WORD_PATTERN.finditer(text)}
    return {term for term in terms if term and term not in _COMMON_STOPWORDS}


def _clip_definition(definition: str, *, max_words: int = 10) -> str:
    clipped = definition.strip()
    clipped = re.sub(r"\([^)]*\)", "", clipped)
    clipped = re.sub(r"^\s*[-]+\s*", "", clipped)
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
    return _tokenize_terms(text)


def _select_synset_for_token(token: str, context_tokens: set[str]):
    token_lower = token.lower()
    synsets = list(wordnet_synsets(token_lower))
    if not synsets:
        return None

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


def _build_token_cloud(token: str, synset) -> TokenCloud:
    token_lower = token.lower()
    if synset is None:
        return TokenCloud(
            raw_definition=token_lower,
            short_definition=token_lower,
            definition_terms=(token_lower,),
            synonyms=(token_lower,),
            broader_concepts=(),
            related_concepts=(),
            preferred_pos="n",
        )

    raw_definition = synset.definition().strip() or token_lower
    short_definition = _clip_definition(raw_definition, max_words=12) or token_lower
    definition_terms = tuple(sorted(_tokenize_terms(raw_definition) | _tokenize_terms(short_definition)))
    synonyms = tuple(sorted({name.replace("_", " ").lower() for name in synset.lemma_names()}))

    broader: set[str] = set()
    for hypernym in synset.hypernyms()[:4]:
        for name in hypernym.lemma_names():
            broader.add(name.replace("_", " ").lower())

    related: set[str] = set()
    for hyponym in synset.hyponyms()[:10]:
        for name in hyponym.lemma_names():
            related.add(name.replace("_", " ").lower())

    return TokenCloud(
        raw_definition=raw_definition,
        short_definition=short_definition,
        definition_terms=definition_terms or (token_lower,),
        synonyms=synonyms or (token_lower,),
        broader_concepts=tuple(sorted(broader)),
        related_concepts=tuple(sorted(related)),
        preferred_pos=synset.pos() or "n",
    )


def _serialize_cloud_for_expand(cloud: TokenCloud, *, definition_style: str) -> str:
    if definition_style == "literal_raw":
        return cloud.raw_definition

    if definition_style == "cloud_compact":
        pieces = [cloud.short_definition]
        if cloud.synonyms:
            pieces.append(" ".join(cloud.synonyms[:3]))
        if cloud.broader_concepts:
            pieces.append(" ".join(cloud.broader_concepts[:2]))
        return _normalize_text(" ".join(pieces))

    return cloud.short_definition


def expand_sentence_to_definitions(
    text: str,
    definition_style: str = "literal_first",
    pipeline: "LRCPipeline | None" = None,
    mode: int = 0,
) -> tuple[DefinitionMapping, ...]:
    """Replace each lexical token in a sentence with deterministic cloud serialization."""
    del pipeline, mode
    if definition_style not in _ALLOWED_DEFINITION_STYLES:
        allowed = ", ".join(_ALLOWED_DEFINITION_STYLES)
        raise ValueError(f"Unsupported definition_style '{definition_style}'. Allowed values: {allowed}")
    ensure_wordnet_available(download=False)
    mappings: list[DefinitionMapping] = []
    context_tokens = _tokenize_terms(text)

    for match in WORD_PATTERN.finditer(text):
        token = match.group(0)
        token_lower = token.lower()
        if token_lower in _COMMON_STOPWORDS:
            definition = token_lower
        else:
            synset = _select_synset_for_token(token=token, context_tokens=context_tokens)
            cloud = _build_token_cloud(token=token, synset=synset)
            definition = _serialize_cloud_for_expand(cloud, definition_style=definition_style)
        mappings.append(DefinitionMapping(token=token, definition=definition))
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


def _serialize_token_mappings(mappings: tuple[DefinitionMapping, ...]) -> list[dict[str, str]]:
    return [{"token": item.token, "definition": item.definition} for item in mappings]


def _build_definition_index(
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
) -> dict[str, str]:
    index: dict[str, str] = {}
    for entry in lexicon:
        if " " in entry.word.strip():
            continue
        normalized = _normalize_for_index(entry.definition)
        if not normalized:
            continue
        if normalized not in index or len(entry.word) < len(index[normalized]):
            index[normalized] = entry.word
    return index


def _build_units(text: str) -> tuple[list[str], list[int], tuple[TokenUnit, ...]]:
    chunks = NON_WORD_PATTERN.findall(text)
    token_chunk_indices: list[int] = []
    token_values: list[str] = []
    for chunk_index, chunk in enumerate(chunks):
        if WORD_PATTERN.fullmatch(chunk):
            token_chunk_indices.append(chunk_index)
            token_values.append(chunk)

    context_tokens = _tokenize_terms(" ".join(token_values))
    units: list[TokenUnit] = []
    for token in token_values:
        token_lower = token.lower()
        if token_lower in _COMMON_STOPWORDS:
            cloud = TokenCloud(
                raw_definition=token_lower,
                short_definition=token_lower,
                definition_terms=(token_lower,),
                synonyms=(token_lower,),
                broader_concepts=(),
                related_concepts=(),
                preferred_pos="n",
            )
        else:
            synset = _select_synset_for_token(token=token, context_tokens=context_tokens)
            cloud = _build_token_cloud(token=token, synset=synset)
        units.append(TokenUnit(token=token, lowered=token_lower, cloud=cloud))

    return chunks, token_chunk_indices, tuple(units)

def _superpose_clouds(units: Sequence[TokenUnit]) -> TokenCloud:
    if not units:
        return TokenCloud(
            raw_definition="",
            short_definition="",
            definition_terms=(),
            synonyms=(),
            broader_concepts=(),
            related_concepts=(),
            preferred_pos="n",
        )

    definition_terms: set[str] = set()
    synonyms: set[str] = set()
    broader: set[str] = set()
    related: set[str] = set()
    pos_counter: Counter[str] = Counter()
    short_parts: list[str] = []

    for unit in units:
        definition_terms.update(unit.cloud.definition_terms)
        synonyms.update(unit.cloud.synonyms)
        broader.update(unit.cloud.broader_concepts)
        related.update(unit.cloud.related_concepts)
        pos_counter.update((unit.cloud.preferred_pos,))
        if unit.cloud.short_definition:
            short_parts.append(unit.cloud.short_definition)

    preferred_pos = pos_counter.most_common(1)[0][0] if pos_counter else "n"
    short_definition = _normalize_text(" ".join(short_parts))
    return TokenCloud(
        raw_definition=short_definition,
        short_definition=short_definition,
        definition_terms=tuple(sorted(definition_terms)),
        synonyms=tuple(sorted(synonyms)),
        broader_concepts=tuple(sorted(broader)),
        related_concepts=tuple(sorted(related)),
        preferred_pos=preferred_pos,
    )


def _cloud_to_semantic_cloud(
    source_text: str,
    cloud: TokenCloud,
) -> SemanticCloud:
    key_terms = tuple(sorted((_tokenize_terms(source_text) | set(cloud.synonyms)) - _COMMON_STOPWORDS))[:12]
    definitions = cloud.definition_terms[:24] if cloud.definition_terms else (source_text.lower(),)
    synonyms = cloud.synonyms[:32]
    broader = cloud.broader_concepts[:24]
    related = cloud.related_concepts[:24]
    composed = " ".join(
        [
            f"source: {source_text.lower()}",
            "definitions: " + "; ".join(definitions),
            "synonyms: " + ", ".join(synonyms),
            "broader concepts: " + ", ".join(broader),
            "related concepts: " + ", ".join(related),
        ]
    )
    return SemanticCloud(
        source_text=source_text.lower(),
        key_terms=key_terms,
        definitions=definitions,
        synonyms=synonyms,
        broader_concepts=broader,
        related_concepts=related,
        constraints=(),
        negative_constraints=(),
        composed_cloud_text=composed,
        preferred_pos=cloud.preferred_pos,
    )


def _best_semantic_candidate(
    *,
    span_text: str,
    span_cloud: TokenCloud,
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
    reducer_config: ReducerConfig,
) -> tuple[str, float, float] | None:
    semantic_cloud = _cloud_to_semantic_cloud(source_text=span_text, cloud=span_cloud)
    candidates = generate_candidates(
        cloud=semantic_cloud,
        lexicon=lexicon,
        max_candidates=reducer_config.max_candidates_per_window,
        use_domain_heuristics=pipeline.use_domain_heuristics,
    )
    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        if " " in candidate.word.strip():
            continue
        breakdown = pipeline.scorer.score_candidate(semantic_cloud, candidate)
        scored.append((breakdown.total, candidate.word))
    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    best_score, best_word = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = max(0.0, best_score - second_score)
    return best_word, best_score, margin


def _compose_reduced_sentence(
    chunks: Sequence[str],
    token_chunk_indices: Sequence[int],
    reduction_segments: Sequence[dict[str, object]],
) -> str:
    if not token_chunk_indices:
        return "".join(chunks).strip()

    start_to_segment: dict[int, dict[str, object]] = {}
    skip_token_indices: set[int] = set()
    for segment in reduction_segments:
        start = int(segment["start_index"])
        end = int(segment["end_index"])
        method = str(segment["method"])
        if method == "no_match":
            continue
        start_to_segment[start] = segment
        for token_index in range(start + 1, end):
            skip_token_indices.add(token_index)

    pieces: list[str] = []
    token_index = 0
    for chunk_index, chunk in enumerate(chunks):
        if chunk_index in token_chunk_indices:
            if token_index in skip_token_indices:
                token_index += 1
                continue
            segment = start_to_segment.get(token_index)
            if segment is None:
                pieces.append(chunk)
                token_index += 1
                continue
            phrase = str(segment["reduced_phrase"])
            if chunk and chunk[0].isupper() and phrase:
                phrase = phrase[0].upper() + phrase[1:]
            pieces.append(phrase)
            token_index += 1
            continue
        pieces.append(chunk)

    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def reduce_expanded_sentence(
    text: str,
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
) -> tuple[str, tuple[dict[str, object], ...], int]:
    """Reduce an expanded sentence using windowed cloud superposition and lexical matching."""
    chunks, token_chunk_indices, units = _build_units(text)
    if not units:
        return text, (), 0

    definition_index = _build_definition_index(lexicon)
    reduction_segments: list[dict[str, object]] = []
    replacements = 0
    index = 0

    while index < len(units):
        selected: dict[str, object] | None = None
        max_span = len(units) - index
        for span_len in range(max_span, 1, -1):
            span_units = units[index : index + span_len]
            span_text = " ".join(item.lowered for item in span_units).strip()
            normalized_span = _normalize_for_index(span_text)

            exact = definition_index.get(normalized_span)
            if exact:
                selected = {
                    "start_index": index,
                    "end_index": index + span_len,
                    "source_tokens": " ".join(item.token for item in span_units),
                    "source_definition_text": span_text,
                    "reduced_phrase": exact,
                    "method": "exact_definition_match",
                    "confidence": 1.0,
                    "margin": 1.0,
                    "score": 1.0,
                }
                break

            if (
                span_len > DEFAULT_REDUCER_CONFIG.max_semantic_window_tokens
                or len(span_text) > DEFAULT_REDUCER_CONFIG.max_semantic_window_chars
            ):
                continue

            span_cloud = _superpose_clouds(span_units)
            winner = _best_semantic_candidate(
                span_text=span_text,
                span_cloud=span_cloud,
                lexicon=lexicon,
                pipeline=pipeline,
                reducer_config=DEFAULT_REDUCER_CONFIG,
            )
            if not winner:
                continue
            winner_word, score, margin = winner
            if score < DEFAULT_REDUCER_CONFIG.tau_score or margin < DEFAULT_REDUCER_CONFIG.tau_margin:
                continue

            selected = {
                "start_index": index,
                "end_index": index + span_len,
                "source_tokens": " ".join(item.token for item in span_units),
                "source_definition_text": span_text,
                "reduced_phrase": winner_word,
                "method": "semantic_window_match",
                "confidence": score,
                "margin": margin,
                "score": score,
            }
            break

        if selected is None:
            preserved = units[index]
            selected = {
                "start_index": index,
                "end_index": index + 1,
                "source_tokens": preserved.token,
                "source_definition_text": preserved.lowered,
                "reduced_phrase": preserved.lowered,
                "method": "no_match",
                "confidence": 0.0,
                "margin": 0.0,
                "score": 0.0,
            }
            reduction_segments.append(selected)
            index += 1
            continue

        reduction_segments.append(selected)
        replacements += 1
        index = int(selected["end_index"])

    reduced_text = _compose_reduced_sentence(
        chunks=chunks,
        token_chunk_indices=token_chunk_indices,
        reduction_segments=reduction_segments,
    )
    return reduced_text, tuple(reduction_segments), replacements


def _build_legacy_mappings(
    mappings: tuple[DefinitionMapping, ...],
    reduction_segments: Sequence[dict[str, object]],
) -> list[dict[str, str]]:
    segment_lookup: dict[str, str] = {}
    for segment in reduction_segments:
        normalized = _normalize_for_index(str(segment["source_definition_text"]))
        if normalized and normalized not in segment_lookup:
            segment_lookup[normalized] = str(segment["reduced_phrase"])

    rendered: list[dict[str, str]] = []
    for mapping in mappings:
        normalized = _normalize_for_index(mapping.definition)
        reduced_term = segment_lookup.get(normalized, mapping.token.lower())
        rendered.append(
            {
                "token": mapping.token,
                "definition": mapping.definition,
                "reduced_term": reduced_term,
            }
        )
    return rendered

def _reduce_definitions_to_terms_with_diagnostics(
    mappings: tuple[DefinitionMapping, ...],
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
    reducer_config: ReducerConfig,
) -> tuple[tuple[str, ...], tuple[dict[str, object], ...], dict[str, object]]:
    diagnostics = _empty_reduction_diagnostics()
    diagnostics["total_token_units"] = len(mappings)

    if not mappings:
        return (), (), diagnostics

    definition_index = _build_definition_index(lexicon)
    reduced_terms: list[str] = []
    segments: list[dict[str, object]] = []
    index = 0
    while index < len(mappings):
        selected: dict[str, object] | None = None
        max_span = len(mappings) - index
        for span_len in range(max_span, 1, -1):
            diagnostics["windows_considered"] = int(diagnostics["windows_considered"]) + 1
            considered_by_span = diagnostics["windows_considered_by_span"]
            if isinstance(considered_by_span, dict):
                _increment_span_bucket(considered_by_span, span_len)

            span = mappings[index : index + span_len]
            span_text = " ".join(item.definition for item in span)
            normalized_span = _normalize_for_index(span_text)
            exact = definition_index.get(normalized_span)
            if exact:
                diagnostics["exact_definition_matches"] = int(diagnostics["exact_definition_matches"]) + 1
                selected = {
                    "start_index": index,
                    "end_index": index + span_len,
                    "source_tokens": " ".join(item.token for item in span),
                    "source_definition_text": span_text,
                    "reduced_phrase": exact,
                    "method": "exact_definition_match",
                    "confidence": 1.0,
                    "margin": 1.0,
                    "score": 1.0,
                }
                break

            if (
                span_len > reducer_config.max_semantic_window_tokens
                or len(span_text) > reducer_config.max_semantic_window_chars
            ):
                diagnostics["windows_pruned_limit"] = int(diagnostics["windows_pruned_limit"]) + 1
                continue

            diagnostics["semantic_windows_attempted"] = int(diagnostics["semantic_windows_attempted"]) + 1
            units = _build_units(span_text)[2]
            span_cloud = _superpose_clouds(units)
            winner = _best_semantic_candidate(
                span_text=span_text,
                span_cloud=span_cloud,
                lexicon=lexicon,
                pipeline=pipeline,
                reducer_config=reducer_config,
            )
            if not winner:
                diagnostics["windows_no_candidates"] = int(diagnostics["windows_no_candidates"]) + 1
                continue
            winner_word, score, margin = winner
            below_score = score < reducer_config.tau_score
            below_margin = margin < reducer_config.tau_margin
            if below_score:
                diagnostics["windows_rejected_score"] = int(diagnostics["windows_rejected_score"]) + 1
            if below_margin:
                diagnostics["windows_rejected_margin"] = int(diagnostics["windows_rejected_margin"]) + 1
            selected = {
                "start_index": index,
                "end_index": index + span_len,
                "source_tokens": " ".join(item.token for item in span),
                "source_definition_text": span_text,
                "reduced_phrase": winner_word,
                "method": "semantic_window_best_guess" if (below_score or below_margin) else "semantic_window_match",
                "confidence": score,
                "margin": margin,
                "score": score,
            }
            break

        if selected is None:
            mapping = mappings[index]
            diagnostics["fallback_preserve_count"] = int(diagnostics["fallback_preserve_count"]) + 1
            selected = {
                "start_index": index,
                "end_index": index + 1,
                "source_tokens": mapping.token,
                "source_definition_text": mapping.definition,
                "reduced_phrase": mapping.token.lower(),
                "method": "no_match",
                "confidence": 0.0,
                "margin": 0.0,
                "score": 0.0,
            }
        else:
            span_len = int(selected["end_index"]) - int(selected["start_index"])
            accepted_by_span = diagnostics["accepted_replacements_by_span"]
            if isinstance(accepted_by_span, dict):
                _increment_span_bucket(accepted_by_span, span_len)
            diagnostics["accepted_replacements"] = int(diagnostics["accepted_replacements"]) + 1

        segments.append(selected)
        reduced_terms.append(str(selected["reduced_phrase"]))
        index = int(selected["end_index"])

    return tuple(reduced_terms), tuple(segments), diagnostics


def reduce_definitions_to_terms(
    mappings: tuple[DefinitionMapping, ...],
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
    mode: int,
    use_semantic_fallback: bool = False,
    reducer_config: ReducerConfig | None = None,
    return_diagnostics: bool = False,
) -> (
    tuple[tuple[str, ...], tuple[dict[str, object], ...]]
    | tuple[tuple[str, ...], tuple[dict[str, object], ...], dict[str, object]]
):
    """Reduce mapping definitions to lexical terms using min-span>=2 windows."""
    del mode, use_semantic_fallback
    config = reducer_config or DEFAULT_REDUCER_CONFIG
    reduced_terms, segments, diagnostics = _reduce_definitions_to_terms_with_diagnostics(
        mappings=mappings,
        lexicon=lexicon,
        pipeline=pipeline,
        reducer_config=config,
    )
    if return_diagnostics:
        return reduced_terms, segments, diagnostics
    return reduced_terms, segments


def _run_reduce_step(
    *,
    step_index: int,
    operation: str,
    current_text: str,
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
    mode: int,
    definition_style: str,
    use_semantic_fallback: bool,
    pending_mappings: tuple[DefinitionMapping, ...] | None,
    pending_expanded_sentence: str | None,
    reducer_config: ReducerConfig,
) -> tuple[dict[str, object], str, int]:
    del mode, use_semantic_fallback

    implicit_expand = False
    mappings_for_reduce = pending_mappings
    expanded_sentence = pending_expanded_sentence
    if mappings_for_reduce is None or expanded_sentence != current_text:
        implicit_expand = True
        mappings_for_reduce = expand_sentence_to_definitions(
            text=current_text,
            definition_style=definition_style,
            pipeline=pipeline,
            mode=0,
        )
        expanded_sentence = compose_expanded_sentence(current_text, mappings_for_reduce)

    # Reduce over token-definition units rather than full expanded word streams.
    # This preserves the declared expand->reduce semantics while avoiding quadratic
    # blowups on long expanded sentences.
    reduced_terms, reduction_segments, reduction_diagnostics = reduce_definitions_to_terms(
        mappings=mappings_for_reduce,
        lexicon=lexicon,
        pipeline=pipeline,
        mode=0,
        use_semantic_fallback=False,
        reducer_config=reducer_config,
        return_diagnostics=True,
    )
    reduced_sentence = " ".join(item for item in reduced_terms if item).strip()
    replacements = sum(1 for segment in reduction_segments if str(segment.get("method")) != "no_match")
    stripped_current = current_text.rstrip()
    if stripped_current and stripped_current[-1] in ".!?" and reduced_sentence and reduced_sentence[-1] not in ".!?":
        reduced_sentence = f"{reduced_sentence}{stripped_current[-1]}"
    step = {
        "step": step_index,
        "operation": operation,
        "input_text": current_text,
        "output_text": reduced_sentence,
        "expanded_sentence": expanded_sentence or current_text,
        "reduced_sentence": reduced_sentence,
        "token_mappings": _serialize_token_mappings(mappings_for_reduce),
        "mappings": _build_legacy_mappings(mappings_for_reduce, reduction_segments),
        "reduction_segments": list(reduction_segments),
        "reduction_diagnostics": reduction_diagnostics,
        "implicit_expand": implicit_expand,
        "replacement_count": replacements,
    }
    return step, reduced_sentence, replacements


def run_sentence_operation_sequence(
    text: str,
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
    mode: int,
    definition_style: str = "literal_first",
    use_semantic_fallback: bool = False,
    operation_sequence: Sequence[str] = DEFAULT_OPERATION_SEQUENCE,
    recursive_reduce_max_steps: int = DEFAULT_RECURSIVE_REDUCTION_LIMIT,
    reducer_config: ReducerConfig | None = None,
) -> dict[str, object]:
    """Run explicit expand/reduce operations with deterministic recursive reduction."""
    del mode
    if definition_style not in _ALLOWED_DEFINITION_STYLES:
        allowed = ", ".join(_ALLOWED_DEFINITION_STYLES)
        raise ValueError(f"Unsupported definition_style '{definition_style}'. Allowed values: {allowed}")
    normalized_sequence = normalize_operation_sequence(operation_sequence)
    if recursive_reduce_max_steps <= 0:
        raise ValueError("recursive_reduce_max_steps must be greater than zero")
    config = reducer_config or DEFAULT_REDUCER_CONFIG

    current_text = text
    steps: list[dict[str, object]] = []
    pending_mappings: tuple[DefinitionMapping, ...] | None = None
    pending_expanded_sentence: str | None = None
    last_expanded_sentence: str | None = None
    last_reduced_sentence: str | None = None

    for step_index, operation in enumerate(normalized_sequence, start=1):
        if operation == SEQUENCE_OPERATION_EXPAND:
            mappings = expand_sentence_to_definitions(
                text=current_text,
                definition_style=definition_style,
                pipeline=pipeline,
                mode=0,
            )
            expanded_sentence = compose_expanded_sentence(current_text, mappings)
            step = {
                "step": step_index,
                "operation": operation,
                "input_text": current_text,
                "output_text": expanded_sentence,
                "expanded_sentence": expanded_sentence,
                "token_mappings": _serialize_token_mappings(mappings),
            }
            steps.append(step)
            current_text = expanded_sentence
            pending_mappings = mappings
            pending_expanded_sentence = expanded_sentence
            last_expanded_sentence = expanded_sentence
            continue

        if operation == SEQUENCE_OPERATION_REDUCE:
            step, reduced_sentence, _ = _run_reduce_step(
                step_index=step_index,
                operation=operation,
                current_text=current_text,
                lexicon=lexicon,
                pipeline=pipeline,
                mode=0,
                definition_style=definition_style,
                use_semantic_fallback=use_semantic_fallback,
                pending_mappings=pending_mappings,
                pending_expanded_sentence=pending_expanded_sentence,
                reducer_config=config,
            )
            steps.append(step)
            current_text = reduced_sentence
            pending_mappings = None
            pending_expanded_sentence = None
            last_reduced_sentence = reduced_sentence
            continue

        if operation == SEQUENCE_OPERATION_RECURSIVE_REDUCE:
            local_current = current_text
            local_pending_mappings = pending_mappings
            local_pending_expanded = pending_expanded_sentence
            recursive_iterations: list[dict[str, object]] = []
            seen_outputs: set[str] = {_normalize_text(local_current)}
            stop_reason = "max_steps_reached"

            for depth in range(1, recursive_reduce_max_steps + 1):
                recursive_step, reduced_sentence, replacements = _run_reduce_step(
                    step_index=step_index,
                    operation=SEQUENCE_OPERATION_REDUCE,
                    current_text=local_current,
                    lexicon=lexicon,
                    pipeline=pipeline,
                    mode=0,
                    definition_style=definition_style,
                    use_semantic_fallback=use_semantic_fallback,
                    pending_mappings=local_pending_mappings,
                    pending_expanded_sentence=local_pending_expanded,
                    reducer_config=config,
                )
                recursive_step = dict(recursive_step)
                recursive_step["depth"] = depth
                recursive_iterations.append(recursive_step)

                normalized_output = _normalize_text(reduced_sentence)
                if replacements == 0:
                    stop_reason = "no_change"
                    local_current = reduced_sentence
                    break
                if normalized_output in seen_outputs:
                    stop_reason = "cycle_detected"
                    local_current = reduced_sentence
                    break

                seen_outputs.add(normalized_output)
                local_current = reduced_sentence
                local_pending_mappings = None
                local_pending_expanded = None

            final_recursive_step = recursive_iterations[-1]
            recursive_diag_totals = _merge_reduction_diagnostics(
                [item.get("reduction_diagnostics", {}) for item in recursive_iterations]
            )
            step = {
                "step": step_index,
                "operation": operation,
                "input_text": current_text,
                "output_text": local_current,
                "reduced_sentence": local_current,
                "recursive_iteration_count": len(recursive_iterations),
                "recursive_stop_reason": stop_reason,
                "recursive_iterations": recursive_iterations,
                "expanded_sentence": final_recursive_step["expanded_sentence"],
                "token_mappings": final_recursive_step["token_mappings"],
                "mappings": final_recursive_step["mappings"],
                "reduction_segments": final_recursive_step["reduction_segments"],
                "reduction_diagnostics": final_recursive_step["reduction_diagnostics"],
                "recursive_reduction_diagnostics_totals": recursive_diag_totals,
                "implicit_expand": final_recursive_step["implicit_expand"],
                "replacement_count": final_recursive_step["replacement_count"],
            }
            steps.append(step)
            current_text = local_current
            pending_mappings = None
            pending_expanded_sentence = None
            last_reduced_sentence = local_current
            continue

        raise ValueError(f"Unsupported operation '{operation}'")

    sequence_step_diagnostics: list[dict[str, object]] = []
    for step in steps:
        operation = str(step.get("operation", ""))
        if operation == SEQUENCE_OPERATION_REDUCE:
            diagnostic = step.get("reduction_diagnostics")
            if isinstance(diagnostic, dict):
                sequence_step_diagnostics.append(diagnostic)
            continue
        if operation == SEQUENCE_OPERATION_RECURSIVE_REDUCE:
            diagnostic = step.get("recursive_reduction_diagnostics_totals", step.get("reduction_diagnostics"))
            if isinstance(diagnostic, dict):
                sequence_step_diagnostics.append(diagnostic)

    return {
        "initial_text": text,
        "final_text": current_text,
        "operation_sequence": list(normalized_sequence),
        "steps": steps,
        "definition_style": definition_style,
        "recursive_reduce_max_steps": recursive_reduce_max_steps,
        "reducer_config": _serialize_reducer_config(config),
        "reduction_diagnostics": _merge_reduction_diagnostics(sequence_step_diagnostics),
        "last_expanded_sentence": last_expanded_sentence,
        "last_reduced_sentence": last_reduced_sentence,
    }

def run_sentence_definition_cycle(
    text: str,
    lexicon: tuple["LexiconEntry", ...] | list["LexiconEntry"],
    pipeline: "LRCPipeline",
    mode: int,
    definition_style: str = "literal_first",
    use_semantic_fallback: bool = False,
    reducer_config: ReducerConfig | None = None,
) -> dict[str, object]:
    """Execute one expand/reduce sentence cycle."""
    if definition_style not in _ALLOWED_DEFINITION_STYLES:
        allowed = ", ".join(_ALLOWED_DEFINITION_STYLES)
        raise ValueError(f"Unsupported definition_style '{definition_style}'. Allowed values: {allowed}")
    sequence_result = run_sentence_operation_sequence(
        text=text,
        lexicon=lexicon,
        pipeline=pipeline,
        mode=mode,
        definition_style=definition_style,
        use_semantic_fallback=use_semantic_fallback,
        operation_sequence=DEFAULT_OPERATION_SEQUENCE,
        reducer_config=reducer_config,
    )
    steps = sequence_result["steps"]
    expand_step = next(
        (step for step in steps if step["operation"] == SEQUENCE_OPERATION_EXPAND),
        None,
    )
    reduce_step = next(
        (step for step in reversed(steps) if step["operation"] == SEQUENCE_OPERATION_REDUCE),
        None,
    )

    expanded_sentence = str(expand_step["expanded_sentence"]) if expand_step else text
    reduced_sentence = str(reduce_step["reduced_sentence"]) if reduce_step else text
    mappings = list(reduce_step["mappings"]) if reduce_step else []
    token_mappings = list(reduce_step["token_mappings"]) if reduce_step else []
    reduction_segments = list(reduce_step["reduction_segments"]) if reduce_step else []
    reduction_diagnostics = dict(reduce_step["reduction_diagnostics"]) if reduce_step else _empty_reduction_diagnostics()

    return {
        "expanded_sentence": expanded_sentence,
        "reduced_sentence": reduced_sentence,
        "definition_style": definition_style,
        "reducer_config": dict(sequence_result["reducer_config"]),
        "mappings": mappings,
        "token_mappings": token_mappings,
        "reduction_segments": reduction_segments,
        "reduction_diagnostics": reduction_diagnostics,
    }
