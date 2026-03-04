"""Semantic expansion helpers for building structured clouds."""

from __future__ import annotations

from collections import Counter
import re

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .lexicon import LexiconResourceError, ensure_wordnet_available, wordnet_synsets
from .models import SemanticCloud

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _safe_tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [token for token in tokens if len(token) > 2 and token not in ENGLISH_STOP_WORDS]


def _safe_pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    if not tokens:
        return []
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        return [(token, "NN") for token in tokens]


def extract_key_terms(text: str, max_terms: int = 12) -> tuple[str, ...]:
    """Extract key terms with lightweight POS-based filtering."""
    normalized = normalize_text(text)
    tokens = _safe_tokenize(normalized)
    if not tokens:
        return ()

    lemmatizer = WordNetLemmatizer()
    tagged = _safe_pos_tag(tokens)
    keep_prefixes = ("NN", "VB", "JJ", "RB")

    selected: list[str] = []
    for token, tag in tagged:
        if not tag.startswith(keep_prefixes):
            continue
        try:
            lemma = lemmatizer.lemmatize(token)
        except LookupError:
            lemma = token
        selected.append(lemma.lower())

    if not selected:
        selected = tokens

    counts = Counter(selected)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(term for term, _ in ordered[:max_terms])


def _preferred_pos(key_terms: tuple[str, ...]) -> str:
    for term in key_terms:
        synsets = wordnet_synsets(term)
        if synsets:
            return synsets[0].pos()
    return "n"


def expand_text(
    source_text: str,
    max_key_terms: int = 12,
    max_synsets_per_term: int = 4,
) -> SemanticCloud:
    """Expand source text into a structured semantic cloud."""
    ensure_wordnet_available(download=False)

    normalized_source = normalize_text(source_text)
    if not normalized_source:
        raise ValueError("source_text must be non-empty")

    key_terms = list(extract_key_terms(normalized_source, max_terms=max_key_terms))
    if normalized_source not in key_terms:
        key_terms.insert(0, normalized_source)
    key_terms = key_terms[:max_key_terms]

    definitions: list[str] = []
    synonyms: set[str] = set()
    broader: set[str] = set()
    related: set[str] = set()
    constraints: set[str] = {"preserve original semantic intent"}
    negatives: set[str] = set()

    for term in key_terms:
        for synset in wordnet_synsets(term)[:max_synsets_per_term]:
            definitions.append(synset.definition())
            constraints.add(f"respect domain: {synset.lexname()}")
            for lemma in synset.lemma_names():
                synonyms.add(lemma.replace("_", " ").lower())
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemma_names():
                    broader.add(lemma.replace("_", " ").lower())
            for hyponym in synset.hyponyms()[:8]:
                for lemma in hyponym.lemma_names():
                    related.add(lemma.replace("_", " ").lower())
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    negatives.add(antonym.name().replace("_", " ").lower())

    if not definitions:
        definitions.append(normalized_source)

    composed_cloud_text = " ".join(
        [
            f"source: {normalized_source}",
            "key terms: " + ", ".join(key_terms),
            "definitions: " + "; ".join(definitions),
            "synonyms: " + ", ".join(sorted(synonyms)),
            "broader concepts: " + ", ".join(sorted(broader)),
            "related concepts: " + ", ".join(sorted(related)),
            "constraints: " + ", ".join(sorted(constraints)),
            "negative constraints: " + ", ".join(sorted(negatives)),
        ]
    ).strip()

    preferred_pos = _preferred_pos(tuple(key_terms))

    return SemanticCloud(
        source_text=normalized_source,
        key_terms=tuple(key_terms),
        definitions=tuple(definitions),
        synonyms=tuple(sorted(synonyms)),
        broader_concepts=tuple(sorted(broader)),
        related_concepts=tuple(sorted(related)),
        constraints=tuple(sorted(constraints)),
        negative_constraints=tuple(sorted(negatives)),
        composed_cloud_text=composed_cloud_text,
        preferred_pos=preferred_pos,
    )


def safe_expand_text(source_text: str) -> SemanticCloud:
    """Expand text while converting resource lookup errors into domain errors."""
    try:
        return expand_text(source_text=source_text)
    except LookupError as error:
        raise LexiconResourceError(
            "Required NLTK data is missing. Run: "
            "python -m nltk.downloader wordnet omw-1.4 punkt averaged_perceptron_tagger"
        ) from error
