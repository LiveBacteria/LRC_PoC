"""Scoring logic for cloud-to-lexicon reduction."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import LexiconEntry, ScoreBreakdown, SemanticCloud

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def _safe_cosine_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([text_a, text_b])
    similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    return max(0.0, min(1.0, similarity))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


@dataclass(frozen=True)
class ScoringWeights:
    """Tunable weights for reduction scoring."""

    definition_similarity: float = 0.55
    synonym_relation_score: float = 0.15
    hypernym_alignment: float = 0.10
    keyword_overlap: float = 0.10
    pos_match: float = 0.05
    concision_bonus: float = 0.05
    ambiguity_penalty: float = 0.10


class SemanticScorer:
    """Score lexical candidates against a semantic cloud."""

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score_candidate(self, cloud: SemanticCloud, candidate: LexiconEntry) -> ScoreBreakdown:
        definition_similarity = _safe_cosine_similarity(
            cloud.composed_cloud_text, candidate.definition
        )
        synonym_relation_score = _jaccard_similarity(
            set(cloud.synonyms), set(candidate.synonyms)
        )
        hypernym_alignment = _jaccard_similarity(
            set(cloud.broader_concepts), set(candidate.hypernyms)
        )
        keyword_overlap = _jaccard_similarity(
            set(cloud.key_terms), _tokenize(candidate.definition)
        )
        pos_match = 1.0 if candidate.part_of_speech == cloud.preferred_pos else 0.0
        concision_bonus = 1.0 / max(len(candidate.word.split()), 1)
        ambiguity_penalty = min(1.0, max(candidate.sense_count - 1, 0) / 10.0)

        weighted = (
            self.weights.definition_similarity * definition_similarity
            + self.weights.synonym_relation_score * synonym_relation_score
            + self.weights.hypernym_alignment * hypernym_alignment
            + self.weights.keyword_overlap * keyword_overlap
            + self.weights.pos_match * pos_match
            + self.weights.concision_bonus * concision_bonus
            - self.weights.ambiguity_penalty * ambiguity_penalty
        )
        total = max(0.0, min(1.0, weighted))

        return ScoreBreakdown(
            definition_similarity=definition_similarity,
            synonym_relation_score=synonym_relation_score,
            hypernym_alignment=hypernym_alignment,
            keyword_overlap=keyword_overlap,
            pos_match=pos_match,
            concision_bonus=concision_bonus,
            ambiguity_penalty=ambiguity_penalty,
            total=total,
        )

    @staticmethod
    def confidence_from_margin(best: float, second_best: float | None) -> float:
        """Convert ranking margin into bounded confidence."""
        if second_best is None:
            return 0.55
        margin = max(0.0, best - second_best)
        logistic = 1.0 / (1.0 + math.exp(-10.0 * (margin - 0.05)))
        return max(0.0, min(1.0, logistic))

    @staticmethod
    def explanation(cloud: SemanticCloud, candidate: LexiconEntry, breakdown: ScoreBreakdown) -> str:
        """Create a concise explanation for candidate ranking."""
        relation_hits = len(set(candidate.synonyms) & set(cloud.synonyms))
        return (
            f"Matched definition similarity={breakdown.definition_similarity:.3f}, "
            f"keyword_overlap={breakdown.keyword_overlap:.3f}, "
            f"synonym_hits={relation_hits}, "
            f"ambiguity_penalty={breakdown.ambiguity_penalty:.3f}."
        )
