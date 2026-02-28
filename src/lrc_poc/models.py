"""Core data models for lexical reduction and recursion analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LexiconEntry:
    """A normalized lexical entry sourced from a lexicon provider."""

    word: str
    lemma: str
    part_of_speech: str
    definition: str
    examples: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    hypernyms: tuple[str, ...] = ()
    hyponyms: tuple[str, ...] = ()
    synset_id: str = ""
    sense_count: int = 0


@dataclass(frozen=True)
class SemanticCloud:
    """Structured expansion object used for candidate reduction."""

    source_text: str
    key_terms: tuple[str, ...]
    definitions: tuple[str, ...]
    synonyms: tuple[str, ...]
    broader_concepts: tuple[str, ...]
    related_concepts: tuple[str, ...]
    constraints: tuple[str, ...]
    negative_constraints: tuple[str, ...]
    composed_cloud_text: str
    preferred_pos: str = "n"


@dataclass(frozen=True)
class ScoreBreakdown:
    """Detailed scoring components for candidate ranking."""

    definition_similarity: float
    synonym_relation_score: float
    hypernym_alignment: float
    keyword_overlap: float
    pos_match: float
    concision_bonus: float
    ambiguity_penalty: float
    total: float


@dataclass(frozen=True)
class CandidateResult:
    """Single ranked candidate with metadata and explainability fields."""

    word: str
    definition: str
    part_of_speech: str
    synset_id: str
    score_breakdown: ScoreBreakdown
    explanation: str


@dataclass(frozen=True)
class ReductionResult:
    """Result of reducing a semantic cloud onto a lexical candidate set."""

    winner: CandidateResult
    top_k: tuple[CandidateResult, ...]
    confidence: float
    candidate_count: int
    mode_used: int
    fallback_reason: str = ""


@dataclass(frozen=True)
class IterationResult:
    """Single recursion iteration output."""

    iteration: int
    input_text: str
    winner_word: str
    winner_score: float
    confidence: float
    drift_from_origin: float
    entropy: float
    fixed_point: bool
    cycle_detected: bool
    cycle_length: int | None
    top_k_words: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttractorRunResult:
    """Aggregate result for batched recursive mapping runs."""

    seed_count: int
    iterations: int
    fixed_points: int
    cycles: int
    average_drift: float
    average_entropy: float
    basin_summary: dict[str, int] = field(default_factory=dict)
    output_files: tuple[str, ...] = ()
