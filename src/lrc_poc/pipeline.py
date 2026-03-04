"""Deterministic LRC pipeline orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .candidates import generate_candidates
from .config import ApiKeys, AppConfig, ConfigError, ModelDefaults, load_config
from .expand import safe_expand_text
from .llm import BaseLLMProvider, LLMProviderError
from .models import CandidateResult, LexiconEntry, ReductionResult, SemanticCloud
from .reduce import reduce_cloud
from .score import SemanticScorer


def _empty_config() -> AppConfig:
    return AppConfig(
        defaults=ModelDefaults(),
        api_keys=ApiKeys(),
        source_path=Path(),
    )


def _merge_cloud(base: SemanticCloud, payload: dict[str, list[str] | str]) -> SemanticCloud:
    def _merged(original: tuple[str, ...], key: str) -> tuple[str, ...]:
        extra = payload.get(key, [])
        if isinstance(extra, str):
            extra_values = [extra]
        else:
            extra_values = [str(item) for item in extra]
        merged = {item.strip().lower() for item in original}
        for item in extra_values:
            cleaned = item.strip().lower()
            if cleaned:
                merged.add(cleaned)
        return tuple(sorted(merged))

    definitions = _merged(base.definitions, "definitions")
    synonyms = _merged(base.synonyms, "synonyms")
    broader_concepts = _merged(base.broader_concepts, "broader_concepts")
    related_concepts = _merged(base.related_concepts, "related_concepts")
    constraints = _merged(base.constraints, "constraints")
    negative_constraints = _merged(base.negative_constraints, "negative_constraints")
    composed_cloud_text = " ".join(
        [
            f"source: {base.source_text}",
            "key terms: " + ", ".join(base.key_terms),
            "definitions: " + "; ".join(definitions),
            "synonyms: " + ", ".join(synonyms),
            "broader concepts: " + ", ".join(broader_concepts),
            "related concepts: " + ", ".join(related_concepts),
            "constraints: " + ", ".join(constraints),
            "negative constraints: " + ", ".join(negative_constraints),
        ]
    )

    return SemanticCloud(
        source_text=base.source_text,
        key_terms=base.key_terms,
        definitions=definitions,
        synonyms=synonyms,
        broader_concepts=broader_concepts,
        related_concepts=related_concepts,
        constraints=constraints,
        negative_constraints=negative_constraints,
        composed_cloud_text=composed_cloud_text,
        preferred_pos=base.preferred_pos,
    )


def _dedupe_entries(entries: Iterable[LexiconEntry], max_items: int) -> tuple[LexiconEntry, ...]:
    deduped: list[LexiconEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        key = (entry.word, entry.part_of_speech, entry.definition)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
        if len(deduped) >= max_items:
            break
    return tuple(deduped)


class LRCPipeline:
    """Deterministic pipeline (mode 0 only)."""

    def __init__(
        self,
        lexicon: tuple[LexiconEntry, ...] | list[LexiconEntry],
        scorer: SemanticScorer | None = None,
        config: AppConfig | None = None,
        provider: BaseLLMProvider | None = None,
        preferred_provider: str | None = None,
        use_domain_heuristics: bool = False,
    ) -> None:
        if not lexicon:
            raise ValueError("lexicon is empty")
        self.lexicon = tuple(lexicon)
        self.scorer = scorer or SemanticScorer()
        self.use_domain_heuristics = use_domain_heuristics

        if config is None:
            try:
                config = load_config()
            except ConfigError:
                config = _empty_config()
        self.config = config

        # Deterministic runtime: providers are intentionally disabled.
        del provider, preferred_provider
        self.provider = None
        self.lexicon_by_word: dict[str, list[LexiconEntry]] = {}
        for entry in self.lexicon:
            self.lexicon_by_word.setdefault(entry.word.lower(), []).append(entry)

    @staticmethod
    def _validate_mode(mode: int) -> None:
        if mode != 0:
            raise ValueError("Only deterministic mode 0 is supported")

    def _provider_or_reason(self, fallback_reasons: list[str]) -> BaseLLMProvider | None:
        if self.provider and self.provider.is_configured():
            return self.provider
        fallback_reasons.append("LLM provider unavailable; deterministic fallback used")
        return None

    def _candidate_pool_from_llm(
        self,
        cloud: SemanticCloud,
        max_candidates: int,
        provider: BaseLLMProvider,
        fallback_reasons: list[str],
    ) -> tuple[LexiconEntry, ...] | None:
        try:
            proposed_words = provider.propose_candidates(cloud.composed_cloud_text, top_n=20)
        except LLMProviderError as error:
            fallback_reasons.append(f"LLM candidate proposal failed: {error}")
            return None

        proposed_entries: list[LexiconEntry] = []
        for word in proposed_words:
            normalized = word.strip().lower()
            if not normalized:
                continue
            matched = self.lexicon_by_word.get(normalized, [])
            if matched:
                proposed_entries.extend(matched)
                continue

            # Preserve short phrase compressions proposed by the LLM even when
            # they are out of lexicon, so sentence-level reduction can remain coherent.
            proposed_entries.append(
                LexiconEntry(
                    word=normalized,
                    lemma=normalized,
                    part_of_speech=cloud.preferred_pos,
                    definition=f"LLM proposed compression candidate for: {cloud.source_text}",
                    synonyms=(normalized,),
                    hypernyms=tuple(cloud.broader_concepts[:5]),
                    hyponyms=(),
                    synset_id=f"llm:{normalized.replace(' ', '_')}",
                    sense_count=1,
                )
            )

        deterministic_entries = generate_candidates(
            cloud,
            self.lexicon,
            max_candidates=max_candidates,
            use_domain_heuristics=self.use_domain_heuristics,
        )
        merged = _dedupe_entries(
            list(proposed_entries) + list(deterministic_entries),
            max_items=max_candidates,
        )
        return merged

    def _apply_llm_rerank(
        self,
        cloud: SemanticCloud,
        reduction: ReductionResult,
        provider: BaseLLMProvider,
        fallback_reasons: list[str],
    ) -> ReductionResult:
        candidates = [item.word for item in reduction.top_k]
        try:
            reranked_words = provider.rerank_candidates(cloud.composed_cloud_text, candidates)
        except LLMProviderError as error:
            fallback_reasons.append(f"LLM rerank failed: {error}")
            return reduction

        ranked_lookup: dict[str, CandidateResult] = {
            item.word.lower(): item for item in reduction.top_k
        }
        ordered: list[CandidateResult] = []
        seen: set[str] = set()
        for word in reranked_words:
            item = ranked_lookup.get(word.lower())
            if not item or item.word.lower() in seen:
                continue
            seen.add(item.word.lower())
            ordered.append(item)

        for item in reduction.top_k:
            lowered = item.word.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(item)

        if not ordered:
            fallback_reasons.append("LLM rerank returned no valid candidates")
            return reduction

        second_score = ordered[1].score_breakdown.total if len(ordered) > 1 else None
        confidence = self.scorer.confidence_from_margin(ordered[0].score_breakdown.total, second_score)
        return ReductionResult(
            winner=ordered[0],
            top_k=tuple(ordered),
            confidence=confidence,
            candidate_count=reduction.candidate_count,
            mode_used=reduction.mode_used,
            fallback_reason=reduction.fallback_reason,
        )

    def run_once(
        self,
        text: str,
        mode: int = 0,
        top_k: int = 10,
        max_candidates: int = 300,
    ) -> ReductionResult:
        """Run one deterministic pass of expansion and reduction."""
        self._validate_mode(mode)
        cloud = safe_expand_text(text)

        reduction = reduce_cloud(
            cloud=cloud,
            lexicon=self.lexicon,
            scorer=self.scorer,
            top_k=top_k,
            max_candidates=max_candidates,
            use_domain_heuristics=self.use_domain_heuristics,
            mode_used=0,
            fallback_reason="",
            candidate_pool=None,
        )
        if reduction.mode_used != 0:
            reduction = replace(reduction, mode_used=0)
        if reduction.fallback_reason:
            reduction = replace(reduction, fallback_reason="")
        return reduction
