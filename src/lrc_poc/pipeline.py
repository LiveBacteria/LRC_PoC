"""Mode-selectable LRC pipeline orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .candidates import generate_candidates
from .config import ApiKeys, AppConfig, ConfigError, ModelDefaults, load_config
from .expand import safe_expand_text
from .llm import AnthropicProvider, BaseLLMProvider, GoogleProvider, LLMProviderError, OpenAIProvider
from .models import CandidateResult, LexiconEntry, ReductionResult, SemanticCloud
from .reduce import reduce_cloud
from .score import SemanticScorer


def _empty_config() -> AppConfig:
    return AppConfig(
        defaults=ModelDefaults(),
        api_keys=ApiKeys(),
        source_path=Path(),
    )


def _choose_provider(config: AppConfig, preferred_provider: str | None = None) -> BaseLLMProvider | None:
    model_name = config.defaults.model_name
    providers: list[BaseLLMProvider] = [
        GoogleProvider(api_key=config.api_keys.google_api_key, model_name=model_name or ""),
        OpenAIProvider(api_key=config.api_keys.openai_api_key, model_name=model_name or "gpt-4o-mini"),
        AnthropicProvider(
            api_key=config.api_keys.anthropic_api_key,
            model_name=model_name or "claude-3-5-haiku-latest",
        ),
    ]

    if preferred_provider:
        for provider in providers:
            if provider.name == preferred_provider and provider.is_configured():
                return provider
        return None

    for provider in providers:
        if provider.is_configured():
            return provider
    return None


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
    """Mode-selectable pipeline with graceful deterministic fallback."""

    def __init__(
        self,
        lexicon: tuple[LexiconEntry, ...] | list[LexiconEntry],
        scorer: SemanticScorer | None = None,
        config: AppConfig | None = None,
        provider: BaseLLMProvider | None = None,
        preferred_provider: str | None = None,
    ) -> None:
        if not lexicon:
            raise ValueError("lexicon is empty")
        self.lexicon = tuple(lexicon)
        self.scorer = scorer or SemanticScorer()

        if config is None:
            try:
                config = load_config()
            except ConfigError:
                config = _empty_config()
        self.config = config

        self.provider = provider or _choose_provider(config, preferred_provider=preferred_provider)
        self.lexicon_by_word: dict[str, list[LexiconEntry]] = {}
        for entry in self.lexicon:
            self.lexicon_by_word.setdefault(entry.word.lower(), []).append(entry)

    @staticmethod
    def _validate_mode(mode: int) -> None:
        if mode not in {0, 1, 2, 3, 4}:
            raise ValueError("mode must be one of 0, 1, 2, 3, 4")

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

        deterministic_entries = generate_candidates(cloud, self.lexicon, max_candidates=max_candidates)
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
        """Run one pass of expansion and reduction under selected mode."""
        self._validate_mode(mode)
        fallback_reasons: list[str] = []

        cloud = safe_expand_text(text)

        provider: BaseLLMProvider | None = None
        if mode in {1, 2, 3, 4}:
            provider = self._provider_or_reason(fallback_reasons)

        if mode in {1, 4} and provider:
            try:
                cloud = _merge_cloud(cloud, provider.expand_cloud(text))
            except LLMProviderError as error:
                fallback_reasons.append(f"LLM expansion failed: {error}")

        candidate_pool = None
        if mode in {2, 4} and provider:
            candidate_pool = self._candidate_pool_from_llm(
                cloud=cloud,
                max_candidates=max_candidates,
                provider=provider,
                fallback_reasons=fallback_reasons,
            )

        reduction = reduce_cloud(
            cloud=cloud,
            lexicon=self.lexicon,
            scorer=self.scorer,
            top_k=top_k,
            max_candidates=max_candidates,
            mode_used=mode,
            fallback_reason="; ".join(fallback_reasons),
            candidate_pool=candidate_pool,
        )

        if mode in {3, 4} and provider:
            reduction = self._apply_llm_rerank(
                cloud=cloud,
                reduction=reduction,
                provider=provider,
                fallback_reasons=fallback_reasons,
            )

        fallback_reason = "; ".join(reason for reason in fallback_reasons if reason)
        if reduction.fallback_reason != fallback_reason:
            reduction = replace(reduction, fallback_reason=fallback_reason)
        return reduction
