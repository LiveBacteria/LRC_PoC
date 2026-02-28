"""OpenAI provider adapter."""

from __future__ import annotations

from .base import BaseLLMProvider, LLMProviderConfig, LLMProviderError, normalize_str_list


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Chat Completions provider."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini") -> None:
        super().__init__(
            LLMProviderConfig(
                provider_name="openai",
                model_name=model_name,
                api_key=api_key,
            )
        )

    def _invoke(self, prompt: str) -> str:
        if not self.is_configured():
            raise LLMProviderError("OpenAI provider not configured.")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.config.api_key)
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Return strict JSON only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
            )
            text = response.choices[0].message.content if response.choices else ""
            if not text:
                raise LLMProviderError("OpenAI provider returned empty response.")
            return text
        except Exception as error:  # pragma: no cover - network/provider dependent
            raise LLMProviderError(f"OpenAI provider invocation failed: {error}") from error

    def expand_cloud(self, source_text: str) -> dict[str, list[str] | str]:
        payload = self.extract_json_block(
            self._invoke(
                "Expand into JSON with keys definitions, synonyms, broader_concepts, "
                f"related_concepts, constraints, negative_constraints. Source: {source_text}"
            )
        )
        return {
            "definitions": normalize_str_list(payload.get("definitions")),
            "synonyms": normalize_str_list(payload.get("synonyms")),
            "broader_concepts": normalize_str_list(payload.get("broader_concepts")),
            "related_concepts": normalize_str_list(payload.get("related_concepts")),
            "constraints": normalize_str_list(payload.get("constraints")),
            "negative_constraints": normalize_str_list(payload.get("negative_constraints")),
        }

    def propose_candidates(self, cloud_text: str, top_n: int = 20) -> list[str]:
        payload = self.extract_json_block(
            self._invoke(
                f"Return JSON {{\"candidates\": [...]}} with {top_n} single-word candidates "
                f"for this cloud: {cloud_text}"
            )
        )
        return normalize_str_list(payload.get("candidates"))[:top_n]

    def rerank_candidates(self, cloud_text: str, candidates: list[str]) -> list[str]:
        payload = self.extract_json_block(
            self._invoke(
                "Rerank candidates for a semantic cloud. Return JSON {\"ranked\": [...]}. "
                f"Cloud: {cloud_text}. Candidates: {candidates}"
            )
        )
        ranked = normalize_str_list(payload.get("ranked"))
        return ranked or [candidate.lower() for candidate in candidates]
