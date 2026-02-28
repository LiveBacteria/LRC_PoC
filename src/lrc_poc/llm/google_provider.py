"""Google Gemini provider adapter."""

from __future__ import annotations

from .base import BaseLLMProvider, LLMProviderConfig, LLMProviderError, normalize_str_list


class GoogleProvider(BaseLLMProvider):
    """Google Generative AI provider."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash") -> None:
        super().__init__(
            LLMProviderConfig(
                provider_name="google",
                model_name=model_name,
                api_key=api_key,
            )
        )

    def _invoke(self, prompt: str) -> str:
        if not self.is_configured():
            raise LLMProviderError("Google provider not configured.")
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.config.api_key)
            model = genai.GenerativeModel(self.config.model_name)
            response = model.generate_content(prompt)
            text = getattr(response, "text", None)
            if not text:
                raise LLMProviderError("Google provider returned empty response.")
            return str(text)
        except Exception as error:  # pragma: no cover - network/provider dependent
            raise LLMProviderError(f"Google provider invocation failed: {error}") from error

    def expand_cloud(self, source_text: str) -> dict[str, list[str] | str]:
        prompt = (
            "Expand the concept into JSON with keys: definitions, synonyms, broader_concepts, "
            "related_concepts, constraints, negative_constraints. "
            "Return only JSON. Source text: "
            f"{source_text}"
        )
        payload = self.extract_json_block(self._invoke(prompt))
        return {
            "definitions": normalize_str_list(payload.get("definitions")),
            "synonyms": normalize_str_list(payload.get("synonyms")),
            "broader_concepts": normalize_str_list(payload.get("broader_concepts")),
            "related_concepts": normalize_str_list(payload.get("related_concepts")),
            "constraints": normalize_str_list(payload.get("constraints")),
            "negative_constraints": normalize_str_list(payload.get("negative_constraints")),
        }

    def propose_candidates(self, cloud_text: str, top_n: int = 20) -> list[str]:
        prompt = (
            f"Given this semantic cloud, return JSON: {{\"candidates\": [..]}} "
            f"with {top_n} single-word candidates. Cloud: {cloud_text}"
        )
        payload = self.extract_json_block(self._invoke(prompt))
        return normalize_str_list(payload.get("candidates"))[:top_n]

    def rerank_candidates(self, cloud_text: str, candidates: list[str]) -> list[str]:
        prompt = (
            "Rerank these candidates for the semantic cloud and return JSON "
            "{\"ranked\": [..]} with best first. "
            f"Cloud: {cloud_text}. Candidates: {candidates}"
        )
        payload = self.extract_json_block(self._invoke(prompt))
        ranked = normalize_str_list(payload.get("ranked"))
        return ranked or [candidate.lower() for candidate in candidates]
