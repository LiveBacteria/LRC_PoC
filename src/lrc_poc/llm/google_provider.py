"""Google Gemini provider adapter."""

from __future__ import annotations

from typing import Iterable
import warnings

from .base import BaseLLMProvider, LLMProviderConfig, LLMProviderError, normalize_str_list


class GoogleProvider(BaseLLMProvider):
    """Google Generative AI provider."""

    _PREFERRED_PREFIXES = (
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    )

    def __init__(self, api_key: str, model_name: str = "") -> None:
        super().__init__(
            LLMProviderConfig(
                provider_name="google",
                model_name=model_name,
                api_key=api_key,
            )
        )
        self._resolved_model_name: str | None = None

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        name = model_name.strip()
        if name.startswith("models/"):
            return name.split("/", 1)[1]
        return name

    def is_configured(self) -> bool:
        # Model can be discovered dynamically, so only the API key is required.
        return bool(self.config.api_key)

    def _pick_preferred_model(self, candidates: Iterable[str]) -> str | None:
        candidate_list = sorted({self._normalize_model_name(item) for item in candidates if item})
        if not candidate_list:
            return None
        for prefix in self._PREFERRED_PREFIXES:
            for candidate in candidate_list:
                if candidate.startswith(prefix):
                    return candidate
        return candidate_list[0]

    def _list_compatible_models(self) -> list[str]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import google.generativeai as genai

            models: list[str] = []
            for model in genai.list_models():
                methods = list(getattr(model, "supported_generation_methods", []) or [])
                if "generateContent" not in methods:
                    continue
                model_name = self._normalize_model_name(getattr(model, "name", ""))
                if model_name:
                    models.append(model_name)
            return models
        except Exception as error:  # pragma: no cover - network/provider dependent
            raise LLMProviderError(f"Google model discovery failed: {error}") from error

    def _resolve_model_name(self, force_refresh: bool = False, ignore_config: bool = False) -> str:
        if self._resolved_model_name and not force_refresh:
            return self._resolved_model_name

        configured_name = self._normalize_model_name(self.config.model_name)
        if configured_name and not ignore_config:
            self._resolved_model_name = configured_name
            return self._resolved_model_name

        available = self._list_compatible_models()
        selected = self._pick_preferred_model(available)
        if not selected:
            raise LLMProviderError(
                "No compatible Google models found with generateContent support."
            )
        self._resolved_model_name = selected
        return self._resolved_model_name

    def _invoke(self, prompt: str) -> str:
        if not self.is_configured():
            raise LLMProviderError("Google provider not configured.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import google.generativeai as genai

            genai.configure(api_key=self.config.api_key)
            model_name = self._resolve_model_name()
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                text = getattr(response, "text", None)
                if text:
                    return str(text)
            except Exception as first_error:
                message = str(first_error).lower()
                stale_model = "not found" in message or "not supported" in message or "404" in message
                if stale_model:
                    # Retry with model discovery in case configured/default model is stale.
                    model_name = self._resolve_model_name(force_refresh=True, ignore_config=True)
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    text = getattr(response, "text", None)
                    if text:
                        return str(text)
                raise

            raise LLMProviderError(
                "Google provider returned empty response."
            )
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
            f"with up to {top_n} concise lexical compressions (single words or short phrases). "
            f"Cloud: {cloud_text}"
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
