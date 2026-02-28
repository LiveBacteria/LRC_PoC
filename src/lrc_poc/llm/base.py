"""LLM provider base interfaces for mode-selectable pipeline behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re


class LLMProviderError(RuntimeError):
    """Raised when LLM provider operations fail."""


@dataclass(frozen=True)
class LLMProviderConfig:
    provider_name: str
    model_name: str
    api_key: str


class BaseLLMProvider(ABC):
    """Abstract provider interface used by pipeline mode handlers."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.provider_name

    def is_configured(self) -> bool:
        return bool(self.config.api_key and self.config.model_name)

    @abstractmethod
    def expand_cloud(self, source_text: str) -> dict[str, list[str] | str]:
        """Return structured expansion fields to merge into deterministic cloud."""

    @abstractmethod
    def propose_candidates(self, cloud_text: str, top_n: int = 20) -> list[str]:
        """Return candidate words/phrases for deterministic reranking."""

    @abstractmethod
    def rerank_candidates(self, cloud_text: str, candidates: list[str]) -> list[str]:
        """Return candidate words ordered from best to worst."""

    @staticmethod
    def extract_json_block(text: str) -> dict[str, object]:
        """Extract first JSON object from model output."""
        text = text.strip()
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMProviderError("Model response did not contain a JSON object.")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise LLMProviderError("Failed to parse JSON from model response.") from error

        if not isinstance(payload, dict):
            raise LLMProviderError("Expected JSON object payload from model response.")
        return payload


def normalize_str_list(value: object) -> list[str]:
    """Normalize provider JSON values to clean string lists."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []
    if isinstance(value, list):
        cleaned: list[str] = []
        for item in value:
            item_str = str(item).strip().lower()
            if item_str:
                cleaned.append(item_str)
        return cleaned
    return [str(value).strip().lower()]
