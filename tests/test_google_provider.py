"""Tests for Google provider model resolution behavior."""

from __future__ import annotations

from lrc_poc.llm.google_provider import GoogleProvider


def test_google_provider_prefers_modern_model(monkeypatch) -> None:
    provider = GoogleProvider(api_key="x", model_name="")
    monkeypatch.setattr(
        provider,
        "_list_compatible_models",
        lambda: ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-pro-exp"],
    )
    chosen = provider._resolve_model_name(force_refresh=True)
    assert chosen.startswith("gemini-2.5-pro")


def test_google_provider_honors_configured_model() -> None:
    provider = GoogleProvider(api_key="x", model_name="gemini-2.0-flash")
    chosen = provider._resolve_model_name()
    assert chosen == "gemini-2.0-flash"


def test_google_provider_can_override_stale_config(monkeypatch) -> None:
    provider = GoogleProvider(api_key="x", model_name="gemini-1.5-flash")
    monkeypatch.setattr(provider, "_list_compatible_models", lambda: ["gemini-2.0-flash"])
    chosen = provider._resolve_model_name(force_refresh=True, ignore_config=True)
    assert chosen == "gemini-2.0-flash"
