"""
MindCheck user config — stored at ~/.mindcheck/config.json

Manages Tier 3 provider settings (API key, model, provider).
Supports: Anthropic, OpenAI, Ollama (local/free).
"""

import json
from pathlib import Path
from typing import Optional


_CONFIG_PATH = Path.home() / ".mindcheck" / "config.json"

_DEFAULTS = {
    "tier3_provider":    None,   # "anthropic" | "openai" | "ollama" | None
    "tier3_key":         None,   # API key (None for Ollama)
    "tier3_model":       None,   # None = use provider default
    "tier3_ollama_url":  "http://localhost:11434",
    "tier3_ollama_model": "llama3.2",
}

# Default models per provider (cheapest/fastest option first)
_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai":    "gpt-4o-mini",
    "ollama":    "llama3.2",
}

# All known models per provider, for display purposes
AVAILABLE_MODELS = {
    "anthropic": [
        ("claude-haiku-4-5",   "cheapest, fastest  [default]"),
        ("claude-sonnet-4-5",  "balanced"),
        ("claude-opus-4-5",    "most accurate, expensive"),
    ],
    "openai": [
        ("gpt-4o-mini",        "cheapest, fastest  [default]"),
        ("gpt-4o",             "balanced"),
        ("o1-mini",            "reasoning model"),
    ],
    "ollama": [
        ("llama3.2",           "good balance  [default]"),
        ("mistral",            "lightweight"),
        ("gemma3",             "Google, fast"),
    ],
}


def load_config() -> dict:
    """Load config from disk, merging with defaults."""
    cfg = dict(_DEFAULTS)
    if _CONFIG_PATH.exists():
        try:
            stored = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update(stored)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    """Persist config to disk."""
    _CONFIG_PATH.parent.mkdir(exist_ok=True)
    _CONFIG_PATH.write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )


def set_key(key: str) -> dict:
    """
    Set API key and auto-detect provider from key prefix.
      sk-ant-*  → Anthropic
      sk-*      → OpenAI
      anything else → ask user to specify provider
    """
    cfg = load_config()
    cfg["tier3_key"] = key

    if key.startswith("sk-ant-"):
        cfg["tier3_provider"] = "anthropic"
    elif key.startswith("sk-"):
        cfg["tier3_provider"] = "openai"
    # else: provider must be set separately

    save_config(cfg)
    return cfg


def set_provider(provider: str, url: Optional[str] = None,
                 model: Optional[str] = None) -> dict:
    """Set provider explicitly (useful for Ollama which needs no API key)."""
    valid = ("anthropic", "openai", "ollama")
    if provider not in valid:
        raise ValueError(f"Provider must be one of: {valid}")

    cfg = load_config()
    cfg["tier3_provider"] = provider
    if url:
        cfg["tier3_ollama_url"] = url
    if model:
        cfg["tier3_model"] = model
    save_config(cfg)
    return cfg


def get_model(cfg: dict) -> str:
    """Return the model to use, falling back to provider default."""
    return cfg.get("tier3_model") or _DEFAULT_MODELS.get(
        cfg.get("tier3_provider", ""), ""
    )


def is_tier3_configured(cfg: Optional[dict] = None) -> bool:
    """Return True if Tier 3 can run (provider + key both set, or Ollama)."""
    if cfg is None:
        cfg = load_config()
    provider = cfg.get("tier3_provider")
    if not provider:
        return False
    if provider == "ollama":
        return True
    return bool(cfg.get("tier3_key"))


def masked_key(cfg: dict) -> str:
    """Return a masked version of the key for display."""
    key = cfg.get("tier3_key") or ""
    if not key:
        return "(not set)"
    return key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
