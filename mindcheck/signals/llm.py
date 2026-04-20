"""
Tier 3: LLM-based classification for ambiguous messages.

Only called when Tier 2 embedding confidence is low (hypothesis_confidence < 0.04).
Only individual user messages are sent — never full sessions or AI responses.
Results feed back into prototype learning via the cache.

Supported providers: Anthropic, OpenAI, Ollama (local/free).
Configure with: mindcheck config --key <api_key>
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from mindcheck.parser import Session
from mindcheck.signals.semantic import SemanticSignals


@dataclass
class LLMSignals:
    ran: bool = False
    messages_reclassified: int = 0      # how many low-confidence messages were resolved
    hypothesis_level_avg: float = 0.0   # corrected avg after LLM overrides
    prototype_candidates: list[dict] = field(default_factory=list)
    # ^ Collected when LLM confidence ≥ 0.85. Reserved for v1.1 prototype
    #   self-improvement loop — not yet wired into semantic.py.


_CLASSIFICATION_PROMPT = """\
You are evaluating a single message a person sent to an AI assistant.
Classify only what is explicitly present — do not infer beyond the text.

Hypothesis levels:
  0 = pure delegation, no attempt  ("fix this", "just do it", "write the code for me")
  1 = symptom only                 ("there's an error", "it doesn't work", "something's wrong")
  2 = locates the problem          ("it fails on line 42", "breaks when X happens")
  3 = forms a hypothesis           ("I think it's X because Y", "my guess is Z")
  4 = tested a hypothesis          ("I tried X, still fails, so maybe Y instead")

Message:
{message}

Respond with ONLY valid JSON, no other text:
{{"hypothesis_level": <0-4>, "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


def extract_llm(
    session: Session,
    semantic: SemanticSignals,
    cfg: dict,
) -> LLMSignals:
    """
    Run Tier 3 on low-confidence Tier 2 messages.
    Updates hypothesis_level_avg if overrides improve it.
    """
    sig = LLMSignals()

    if not semantic.per_message:
        return sig

    provider = cfg.get("tier3_provider")
    if not provider:
        return sig

    key     = cfg.get("tier3_key")
    model   = cfg.get("tier3_model") or _default_model(provider)
    ollama_url = cfg.get("tier3_ollama_url", "http://localhost:11434")
    ollama_model = cfg.get("tier3_ollama_model", "llama3.2")

    user_msgs = session.user_messages
    updated_levels = []
    reclassified = 0

    for i, classification in enumerate(semantic.per_message):
        level = classification["hypothesis_level"]

        if not classification.get("low_confidence", False):
            updated_levels.append(level)
            continue

        # This message was uncertain — ask the LLM
        if i >= len(user_msgs):
            updated_levels.append(level)
            continue

        text = user_msgs[i].content[:800]   # cap at 800 chars to keep cost low
        if not text.strip():
            updated_levels.append(level)
            continue

        try:
            result = _call_provider(text, provider, key, model,
                                    ollama_url, ollama_model)
            llm_level      = int(result.get("hypothesis_level", level))
            llm_confidence = float(result.get("confidence", 0.0))
            reasoning      = result.get("reasoning", "")

            if 0 <= llm_level <= 4:
                updated_levels.append(llm_level)
                reclassified += 1

                # Save as prototype candidate if LLM is confident
                if llm_confidence >= 0.75:
                    sig.prototype_candidates.append({
                        "text":      text,
                        "level":     llm_level,
                        "reasoning": reasoning,
                        "provider":  provider,
                        "model":     model,
                    })
            else:
                updated_levels.append(level)

        except Exception:
            updated_levels.append(level)   # LLM failed — keep Tier 2 result

    if updated_levels:
        sig.ran = True
        sig.messages_reclassified = reclassified
        sig.hypothesis_level_avg = sum(updated_levels) / len(updated_levels)

    # Persist high-confidence classifications for prototype self-improvement
    if sig.prototype_candidates:
        _save_prototype_candidates(sig.prototype_candidates)

    return sig


def _save_prototype_candidates(candidates: list[dict]) -> None:
    """Append new high-confidence LLM classifications to the learned prototypes file.

    File: ~/.mindcheck/learned_prototypes.json
    Format: list of {text, level, reasoning, provider, model}
    Deduplicated by text so the same message is never added twice.
    """
    from pathlib import Path

    path = Path.home() / ".mindcheck" / "learned_prototypes.json"

    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing_texts = {e.get("text", "") for e in existing}
    new_entries = [c for c in candidates if c.get("text", "") not in existing_texts]

    if new_entries:
        existing.extend(new_entries)
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ── Provider dispatch ─────────────────────────────────────────────────────────

def _call_provider(text: str, provider: str, key: Optional[str],
                   model: str, ollama_url: str, ollama_model: str) -> dict:
    if provider == "anthropic":
        return _call_anthropic(text, key, model)
    elif provider == "openai":
        return _call_openai(text, key, model)
    elif provider == "gemini":
        return _call_gemini(text, key, model)
    elif provider == "ollama":
        return _call_ollama(text, ollama_url, ollama_model)
    raise ValueError(f"Unknown provider: {provider}")


def _call_anthropic(text: str, api_key: str, model: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=128,
        messages=[{"role": "user",
                   "content": _CLASSIFICATION_PROMPT.format(message=text)}],
    )
    return json.loads(response.content[0].text)


def _call_openai(text: str, api_key: str, model: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=128,
        response_format={"type": "json_object"},
        messages=[{"role": "user",
                   "content": _CLASSIFICATION_PROMPT.format(message=text)}],
    )
    return json.loads(response.choices[0].message.content)


def _call_gemini(text: str, api_key: str, model: str) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=_CLASSIFICATION_PROMPT.format(message=text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=128,
        ),
    )
    return json.loads(response.text)


def _call_ollama(text: str, url: str, model: str) -> dict:
    import urllib.request
    payload = json.dumps({
        "model":  model,
        "prompt": _CLASSIFICATION_PROMPT.format(message=text),
        "format": "json",
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        return json.loads(result["response"])


def _default_model(provider: str) -> str:
    return {
        "anthropic": "claude-haiku-4-5",
        "openai":    "gpt-4.1-nano",
        "gemini":    "gemini-2.5-flash",
        "ollama":    "llama3.2",
    }.get(provider, "")
