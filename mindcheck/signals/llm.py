"""
Tier 3: LLM-based classification for ambiguous messages.

Classifies ALL signals (hypothesis, ownership, critical engagement,
self-reliance, metacognition, delegation) — not just hypothesis level.
Only called when Tier 2 embedding confidence is low.
Only individual user messages are sent — never full sessions or AI responses.
Results feed back into prototype learning via learned_prototypes.json.

Supported providers: Anthropic, OpenAI, Gemini, Ollama (local/free).
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
    ownership_score: float = 0.0        # corrected ownership
    critical_engagement: float = 0.0    # corrected critical engagement
    self_reliance: float = 0.0          # corrected self-reliance
    metacognition_score: float = 0.0    # corrected metacognition
    delegation_penalty: float = 0.0     # corrected delegation
    t2_composite: float = 0.0           # T2-only score (before T3 corrections)
    prototype_candidates: list[dict] = field(default_factory=list)


_CLASSIFICATION_PROMPT = """\
You are evaluating a single message a person sent to an AI assistant.
Classify only what is explicitly present — do not infer beyond the text.

Hypothesis levels:
  0 = pure delegation, no attempt  ("fix this", "just do it", "write the code for me")
  1 = symptom only                 ("there's an error", "it doesn't work", "something's wrong")
  2 = locates the problem          ("it fails on line 42", "breaks when X happens")
  3 = forms a hypothesis           ("I think it's X because Y", "my guess is Z")
  4 = tested a hypothesis          ("I tried X, still fails, so maybe Y instead")

Signals (true/false):
  is_user_driven   = user is steering direction ("I want to try X", "no, let's do Y instead")
                     vs deferring ("what should I do?", "you decide")
  is_critical      = user pushes back or verifies ("that doesn't seem right", "I tested and got different results")
                     vs accepting passively ("looks good", "thanks")
  is_self_reliant  = user showed prior effort ("I tried X", "I read the docs", "I've been debugging")
                     vs no attempt ("I have no idea", "can you just do it")
  is_metacognitive = user reflects on own thinking ("am I approaching this wrong?", "help me understand why")
                     vs wants answer only ("just give me the code", "skip the explanation")
  is_delegation    = user outsources thinking entirely ("just do it all", "implement everything")
                     vs engaging ("explain so I can do it myself", "review my attempt")

Message:
{message}

Respond with ONLY valid JSON, no other text:
{{"hypothesis_level": <0-4>, "is_user_driven": <true/false>, "is_critical": <true/false>, "is_self_reliant": <true/false>, "is_metacognitive": <true/false>, "is_delegation": <true/false>, "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


def extract_llm(
    session: Session,
    semantic: SemanticSignals,
    cfg: dict,
) -> LLMSignals:
    """
    Run Tier 3 on low-confidence Tier 2 messages.
    Reclassifies ALL signals, not just hypothesis level.
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

    # Per-message tracking — start with Tier 2 values
    updated_levels = []
    updated_ownership = []
    updated_critical = []
    updated_self_reliant = []
    updated_metacognitive = []
    updated_delegation = []
    reclassified = 0

    for i, classification in enumerate(semantic.per_message):
        level = classification["hypothesis_level"]

        # Carry forward Tier 2 values as defaults
        t2_ownership = classification.get("is_user_driven", False)
        t2_critical = classification.get("is_critical", False)
        t2_self_reliant = classification.get("is_self_reliant", False)
        t2_metacognitive = classification.get("is_metacognitive", False)
        t2_delegation = classification.get("is_delegation", False)

        if not classification.get("low_confidence", False):
            # High confidence — keep Tier 2 result
            updated_levels.append(level)
            updated_ownership.append(t2_ownership)
            updated_critical.append(t2_critical)
            updated_self_reliant.append(t2_self_reliant)
            updated_metacognitive.append(t2_metacognitive)
            updated_delegation.append(t2_delegation)
            continue

        # This message was uncertain — ask the LLM
        if i >= len(user_msgs):
            updated_levels.append(level)
            updated_ownership.append(t2_ownership)
            updated_critical.append(t2_critical)
            updated_self_reliant.append(t2_self_reliant)
            updated_metacognitive.append(t2_metacognitive)
            updated_delegation.append(t2_delegation)
            continue

        text = user_msgs[i].content[:800]   # cap at 800 chars to keep cost low
        if not text.strip():
            updated_levels.append(level)
            updated_ownership.append(t2_ownership)
            updated_critical.append(t2_critical)
            updated_self_reliant.append(t2_self_reliant)
            updated_metacognitive.append(t2_metacognitive)
            updated_delegation.append(t2_delegation)
            continue

        try:
            result = _call_provider(text, provider, key, model,
                                    ollama_url, ollama_model)
            llm_level      = int(result.get("hypothesis_level", level))
            llm_confidence = float(result.get("confidence", 0.0))
            reasoning      = result.get("reasoning", "")

            if 0 <= llm_level <= 4:
                updated_levels.append(llm_level)
                updated_ownership.append(result.get("is_user_driven", t2_ownership))
                updated_critical.append(result.get("is_critical", t2_critical))
                updated_self_reliant.append(result.get("is_self_reliant", t2_self_reliant))
                updated_metacognitive.append(result.get("is_metacognitive", t2_metacognitive))
                updated_delegation.append(result.get("is_delegation", t2_delegation))
                reclassified += 1

                # Save as prototype candidate if LLM is confident
                if llm_confidence >= 0.75:
                    sig.prototype_candidates.append({
                        "text":             text,
                        "level":            llm_level,
                        "is_user_driven":   result.get("is_user_driven"),
                        "is_critical":      result.get("is_critical"),
                        "is_self_reliant":  result.get("is_self_reliant"),
                        "is_metacognitive": result.get("is_metacognitive"),
                        "is_delegation":    result.get("is_delegation"),
                        "reasoning":        reasoning,
                        "provider":         provider,
                        "model":            model,
                    })
            else:
                updated_levels.append(level)
                updated_ownership.append(t2_ownership)
                updated_critical.append(t2_critical)
                updated_self_reliant.append(t2_self_reliant)
                updated_metacognitive.append(t2_metacognitive)
                updated_delegation.append(t2_delegation)

        except Exception:
            # LLM failed — keep Tier 2 result
            updated_levels.append(level)
            updated_ownership.append(t2_ownership)
            updated_critical.append(t2_critical)
            updated_self_reliant.append(t2_self_reliant)
            updated_metacognitive.append(t2_metacognitive)
            updated_delegation.append(t2_delegation)

    if updated_levels:
        n = len(updated_levels)
        sig.ran = True
        sig.messages_reclassified = reclassified
        sig.hypothesis_level_avg = sum(updated_levels) / n
        sig.ownership_score = sum(1 for x in updated_ownership if x) / n
        sig.critical_engagement = sum(1 for x in updated_critical if x) / n
        sig.self_reliance = sum(1 for x in updated_self_reliant if x) / n
        sig.metacognition_score = sum(1 for x in updated_metacognitive if x) / n
        sig.delegation_penalty = sum(1 for x in updated_delegation if x) / n

    # Persist high-confidence classifications for prototype self-improvement
    if sig.prototype_candidates:
        _save_prototype_candidates(sig.prototype_candidates)

    return sig


def _save_prototype_candidates(candidates: list[dict]) -> None:
    """Append new high-confidence LLM classifications to the learned prototypes file.

    File: ~/.mindcheck/learned_prototypes.json
    Format: list of {text, level, is_*, reasoning, provider, model}
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
        max_tokens=200,
        messages=[{"role": "user",
                   "content": _CLASSIFICATION_PROMPT.format(message=text)}],
    )
    return json.loads(response.content[0].text)


def _call_openai(text: str, api_key: str, model: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=200,
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
            max_output_tokens=200,
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
