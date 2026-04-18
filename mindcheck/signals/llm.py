"""
Tier 3: LLM-based classification for ambiguous messages.
Only called when Tier 2 embedding confidence is low.
Only user messages are sent — never full sessions.
Results are cached to minimise API cost.
"""

from dataclasses import dataclass, field
from mindcheck.parser import Session


@dataclass
class LLMSignals:
    ran: bool = False
    confidence_adjustment: float = 0.0   # ± adjustment to composite score
    ambiguous_messages: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_CLASSIFICATION_PROMPT = """
You are evaluating a single message a person sent to an AI assistant.
Your job is to assess the person's cognitive engagement — are they thinking
for themselves, or outsourcing their thinking to the AI?

Message:
{message}

Classify ONLY what is explicitly present. Do not infer beyond the text.

Return a JSON object with these fields:
{{
  "hypothesis_level": 0-4,
  "is_user_driven": true/false,
  "is_critical": true/false,
  "is_self_reliant": true/false,
  "is_metacognitive": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "one sentence"
}}

Hypothesis levels:
  0 = no attempt, pure delegation ("fix this", "just do it")
  1 = describes symptom only ("there's an error")
  2 = locates the problem ("it fails on line 42")
  3 = forms a hypothesis ("I think X because Y")
  4 = tested a hypothesis ("I tried X, it still fails, so maybe Y")
"""


def extract_llm(
    session: Session,
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    confidence_threshold: float = 0.6,
) -> LLMSignals:
    """
    Run Tier 3 LLM classification on ambiguous user messages.

    Only messages where the Tier 2 embedding confidence was below
    confidence_threshold should be passed here. In practice, call this
    from the scorer after Tier 2 has flagged low-confidence messages.

    Args:
        session:              The session being analysed
        provider:             "anthropic" | "openai" | "ollama"
        api_key:              API key (not needed for Ollama)
        model:                Model to use (defaults to cheapest available)
        confidence_threshold: Only process messages below this confidence
    """
    sig = LLMSignals()
    user_msgs = session.user_messages

    if not user_msgs:
        return sig

    # TODO: integrate with Tier 2 confidence scores to only send ambiguous messages
    # For now, this is a stub that can be wired up when the scorer passes
    # per-message confidence data through.

    sig.ran = False
    sig.notes.append("Tier 3 LLM classification not yet wired up — skipped.")
    return sig


def _call_anthropic(message: str, api_key: str, model: str) -> dict:
    """Call Anthropic API to classify a single message."""
    import json
    import anthropic

    model = model or "claude-haiku-4-20250506"  # cheapest, fast
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": _CLASSIFICATION_PROMPT.format(message=message)
        }]
    )

    text = response.content[0].text
    return json.loads(text)


def _call_openai(message: str, api_key: str, model: str) -> dict:
    """Call OpenAI API to classify a single message."""
    import json
    import openai

    model = model or "gpt-4o-mini"
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": _CLASSIFICATION_PROMPT.format(message=message)
        }]
    )

    return json.loads(response.choices[0].message.content)


def _call_ollama(message: str, model: str) -> dict:
    """Call local Ollama instance to classify a single message."""
    import json
    import urllib.request

    model = model or "llama3.2"
    payload = json.dumps({
        "model": model,
        "prompt": _CLASSIFICATION_PROMPT.format(message=message),
        "format": "json",
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        return json.loads(result["response"])
