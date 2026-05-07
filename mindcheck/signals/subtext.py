"""
Subtext detection — illocutionary intent analysis.

Detects the gap between what users *say* and what they *actually intend*.
Works in two modes:

  Local (free):   Pattern-based contradiction detection across message
                  sequences — catches say-then-contradict, empty
                  self-reliance, passive acceptance streaks.

  LLM (Tier 3):   Sends short message sequences to an LLM to infer
                  illocutionary intent — catches performative hypothesis,
                  fake curiosity, surface ownership.

The subtext score acts as a modifier on the composite score: if
engagement looks performative, the score gets penalized.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from mindcheck.parser import Session

SUBTEXT_DETECTION_VERSION = 1


@dataclass
class SubtextSignals:
    # Overall
    authenticity_score: float = 1.0       # 0–1, how genuine the engagement appears
    contradictions_found: int = 0         # number of say-then-contradict patterns
    performative_count: int = 0           # messages flagged as performative

    # Specific patterns detected
    say_then_contradict: int = 0          # "I want to understand" → "just give me code"
    empty_self_reliance: int = 0          # claims effort but gives no specifics
    passive_acceptance_streak: int = 0    # consecutive "looks good" / "makes sense"
    hypothesis_without_followup: int = 0  # forms hypothesis, never references it again
    performative_hypothesis: int = 0      # LLM-detected fake hypothesis (Tier 3 only)
    fake_curiosity: int = 0              # LLM-detected fake curiosity (Tier 3 only)

    # Per-message flags (parallel to semantic.per_message)
    flags: list[dict] = field(default_factory=list)

    # LLM subtext (Tier 3 only)
    llm_ran: bool = False
    llm_intents: list[dict] = field(default_factory=list)  # per-message intent analysis


# ── Local pattern detection (free, runs on per-message classifications) ──────

def extract_subtext_local(
    session: Session,
    classifications: list[dict],
) -> SubtextSignals:
    """Detect subtext patterns from Tier 2 per-message classifications.

    Analyses message *sequences* to find contradictions between what
    users say and what they actually do. Requires the per-message
    classification list from semantic.extract_semantic().

    Args:
        session: The parsed session.
        classifications: List of per-message classification dicts from Tier 2.

    Returns:
        SubtextSignals with local pattern detections.
    """
    sig = SubtextSignals()

    if len(classifications) < 2:
        return sig

    user_msgs = session.user_messages
    flags = []

    for i, c in enumerate(classifications):
        msg_flags = []

        # ── 1. Say-then-contradict ───────────────────────────────────────
        # User claims engagement in message i, then delegates in message i+1
        if i < len(classifications) - 1:
            next_c = classifications[i + 1]

            # "I want to understand" → "just do it"
            if (c.get("is_metacognitive") or c.get("hypothesis_level", 0) >= 3):
                if next_c.get("is_delegation") and next_c.get("hypothesis_level", 0) <= 1:
                    sig.say_then_contradict += 1
                    msg_flags.append("say_then_contradict")

            # "I'll do it myself" → "write it for me"
            # Only flag if the delegation is a low-effort handoff (hyp <= 1).
            # If the delegation itself contains detailed thinking (hyp >= 2),
            # the user is giving specific instructions — that's continuation,
            # not contradiction.
            if c.get("is_self_reliant") and not c.get("is_delegation"):
                if (next_c.get("is_delegation") and
                        next_c.get("hypothesis_level", 0) <= 1):
                    sig.say_then_contradict += 1
                    msg_flags.append("say_then_contradict")

        # ── 2. Empty self-reliance ───────────────────────────────────────
        # Claims prior effort but message is too short to contain specifics
        if c.get("is_self_reliant"):
            content = c.get("content", "")
            # "I tried everything" = 19 chars, no specifics
            # "I tried adding a lock around the callback" = 42 chars, has specifics
            if len(content) < 40 and not _has_specifics(content):
                sig.empty_self_reliance += 1
                msg_flags.append("empty_self_reliance")

        # ── 3. Hypothesis without follow-up ──────────────────────────────
        # Forms hypothesis (level 3+) but never references it in later messages
        if c.get("hypothesis_level", 0) >= 3 and i < len(classifications) - 1:
            # Check if any subsequent message references the hypothesis
            hypothesis_text = c.get("content", "")
            has_followup = False
            for j in range(i + 1, min(i + 4, len(classifications))):
                later = classifications[j]
                # Follow-up = tests hypothesis (level 4), or references it critically
                if later.get("hypothesis_level", 0) >= 4:
                    has_followup = True
                    break
                if later.get("is_critical") and later.get("hypothesis_level", 0) >= 2:
                    has_followup = True
                    break
            # Only flag if they dropped the hypothesis AND started delegating
            if not has_followup:
                for j in range(i + 1, min(i + 3, len(classifications))):
                    if classifications[j].get("is_delegation"):
                        sig.hypothesis_without_followup += 1
                        msg_flags.append("hypothesis_without_followup")
                        break

        flags.append({"index": i, "flags": msg_flags})

    # ── 4. Passive acceptance streak ─────────────────────────────────────
    # Count longest streak of passive (non-critical, non-questioning) messages
    streak = 0
    max_streak = 0
    for c in classifications:
        if (not c.get("is_critical") and
            not c.get("is_metacognitive") and
            not c.get("is_user_driven") and
            c.get("hypothesis_level", 0) <= 1):
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    sig.passive_acceptance_streak = max_streak

    # ── Compute totals ───────────────────────────────────────────────────
    sig.contradictions_found = sig.say_then_contradict + sig.empty_self_reliance
    sig.performative_count = (sig.say_then_contradict +
                              sig.empty_self_reliance +
                              sig.hypothesis_without_followup)
    sig.flags = flags

    # ── Authenticity score ───────────────────────────────────────────────
    # Start at 1.0 (fully authentic), penalise based on *rate* of patterns
    # found (not absolute counts), so long sessions aren't unfairly penalised.
    n = max(len(classifications), 1)
    penalty = 0.0

    # Rate-based: what fraction of messages showed each pattern?
    # Then scale by a severity weight and cap each component.
    contradiction_rate = sig.say_then_contradict / n
    penalty += min(contradiction_rate * 2.0, 0.30)   # up to -30% for contradictions

    empty_rate = sig.empty_self_reliance / n
    penalty += min(empty_rate * 1.5, 0.15)            # up to -15% for empty claims

    dropped_rate = sig.hypothesis_without_followup / n
    penalty += min(dropped_rate * 1.5, 0.15)          # up to -15% for dropped hypotheses

    # Passive acceptance streak penalty (only if streak is significant)
    if max_streak >= 3:
        streak_ratio = max_streak / n
        penalty += min(streak_ratio * 1.0, 0.20)     # up to -20% for passive streaks

    sig.authenticity_score = max(0.0, min(1.0, 1.0 - penalty))

    return sig


# ── LLM subtext analysis (Tier 3) ───────────────────────────────────────────

_SUBTEXT_PROMPT = """\
You are analysing a short sequence of messages that a person sent to an AI assistant.
Your job is to infer the SUBTEXT — the underlying intent behind the words,
not just what they literally say. Humans often say one thing and mean another.

Look for:
- Performative engagement: saying "I think..." without actually believing it,
  just to appear engaged before asking the AI to do the work
- Fake curiosity: "can you explain?" when they really just want the answer
- Surface ownership: "I want to try X" but they're actually asking AI to do X
- Inflated effort: "I tried everything" when they didn't really try
- Genuine engagement: real hypotheses, real curiosity, real critical thinking

Messages (in order):
{messages}

For EACH message, respond with ONLY valid JSON — a list of objects:
[
  {{"index": 0, "surface": "<what they literally say>", "subtext": "<what they actually mean>", "is_performative": <true/false>, "pattern": "<none|performative_hypothesis|fake_curiosity|surface_ownership|inflated_effort>", "confidence": <0.0-1.0>}},
  ...
]"""


def extract_subtext_llm(
    session: Session,
    classifications: list[dict],
    cfg: dict,
) -> SubtextSignals:
    """Run LLM-based subtext analysis on message sequences.

    Sends groups of 3-5 consecutive user messages to the LLM to infer
    illocutionary intent. Only called when Tier 3 is configured.

    Args:
        session: The parsed session.
        classifications: Per-message classification list from Tier 2.
        cfg: MindCheck config dict (must have tier3_provider set).

    Returns:
        SubtextSignals with LLM intent analysis.
    """
    import json

    sig = extract_subtext_local(session, classifications)

    provider = cfg.get("tier3_provider")
    if not provider:
        return sig

    key = cfg.get("tier3_key")
    model = cfg.get("tier3_model") or _default_model(provider)
    ollama_url = cfg.get("tier3_ollama_url", "http://localhost:11434")
    ollama_model = cfg.get("tier3_ollama_model", "llama3.2")

    user_msgs = session.user_messages

    # Build set of flagged message indices from local detection —
    # only send windows containing at least one flagged message to the LLM.
    flagged_indices = set()
    for f in sig.flags:
        if f.get("flags"):
            flagged_indices.add(f["index"])

    # Build message windows (groups of up to 5 consecutive messages)
    window_size = 5
    all_intents = []

    for start in range(0, len(user_msgs), window_size):
        end = min(start + window_size, len(user_msgs))

        # Skip windows with no locally-flagged messages
        if not any(i in flagged_indices for i in range(start, end)):
            continue

        window_msgs = user_msgs[start:end]

        # Format messages for the prompt
        msg_text = "\n".join(
            f"[{i}] {m.content[:300]}"
            for i, m in enumerate(window_msgs)
        )

        prompt = _SUBTEXT_PROMPT.format(messages=msg_text)

        try:
            result = _call_provider(prompt, provider, key, model,
                                    ollama_url, ollama_model)
            if isinstance(result, list):
                for item in result:
                    item["global_index"] = start + item.get("index", 0)
                    all_intents.append(item)

                    if item.get("is_performative"):
                        pattern = item.get("pattern", "none")
                        if pattern == "performative_hypothesis":
                            sig.performative_hypothesis += 1
                        elif pattern == "fake_curiosity":
                            sig.fake_curiosity += 1
                        sig.performative_count += 1
        except Exception:
            continue

    sig.llm_ran = True
    sig.llm_intents = all_intents

    # Adjust authenticity score with LLM findings
    if all_intents:
        n_performative = sum(1 for i in all_intents if i.get("is_performative"))
        n_total = len(all_intents)
        if n_total > 0:
            llm_penalty = (n_performative / n_total) * 0.30  # up to 30% penalty
            sig.authenticity_score = max(0.0, sig.authenticity_score - llm_penalty)

    return sig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _has_specifics(text: str) -> bool:
    """Check if a 'I tried...' message contains specific details."""
    # Specifics: line numbers, file names, error messages, function names,
    # technical terms, multiple attempts described
    specifics_patterns = re.compile(
        r'(line \d|error|exception|function|method|file|class|module|log|output|result'
        r'|\.py|\.js|\.ts|\.java|\.go|\.rs'  # file extensions
        r'|tried .+ and .+|tested .+ but'     # multiple attempts
        r'|第\d|行\d|函数|方法|文件|错误|日志|输出'  # Chinese specifics
        r')',
        re.I,
    )
    return bool(specifics_patterns.search(text))


def _default_model(provider: str) -> str:
    return {
        "anthropic": "claude-haiku-4-5",
        "openai":    "gpt-4.1-nano",
        "gemini":    "gemini-2.5-flash",
        "ollama":    "llama3.2",
    }.get(provider, "")


# ── Provider dispatch (reuses same pattern as llm.py) ────────────────────────

def _call_provider(prompt: str, provider: str, key: Optional[str],
                   model: str, ollama_url: str, ollama_model: str) -> list:
    import json

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.content[0].text)

    elif provider == "openai":
        import openai
        client = openai.OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model, max_tokens=1000,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(response.choices[0].message.content)
        return result if isinstance(result, list) else result.get("messages", [])

    elif provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=1000,
            ),
        )
        result = json.loads(response.text)
        return result if isinstance(result, list) else result.get("messages", [])

    elif provider == "ollama":
        import urllib.request
        payload = json.dumps({
            "model": ollama_model, "prompt": prompt,
            "format": "json", "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            parsed = json.loads(result["response"])
            return parsed if isinstance(parsed, list) else parsed.get("messages", [])

    raise ValueError(f"Unknown provider: {provider}")
