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

SUBTEXT_DETECTION_VERSION = 2  # v2: say-then-contradict moved to Tier 3 confirmation


@dataclass
class SubtextSignals:
    # Overall
    authenticity_score: float = 1.0       # 0–1, how genuine the engagement appears
    contradictions_found: int = 0         # number of confirmed say-then-contradict
    performative_count: int = 0           # messages flagged as performative

    # Tier 2 candidates (flagged locally, NOT scored — need LLM confirmation)
    say_then_contradict_candidates: int = 0  # candidate pairs found by Tier 2
    say_then_contradict: int = 0          # LLM-confirmed contradictions (Tier 3 only)

    # Tier 2 reliable patterns (scored directly)
    empty_self_reliance: int = 0          # claims effort but gives no specifics
    passive_acceptance_streak: int = 0    # consecutive "looks good" / "makes sense"
    hypothesis_without_followup: int = 0  # forms hypothesis, never references it again

    # LLM-detected patterns (Tier 3 only)
    performative_hypothesis: int = 0      # LLM-detected fake hypothesis
    fake_curiosity: int = 0              # LLM-detected fake curiosity

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
            # Flagged as CANDIDATE only — Tier 2 can't distinguish pasted
            # content from genuine thought, or follow-up questions from
            # delegation.  LLM confirmation (Tier 3) required before scoring.
            if (c.get("is_metacognitive") or c.get("hypothesis_level", 0) >= 3):
                if next_c.get("is_delegation") and next_c.get("hypothesis_level", 0) <= 1:
                    sig.say_then_contradict_candidates += 1
                    msg_flags.append("say_then_contradict_candidate")

            # "I'll do it myself" → "write it for me"
            if c.get("is_self_reliant") and not c.get("is_delegation"):
                if (next_c.get("is_delegation") and
                        next_c.get("hypothesis_level", 0) <= 1):
                    sig.say_then_contradict_candidates += 1
                    msg_flags.append("say_then_contradict_candidate")

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
    # say_then_contradict stays 0 at Tier 2 — only set when LLM confirms.
    # contradictions_found counts only confirmed contradictions.
    sig.contradictions_found = sig.say_then_contradict + sig.empty_self_reliance
    sig.performative_count = (sig.say_then_contradict +
                              sig.empty_self_reliance +
                              sig.hypothesis_without_followup)
    sig.flags = flags

    # ── Authenticity score ───────────────────────────────────────────────
    # Start at 1.0 (fully authentic), penalise based on *rate* of patterns
    # found (not absolute counts), so long sessions aren't unfairly penalised.
    #
    # NOTE: say-then-contradict is NOT penalised here — Tier 2 can't
    # reliably distinguish pasted content / follow-up questions from real
    # contradictions (~100% false positive rate in testing).  The penalty
    # is applied in extract_subtext_llm() after LLM confirmation.
    n = max(len(classifications), 1)
    penalty = 0.0

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


_CONFIRM_STC_PROMPT = """\
A local detector flagged the following message pair as a potential
"say-then-contradict" pattern, where the user claims engagement then
immediately contradicts it by delegating.

However, the local detector has a high false-positive rate — it cannot
distinguish between:
- Pasted/copied content vs. the user's own thinking
- Follow-up questions vs. actual delegation
- Giving detailed instructions vs. mindless handoff
- Natural conversation flow vs. genuine contradiction

Message A (flagged as "engagement"):
{msg_a}

Message B (flagged as "delegation"):
{msg_b}

Is this a GENUINE say-then-contradict? Answer with ONLY valid JSON:
{{"is_contradiction": <true/false>, "reason": "<brief explanation>"}}

Rules:
- If Message A is pasted content (code, error logs, assignment text, data),
  it is NOT the user's own thinking — answer false.
- If Message B is a follow-up question (ends with ? or 吗/呢/吧), it is
  likely continued engagement, not delegation — answer false.
- If Message B contains specific instructions or observations, it is
  continuation, not contradiction — answer false.
- Only answer true if the user genuinely showed their OWN thinking in A,
  then dropped it and handed off mindlessly in B."""


def extract_subtext_llm(
    session: Session,
    classifications: list[dict],
    cfg: dict,
) -> SubtextSignals:
    """Run LLM-based subtext analysis on message sequences.

    Two-phase approach:
    1. Confirm say-then-contradict candidates — the local detector flags
       potential pairs but has ~100% false-positive rate on real data
       (can't distinguish pasted content from genuine thought).  The LLM
       reads each candidate pair and confirms or rejects it.
    2. Performative intent scan — sends message windows to the LLM to
       detect performative hypothesis, fake curiosity, etc.

    Only confirmed contradictions affect the authenticity score.

    Args:
        session: The parsed session.
        classifications: Per-message classification list from Tier 2.
        cfg: MindCheck config dict (must have tier3_provider set).

    Returns:
        SubtextSignals with LLM-confirmed contradictions and intent analysis.
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

    # ── Phase 1: Confirm say-then-contradict candidates ──────────────
    stc_candidate_indices = []
    for f in sig.flags:
        if "say_then_contradict_candidate" in f.get("flags", []):
            stc_candidate_indices.append(f["index"])

    confirmed_stc = 0
    for idx in stc_candidate_indices:
        if idx >= len(user_msgs) or idx + 1 >= len(user_msgs):
            continue
        msg_a = user_msgs[idx].content[:400]
        msg_b = user_msgs[idx + 1].content[:400]

        prompt = _CONFIRM_STC_PROMPT.format(msg_a=msg_a, msg_b=msg_b)
        try:
            result = _call_provider(prompt, provider, key, model,
                                    ollama_url, ollama_model)
            if isinstance(result, dict) and result.get("is_contradiction"):
                confirmed_stc += 1
                # Update the flag from candidate to confirmed
                for f in sig.flags:
                    if f["index"] == idx and "say_then_contradict_candidate" in f["flags"]:
                        f["flags"].append("say_then_contradict")
                        break
        except Exception:
            continue

    sig.say_then_contradict = confirmed_stc
    sig.contradictions_found = confirmed_stc + sig.empty_self_reliance

    # Apply say-then-contradict penalty only for LLM-confirmed cases
    if confirmed_stc > 0:
        n = max(len(classifications), 1)
        stc_rate = confirmed_stc / n
        stc_penalty = min(stc_rate * 2.0, 0.30)
        sig.authenticity_score = max(0.0, sig.authenticity_score - stc_penalty)

    # ── Phase 2: Performative intent scan ────────────────────────────
    # Build set of flagged message indices — only send windows with flags
    flagged_indices = set()
    for f in sig.flags:
        if f.get("flags"):
            flagged_indices.add(f["index"])

    window_size = 5
    all_intents = []

    for start in range(0, len(user_msgs), window_size):
        end = min(start + window_size, len(user_msgs))

        # Skip windows with no locally-flagged messages
        if not any(i in flagged_indices for i in range(start, end)):
            continue

        window_msgs = user_msgs[start:end]

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

    # Adjust authenticity score with LLM performative findings
    if all_intents:
        n_performative = sum(1 for i in all_intents if i.get("is_performative"))
        n_total = len(all_intents)
        if n_total > 0:
            llm_penalty = (n_performative / n_total) * 0.30
            sig.authenticity_score = max(0.0, sig.authenticity_score - llm_penalty)

    # Recompute totals after LLM confirmation
    sig.performative_count += confirmed_stc

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
