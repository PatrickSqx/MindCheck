"""
Tier 2: Semantic signal extraction via local embeddings.
Uses sentence-transformers (runs offline, no API cost).
Handles phrasing variation — catches signals that keywords miss.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from mindcheck.parser import Session


@dataclass
class SemanticSignals:
    hypothesis_level_avg: float = 0.0    # 0–4 avg hypothesis quality
    agency_score: float = 0.0            # 0–1 fraction of user-driven turns
    critical_engagement: float = 0.0     # 0–1 fraction with pushback / verification
    self_reliance: float = 0.0           # 0–1 showed prior attempt
    metacognition_score: float = 0.0     # 0–1 reflected on own approach
    delegation_penalty: float = 0.0      # 0–1 proportion of outsourcing language

    # Per-message details for reporting
    per_message: list[dict] = field(default_factory=list)


# ── Prototype meanings ────────────────────────────────────────────────────────
# These are the "tuning knobs" — improve these descriptions to improve accuracy.
# Each prototype is a natural-language description of what the signal means,
# plus diverse examples to widen the semantic coverage.

PROTOTYPES: dict[str, list[str]] = {
    # Hypothesis levels (0–4)
    "hypothesis_0": [
        "Fix this. Do this for me. Make it work. Just do it. Write the code.",
        "This is broken. It doesn't work. I have an error.",
    ],
    "hypothesis_1": [
        "I'm getting a null pointer exception. There's an error in my code. It crashed.",
        "Something is wrong with the login. The tests are failing.",
    ],
    "hypothesis_2": [
        "The error happens on line 42. It fails when the user has no session.",
        "It breaks when I submit the form with an empty field.",
    ],
    "hypothesis_3": [
        "I think the problem is that the session isn't initialized before the middleware runs.",
        "My guess is it's a timing issue with the async call. Could it be the promise isn't awaited?",
        "I suspect the null check is missing here. It seems like the config isn't loaded yet.",
    ],
    "hypothesis_4": [
        "I tried moving the session init earlier but it still fails. Maybe it's the order of middleware?",
        "I tested with a hardcoded value and it worked, so the issue must be in how I'm reading the env variable.",
        "I already checked the network tab and the request is correct, so it must be a server-side issue.",
    ],

    # Agency
    "user_driven": [
        "I want to understand how this works. Can you explain the tradeoffs?",
        "I've designed the architecture like this — does this make sense?",
        "I'm thinking about using X approach. What would you change?",
        "Here's what I've built so far. I need help with this specific part.",
    ],
    "ai_driven": [
        "What should I do next? Tell me what to build. What's the best approach?",
        "Just decide for me. You know best. Whatever you think is fine.",
        "I'll do whatever you suggest. What do you recommend?",
    ],

    # Critical engagement
    "critical": [
        "Wait, that doesn't seem right because. I disagree — here's why.",
        "Are you sure about that? I thought it worked differently.",
        "That approach would break if X happens. You missed the edge case.",
        "I checked your solution and it has a bug. Let me explain.",
    ],
    "passive": [
        "Thanks, looks good. That works, great. Perfect, I'll use that.",
        "OK I'll copy that. Great answer, thank you.",
    ],

    # Self-reliance
    "self_reliant": [
        "I tried X but it didn't work. I already attempted Y. I've been debugging this for an hour.",
        "I figured it out but want to double-check. I solved it but curious if there's a better way.",
        "I've narrowed it down to these three lines. I think I know the issue.",
    ],
    "not_self_reliant": [
        "I have no idea where to start. I don't know how to do this. Can you just write it?",
        "I give up. Can you fix it for me? I don't understand the error.",
    ],

    # Metacognition
    "metacognitive": [
        "Am I approaching this the wrong way? What am I missing in my thinking?",
        "I'm not sure my mental model of this is right. Can you critique my approach?",
        "What blind spots might I have here? Is this the right way to think about it?",
        "I want to make sure I understand, not just copy the solution.",
    ],
    "not_metacognitive": [
        "Just give me the answer. I don't need the explanation. Skip the details.",
    ],

    # Delegation — outsourcing thinking/execution rather than engaging
    "delegation": [
        "Just do it. Fix it for me. Write the whole thing. Go ahead and implement it.",
        "Can you just handle this? Do whatever you think is best. You decide.",
        "Build this feature. Write this function. Generate the code. Create the file.",
        "I need you to write this for me. Can you take care of this? Please implement.",
        "Just complete it. Finish the rest. Do the remaining parts.",
    ],
    "not_delegation": [
        "I tried this approach and want to understand why it fails.",
        "Here's my attempt — what did I get wrong?",
        "Can you explain how this works so I can fix it myself?",
        "I want to understand the tradeoffs before we decide.",
    ],
}

_model = None
_prototype_embeddings: dict[str, np.ndarray] = {}


def _get_model():
    """Lazy-load the embedding model (first call only)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _model


def _get_prototype_embeddings() -> dict[str, np.ndarray]:
    """Compute prototype embeddings once, cache them."""
    global _prototype_embeddings
    if not _prototype_embeddings:
        model = _get_model()
        for key, texts in PROTOTYPES.items():
            combined = " ".join(texts)
            _prototype_embeddings[key] = model.encode(combined, normalize_embeddings=True)
    return _prototype_embeddings


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both already normalized


def _classify_message(text: str) -> dict:
    """
    Classify a single user message against all prototype categories.
    Returns similarity scores (0–1) for each signal.
    """
    model = _get_model()
    protos = _get_prototype_embeddings()

    msg_emb = model.encode(text, normalize_embeddings=True)

    def sim(key: str) -> float:
        return _cosine_sim(msg_emb, protos[key])

    # Hypothesis level (0–4 by highest similarity)
    hyp_scores = {
        0: sim("hypothesis_0"),
        1: sim("hypothesis_1"),
        2: sim("hypothesis_2"),
        3: sim("hypothesis_3"),
        4: sim("hypothesis_4"),
    }
    hypothesis_level = max(hyp_scores, key=hyp_scores.get)

    return {
        "hypothesis_level": hypothesis_level,
        "is_user_driven": sim("user_driven") > sim("ai_driven"),
        "is_critical": sim("critical") > sim("passive") + 0.05,  # small threshold
        "is_self_reliant": sim("self_reliant") > sim("not_self_reliant") + 0.05,
        "is_metacognitive": sim("metacognitive") > sim("not_metacognitive") + 0.1,
        "is_delegation": sim("delegation") > sim("not_delegation") + 0.02,
    }


def extract_semantic(session: Session) -> SemanticSignals:
    """Run Tier 2 embedding classification on all user messages."""
    sig = SemanticSignals()
    user_msgs = session.user_messages

    if not user_msgs:
        return sig

    classifications = []
    for msg in user_msgs:
        if not msg.content.strip():
            continue
        result = _classify_message(msg.content)
        result["content"] = msg.content[:100]
        classifications.append(result)

    if not classifications:
        return sig

    n = len(classifications)
    sig.hypothesis_level_avg = sum(c["hypothesis_level"] for c in classifications) / n
    sig.agency_score = sum(1 for c in classifications if c["is_user_driven"]) / n
    sig.critical_engagement = sum(1 for c in classifications if c["is_critical"]) / n
    sig.self_reliance = sum(1 for c in classifications if c["is_self_reliant"]) / n
    sig.metacognition_score = sum(1 for c in classifications if c["is_metacognitive"]) / n
    sig.delegation_penalty = sum(1 for c in classifications if c["is_delegation"]) / n
    sig.per_message = classifications

    return sig
