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

    # Per-task-domain breakdown: {domain → {count, hypothesis_avg, delegation_rate}}
    # Domains: code, data, writing, research, planning, config
    task_breakdown: dict = field(default_factory=dict)

    # Per-message details for reporting
    per_message: list[dict] = field(default_factory=list)


# ── Prototype meanings ────────────────────────────────────────────────────────
# These are the "tuning knobs" — improve these descriptions to improve accuracy.
# Each prototype is a natural-language description of what the signal means,
# plus diverse examples to widen the semantic coverage.

PROTOTYPES: dict[str, list[str]] = {
    # Hypothesis levels (0–4)
    # Each level should cover: coding, data analysis, research/writing, general reasoning
    "hypothesis_0": [
        # Coding
        "Fix this. Do this for me. Make it work. Just do it. Write the code.",
        "This is broken. It doesn't work. I have an error.",
        # Data / analysis
        "Run the analysis. Generate the report. Clean this data for me.",
        "Process this dataset. Build the model. Create the chart.",
        # Research / writing
        "Write this section. Summarize this paper. Find sources for this claim.",
        "Draft the introduction. Rewrite this paragraph. Create the outline.",
        # General
        "Handle this. Take care of it. Do this task for me. Just complete it.",
    ],
    "hypothesis_1": [
        # Coding
        "I'm getting a null pointer exception. There's an error in my code. It crashed.",
        "Something is wrong with the login. The tests are failing.",
        # Data / analysis
        "The chart looks wrong. The numbers don't add up. My results seem off.",
        "The model isn't performing well. The output isn't what I expected.",
        "Something is wrong with my analysis. The figures don't make sense.",
        # Research / writing
        "The argument doesn't flow well. The structure feels off.",
        "The explanation isn't clear. Something is missing from this section.",
        # General
        "This doesn't work correctly. The output is wrong. There's a problem somewhere.",
    ],
    "hypothesis_2": [
        # Coding
        "The error happens on line 42. It fails when the user has no session.",
        "It breaks when I submit the form with an empty field.",
        # Data / analysis
        "The outliers in column B are skewing the average.",
        "The model underperforms specifically on the test set, not training.",
        "It only fails when there are missing values in the date column.",
        "The issue appears in Q3 data but not Q1 — something changed mid-year.",
        # Research / writing
        "The argument breaks down in the third paragraph where I switch topics.",
        "The conclusion contradicts what I said in the introduction.",
        # General
        "The problem only happens under X condition. It works fine until Y occurs.",
        "It fails specifically when the input is large — small inputs are fine.",
    ],
    "hypothesis_3": [
        # Coding
        "I think the problem is that the session isn't initialized before the middleware runs.",
        "My guess is it's a timing issue with the async call. Could it be the promise isn't awaited?",
        "I suspect the null check is missing here. It seems like the config isn't loaded yet.",
        # Data / analysis
        "I suspect the model is overfitting — training accuracy is much higher than validation.",
        "My hypothesis is the correlation is spurious because X and Y are both driven by Z.",
        "I think the outliers are real signal, not noise — they cluster around a specific date.",
        "I believe the feature importance is misleading because the variables are collinear.",
        # Research / writing
        "I think the argument is weak because I'm assuming X without evidence.",
        "My guess is the reader loses track here because I haven't defined the key term yet.",
        "I suspect the structure is wrong — the conclusion should probably come earlier.",
        # General
        "I think the root cause is X because Y only happens when Z is true.",
        "My hypothesis is that A is caused by B, not C, because the pattern matches B.",
    ],
    "hypothesis_4": [
        # Coding
        "I tried moving the session init earlier but it still fails. Maybe it's the order of middleware?",
        "I tested with a hardcoded value and it worked, so the issue must be in how I'm reading the env variable.",
        "I already checked the network tab and the request is correct, so it must be a server-side issue.",
        # Data / analysis
        "I tested with a smaller subset and the pattern holds, so it's not a sample size issue.",
        "I already tried normalizing the data — problem persists, so it might be the model architecture.",
        "I checked both approaches: A has better precision but worse recall. I think we should optimise for precision here because of X.",
        "I removed the outliers and re-ran — the correlation weakened, which confirms they were driving it.",
        # Research / writing
        "I tried restructuring the argument but the same objection applies. Maybe the premise itself is wrong.",
        "I already cut 500 words and it's still too long. I think the second section can be merged with the third.",
        # General
        "I already ruled out X and Y by testing them separately. The only remaining explanation is Z.",
        "I tried both approaches — A is faster but B is more accurate. Given our constraints, I think B is right.",
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

    # ── Task domains ──────────────────────────────────────────────────────────
    # Used to classify what kind of task each message is about.
    # Higher similarity to a domain → message belongs to that domain.

    "task_code": [
        "Write a function to do X. Fix this bug. Implement this feature.",
        "I'm getting an error in my Python code. Debug this TypeScript.",
        "Refactor this code. Add unit tests. Review this pull request.",
        "How do I implement X in React? Help me write a SQL query.",
        "This class is broken. The API call is failing. The test won't pass.",
        "Add error handling. Optimise this loop. Why is this function slow?",
    ],
    "task_data": [
        "Analyse this dataset. Build a model to predict X. Train a classifier.",
        "Run a statistical test on this data. Create a visualization.",
        "Clean this CSV. Why is my pandas code slow? Optimise this query.",
        "The model isn't performing well. What features should I use?",
        "Plot this data. Calculate the correlation. Evaluate model accuracy.",
        "Feature engineering, data preprocessing, cross-validation, overfitting.",
    ],
    "task_writing": [
        "Write an introduction for my essay. Edit this paragraph.",
        "Summarize this article. Draft an email to my client.",
        "Rewrite this to be more concise. Improve the flow of this section.",
        "Write a cover letter. Create a report. Polish this text.",
        "The argument isn't clear. Help me structure this piece.",
        "Proofread this. Make it sound more professional. Shorten this.",
    ],
    "task_research": [
        "Explain how transformers work. What is X and how does it work?",
        "What's the difference between X and Y? How does Z work under the hood?",
        "Summarize this paper. What are the tradeoffs of approach X?",
        "I want to understand X better. Can you teach me about Y?",
        "What causes X? Why does Y happen? What's the best way to learn Z?",
        "Give me an overview of this topic. What does the research say?",
    ],
    "task_planning": [
        "Help me design the architecture for this system.",
        "How should I structure this project? What approach should I take?",
        "I need to plan a roadmap. What's the best design pattern here?",
        "Should I use X or Y for this? How should I organise the codebase?",
        "Think through the tradeoffs. Help me decide between these options.",
        "What are the risks? How do I prioritise? Plan out the next steps.",
    ],
    "task_config": [
        "How do I install X? Set up this environment. Configure this tool.",
        "My Docker container won't start. Deploy this to production.",
        "Set up CI/CD. Write a Makefile. Configure nginx. Set up the database.",
        "I can't get this package to install. The build is failing.",
        "Set up authentication. Configure environment variables. Write a Dockerfile.",
        "Permission denied. Port already in use. The server won't start.",
    ],
}

_model = None
_prototype_embeddings: dict[str, np.ndarray] = {}


def _get_model():
    """Lazy-load the embedding model (first call only)."""
    global _model
    if _model is None:
        try:
            import logging
            import warnings
            from sentence_transformers import SentenceTransformer

            # Suppress noisy HF/transformers warnings that confuse first-time users
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

            _MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

            # Show a loading message only if the model isn't cached locally yet
            from pathlib import Path
            import os
            cache_dir = Path(os.environ.get("SENTENCE_TRANSFORMERS_HOME",
                             Path.home() / ".cache" / "torch" / "sentence_transformers"))
            model_cached = any(cache_dir.glob(f"*{_MODEL_NAME}*")) if cache_dir.exists() else False

            if not model_cached:
                from rich.console import Console
                Console().print(
                    "[dim]Downloading embedding model (~118 MB, first run only)…[/dim]"
                )

            # paraphrase-multilingual-MiniLM-L12-v2: 50+ languages, ~118MB
            # Maps cross-lingual meaning to same embedding space —
            # English prototypes correctly classify Chinese/French/etc. input.
            _model = SentenceTransformer(_MODEL_NAME)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _model


def _get_prototype_embeddings() -> dict[str, np.ndarray]:
    """Compute prototype embeddings once, cache them.

    Merges built-in PROTOTYPES with any user-learned prototypes saved by Tier 3
    in ~/.mindcheck/learned_prototypes.json. Learned examples are appended to the
    relevant hypothesis_N prototype so the embeddings improve over time.
    """
    global _prototype_embeddings
    if not _prototype_embeddings:
        model = _get_model()

        # Start from a copy of built-in prototypes
        extended: dict[str, list[str]] = {k: list(v) for k, v in PROTOTYPES.items()}

        # Merge learned prototypes saved by Tier 3
        import json as _json
        from pathlib import Path
        learned_path = Path.home() / ".mindcheck" / "learned_prototypes.json"
        if learned_path.exists():
            try:
                learned = _json.loads(learned_path.read_text(encoding="utf-8"))
                added = 0
                for entry in learned:
                    level = entry.get("level")
                    text  = entry.get("text", "").strip()
                    if isinstance(level, int) and 0 <= level <= 4 and text:
                        extended[f"hypothesis_{level}"].append(text)
                        added += 1
                if added:
                    pass  # Loaded silently — no noise on every run
            except Exception:
                pass  # Corrupt file — fall back to built-ins only

        for key, texts in extended.items():
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
    sorted_hyp = sorted(hyp_scores.items(), key=lambda x: x[1], reverse=True)
    hypothesis_level = sorted_hyp[0][0]
    # Confidence = gap between top and runner-up. Small gap = uncertain classification.
    hypothesis_confidence = sorted_hyp[0][1] - sorted_hyp[1][1]

    # Task domain — highest similarity wins
    _TASK_DOMAINS = ("task_code", "task_data", "task_writing",
                     "task_research", "task_planning", "task_config")
    task_domain = max(_TASK_DOMAINS, key=sim).replace("task_", "")

    return {
        "hypothesis_level":      hypothesis_level,
        "hypothesis_confidence": hypothesis_confidence,   # 0.0 = tie, ~0.1+ = confident
        "task_domain":           task_domain,
        "is_user_driven":    sim("user_driven")    > sim("ai_driven"),
        "is_critical":       sim("critical")       > sim("passive")          + 0.05,
        "is_self_reliant":   sim("self_reliant")   > sim("not_self_reliant") + 0.05,
        "is_metacognitive":  sim("metacognitive")  > sim("not_metacognitive")+ 0.1,
        "is_delegation":     sim("delegation")     > sim("not_delegation")   + 0.02,
        "low_confidence":    hypothesis_confidence < 0.04,  # flag for prototype review
    }


def extract_semantic(session: Session) -> SemanticSignals:
    """Run Tier 2 embedding classification on all user messages."""
    sig = SemanticSignals()
    user_msgs = session.user_messages

    if not user_msgs:
        return sig

    classifications = []
    for msg in user_msgs:
        text = msg.content.strip()
        # Skip messages too short to classify meaningfully —
        # single words / button clicks / confirmations add noise, not signal.
        if len(text) < 12:
            continue
        result = _classify_message(text)
        result["content"] = text[:100]
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

    # ── Task domain breakdown ─────────────────────────────────────────────────
    raw_breakdown: dict[str, dict] = {}
    for c in classifications:
        domain = c["task_domain"]
        if domain not in raw_breakdown:
            raw_breakdown[domain] = {"count": 0, "hyp_sum": 0, "deleg_count": 0}
        raw_breakdown[domain]["count"] += 1
        raw_breakdown[domain]["hyp_sum"] += c["hypothesis_level"]
        raw_breakdown[domain]["deleg_count"] += 1 if c["is_delegation"] else 0

    sig.task_breakdown = {
        domain: {
            "count":           d["count"],
            "hypothesis_avg":  round(d["hyp_sum"] / d["count"], 2),
            "delegation_rate": round(d["deleg_count"] / d["count"], 2),
        }
        for domain, d in raw_breakdown.items()
        if d["count"] >= 2   # only report domains with at least 2 messages
    }

    return sig
