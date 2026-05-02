"""
Session type classifier — detects whether a conversation is coding, research,
creative, or casual, so scoring weights and prototype interpretation can adapt.

Uses the same multilingual embedding model as Tier 2. Classification is based
on the aggregate signal from all user messages, not individual messages.

Session types:
  - coding:   debugging, implementation, config, data work
  - research: learning, exploring concepts, comparing approaches
  - creative: writing, drafting, content creation
  - casual:   general chat, quick questions, no dominant theme
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from mindcheck.parser import Session


@dataclass
class SessionType:
    type: str = "coding"            # "coding" | "research" | "creative" | "casual"
    confidence: float = 0.0         # gap between top and runner-up (higher = more certain)
    scores: dict = None             # per-type similarity scores for debugging

    def __post_init__(self):
        if self.scores is None:
            self.scores = {}


# ── Session type prototypes ──────────────────────────────────────────────────
# These capture the *flavor* of an entire session type — what kinds of messages
# dominate that type of conversation.

SESSION_TYPE_PROTOTYPES: dict[str, list[str]] = {
    "coding": [
        # Bug fixing & debugging
        "Fix this bug. Debug this error. I'm getting an exception.",
        "The function returns null. The test is failing. My code crashes.",
        "I tried debugging with breakpoints. The stack trace shows this error.",
        # Implementation & features
        "Write a function to do X. Implement this feature. Add a button.",
        "Refactor this code. Add unit tests. Review this pull request.",
        "How do I implement X in React? Help me write a SQL query.",
        # Config & devops
        "Set up Docker. Configure CI/CD. Deploy to production.",
        "The build is failing. Permission denied. Port already in use.",
        "Install this package. Configure nginx. Write a Dockerfile.",
        # Data & ML (technical)
        "Train a classifier. Build a model. Optimize this query.",
        "The model is overfitting. Feature engineering. Cross-validation.",
        "Clean this CSV. Parse this data. Fix the preprocessing pipeline.",
    ],
    "research": [
        # Conceptual questions
        "What is X and how does it work? Explain how transformers work.",
        "What's the difference between X and Y? Compare these approaches.",
        "How does Z work under the hood? What causes this phenomenon?",
        # Learning & understanding
        "I want to understand X better. Can you teach me about Y?",
        "Walk me through the concepts. What does the research say?",
        "Give me an overview of this topic. What are the tradeoffs?",
        # Analysis & evaluation
        "What are the pros and cons? Analyze the implications.",
        "What are the risks? How do these options compare?",
        "Summarize this paper. What's the current state of the art?",
        # Strategic thinking
        "Help me think through this decision. What should I consider?",
        "What's the best way to approach this problem conceptually?",
        "I'm trying to understand the broader picture here.",
    ],
    "creative": [
        # Writing & drafting
        "Write a poem about X. Create a story. Draft a blog post.",
        "Write an introduction for my essay. Edit this paragraph.",
        "Rewrite this to be more concise. Improve the flow.",
        # Content creation
        "Draft an email. Write a cover letter. Create a report.",
        "Help me write a speech. Compose a message. Write copy.",
        "Create a social media post. Write a product description.",
        # Style & iteration
        "Make it more dramatic. Change the tone to be formal.",
        "Make it sound more professional. Shorten this. Polish it.",
        "The mood isn't right. Make it funnier. More emotional.",
        # Translation & adaptation
        "Translate this to English. Rewrite for a different audience.",
        "Adapt this for social media. Convert to bullet points.",
    ],
    "casual": [
        # General chat
        "What do you think about X? Just a quick question.",
        "I'm curious about something. Random thought.",
        "Hey, can you help me with something quick?",
        # Opinion & advice
        "What's your opinion on this? Do you think X is a good idea?",
        "Give me some advice about X. What would you do?",
        "Should I do X or Y? Quick recommendation.",
        # Life & personal
        "Help me plan my day. What should I eat for dinner?",
        "Tell me something interesting. I'm bored.",
        "What's a good movie to watch? Any book recommendations?",
        # Quick lookups
        "What's the capital of X? How many Y in Z?",
        "When was X invented? Who founded Y?",
        "Quick math: what's X times Y?",
    ],
}

SESSION_TYPE_PROTOTYPES_ZH: dict[str, list[str]] = {
    "coding_zh": [
        "修这个bug。代码报错了。调试一下。函数返回空。",
        "测试没通过。编译失败。运行报错。实现这个功能。",
        "写一个函数。重构代码。加单元测试。代码review。",
        "配置Docker。部署。CI/CD。安装这个包。构建失败。",
        "训练模型。优化查询。清洗数据。过拟合了。特征工程。",
        "帮我debug。写一个API。这个类有问题。接口调不通。",
    ],
    "research_zh": [
        "X是什么？怎么工作的？解释一下原理。",
        "X和Y有什么区别？对比一下这几种方案。",
        "我想了解X。教我Y。底层原理是什么？",
        "这个领域的研究现状是什么？有什么好的论文推荐？",
        "帮我分析一下利弊。这个方案的风险是什么？",
        "为什么会出现这种现象？背后的逻辑是什么？",
        "帮我理解一下这个概念。整体思路是什么？",
    ],
    "creative_zh": [
        "写一首诗。写个故事。帮我写博客。创作一篇文章。",
        "写一个开头。编辑这段话。改通顺一点。改得更有感染力。",
        "帮我写邮件。写求职信。写文案。写报告。起草一份方案。",
        "语气改正式一点。改幽默一点。更有感染力。风格改一下。",
        "翻译成英文。改写一下。润色。缩短一些。改成口语化。",
        "帮我写一段介绍。写一个方案。创作一下。写一段描述。",
        "这段话不太好。改得更简洁。措辞调整一下。",
        "帮我改一下这个结尾。语气要温暖但专业。",
    ],
    "casual_zh": [
        "你觉得X怎么样？随便问一下。闲聊。",
        "推荐一下。有什么建议？你怎么看？",
        "帮我想想。随便说说。快速问一下。",
        "今天吃什么？有什么好电影推荐吗？",
        "X是什么时候发明的？谁创建了Y？",
        "这个有意思。无聊了。说点什么。",
    ],
}


# ── Scoring weight profiles per session type ─────────────────────────────────
# Different session types reward different behaviors.

SCORING_WEIGHTS: dict[str, dict] = {
    "coding": {
        "hypothesis":          0.25,
        "ownership":           0.20,
        "critical_engagement": 0.20,
        "self_reliance":       0.15,
        "metacognition":       0.10,
        "structural":          0.05,
        "delegation_max":      20,      # full penalty
    },
    "research": {
        "hypothesis":          0.15,    # asking good questions ≠ forming code hypotheses
        "ownership":           0.15,    # steering the inquiry
        "critical_engagement": 0.25,    # questioning AI explanations is crucial
        "self_reliance":       0.10,    # prior reading
        "metacognition":       0.20,    # reflecting on understanding is key
        "structural":          0.05,
        "delegation_max":      10,      # "explain X" isn't delegation in research
    },
    "creative": {
        "hypothesis":          0.10,    # less relevant for creative work
        "ownership":           0.30,    # creative direction IS engagement
        "critical_engagement": 0.25,    # iterating on drafts, pushing back
        "self_reliance":       0.05,    # less relevant
        "metacognition":       0.10,
        "structural":          0.05,
        "delegation_max":      10,      # "write X for me" IS the workflow
    },
    "casual": {
        "hypothesis":          0.15,
        "ownership":           0.25,    # steering the conversation
        "critical_engagement": 0.20,
        "self_reliance":       0.10,
        "metacognition":       0.10,
        "structural":          0.10,
        "delegation_max":      5,       # casual chat has low expectations
    },
}


def classify_session_type(session: Session) -> SessionType:
    """Classify a session's type based on its user messages.

    Embeds all user messages, compares against session-type prototypes,
    and returns the best-matching type. Uses majority-of-messages approach:
    each message votes for a type, and the type with the most votes wins.
    Falls back to "coding" for CLI tool sessions (Claude Code, Cursor, etc.)
    with too few messages to classify reliably.
    """
    from mindcheck.signals.semantic import _get_model

    user_msgs = session.user_messages
    meaningful = [m for m in user_msgs if len(m.content.strip()) >= 12]

    # Default: CLI tool sessions are coding unless proven otherwise
    default_type = "coding" if session.tool in ("claude", "cursor", "codex", "gemini") else "casual"

    if len(meaningful) < 3:
        return SessionType(type=default_type, confidence=0.0)

    model = _get_model()
    proto_embs = _get_session_type_embeddings()

    # Classify each message and tally votes
    type_votes: dict[str, float] = {"coding": 0, "research": 0, "creative": 0, "casual": 0}

    for msg in meaningful:
        msg_emb = model.encode(msg.content.strip(), normalize_embeddings=True)
        best_type = None
        best_score = -1.0

        for stype in type_votes:
            # Max of EN and ZH similarity
            en_score = float(np.dot(msg_emb, proto_embs[stype]))
            zh_key = f"{stype}_zh"
            zh_score = float(np.dot(msg_emb, proto_embs[zh_key])) if zh_key in proto_embs else 0.0
            score = max(en_score, zh_score)

            if score > best_score:
                best_score = score
                best_type = stype

        type_votes[best_type] += 1

    # Normalize to proportions
    total = sum(type_votes.values())
    type_scores = {k: v / total for k, v in type_votes.items()}

    # Winner = highest proportion
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    winner = sorted_types[0][0]
    confidence = sorted_types[0][1] - sorted_types[1][1]

    # If confidence is very low and it's a CLI tool, default to coding
    if confidence < 0.05 and session.tool in ("claude", "cursor", "codex", "gemini"):
        winner = "coding"

    return SessionType(
        type=winner,
        confidence=round(confidence, 3),
        scores={k: round(v, 3) for k, v in type_scores.items()},
    )


# ── Prototype embeddings (cached) ────────────────────────────────────────────

_session_type_embeddings: dict[str, np.ndarray] = {}


def _get_session_type_embeddings() -> dict[str, np.ndarray]:
    """Compute and cache session-type prototype embeddings."""
    global _session_type_embeddings
    if not _session_type_embeddings:
        from mindcheck.signals.semantic import _get_model
        model = _get_model()

        for key, texts in SESSION_TYPE_PROTOTYPES.items():
            combined = " ".join(texts)
            _session_type_embeddings[key] = model.encode(combined, normalize_embeddings=True)

        for key, texts in SESSION_TYPE_PROTOTYPES_ZH.items():
            combined = " ".join(texts)
            _session_type_embeddings[key] = model.encode(combined, normalize_embeddings=True)

    return _session_type_embeddings
