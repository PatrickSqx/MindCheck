"""
Tier 1: Structural signal extraction.
Pure math — no LLM, no embeddings, no API cost.
Captures signals that don't depend on language or phrasing.
"""

import re
from dataclasses import dataclass, field
from mindcheck.parser import Session


@dataclass
class StructuralSignals:
    # Ratios
    question_ratio: float = 0.0      # fraction of user messages containing "?"
    message_ratio: float = 0.0       # user chars / AI chars (higher = more engaged)
    turn_count: int = 0              # number of user turns

    # Language patterns (high-confidence keywords only)
    urgency_count: int = 0           # "asap", "quickly", "urgent", "hurry"
    delegation_count: int = 0        # "just do it", "just fix", "just write"
    gratitude_count: int = 0         # "thanks", "perfect", "great" (often = passive)
    prior_attempt_count: int = 0     # "I tried", "I tested", "I checked"

    # Structural patterns
    avg_user_length: float = 0.0     # average characters per user message
    long_message_ratio: float = 0.0  # fraction of messages > 200 chars (more context = engaged)
    single_line_ratio: float = 0.0   # fraction of messages that are one short line


# High-confidence keyword sets (phrasing-agnostic signals are handled by Tier 2)
# Each pattern covers English + common CJK equivalents.

_URGENCY = re.compile(
    r'(\b(asap|urgent|urgently|quickly|hurry|right now|immediately|deadline)\b'
    r'|马上|立刻|赶紧|紧急|尽快|快点|截止|deadline)',
    re.I
)
_DELEGATION = re.compile(
    r'(\bjust\s+(do|fix|write|make|create|add|implement|change|update|delete|remove|build)\b'
    r'|帮我(写|做|修|创建|实现|添加|删除|改|生成|完成|处理)'
    r'|直接(写|做|修|帮我|实现|生成))',
    re.I
)
_GRATITUDE = re.compile(
    r'(\b(thanks|thank you|perfect|great|awesome|looks good|that works|nice)\b'
    r'|谢谢|好的|完美|太好了|可以|没问题)',
    re.I
)
_PRIOR_ATTEMPT = re.compile(
    r'(\b(i tried|i tested|i checked|i already|i attempted|i ran|i moved|i added|i changed)\b'
    r'|我试过|我尝试|我检查|我已经|我测试|我发现|我发现了)',
    re.I
)


def extract_structural(session: Session) -> StructuralSignals:
    sig = StructuralSignals()
    user_msgs = session.user_messages

    if not user_msgs:
        return sig

    sig.turn_count = len(user_msgs)

    # Message length stats
    lengths = [len(m.content) for m in user_msgs]
    sig.avg_user_length = sum(lengths) / len(lengths)
    sig.long_message_ratio = sum(1 for l in lengths if l > 200) / len(lengths)
    sig.single_line_ratio = sum(1 for l in lengths if l < 60) / len(lengths)

    # User/AI char ratio
    ai_chars = session.total_ai_chars
    user_chars = session.total_user_chars
    sig.message_ratio = user_chars / ai_chars if ai_chars > 0 else 1.0

    # Per-message signals
    question_count = 0
    for msg in user_msgs:
        text = msg.content

        if "?" in text or "？" in text:
            question_count += 1

        if _URGENCY.search(text):
            sig.urgency_count += 1

        if _DELEGATION.search(text):
            sig.delegation_count += 1

        if _GRATITUDE.search(text):
            sig.gratitude_count += 1

        if _PRIOR_ATTEMPT.search(text):
            sig.prior_attempt_count += 1

    sig.question_ratio = question_count / len(user_msgs)

    return sig
