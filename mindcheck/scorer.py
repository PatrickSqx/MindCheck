"""
Scorer — aggregates signals into a composite cognitive engagement score.
"""

from dataclasses import dataclass, field
from mindcheck.parser import Session
from mindcheck.signals.structural import StructuralSignals, extract_structural
from mindcheck.signals.semantic import SemanticSignals, extract_semantic
from mindcheck.signals.llm import LLMSignals, extract_llm


@dataclass
class SessionScore:
    session: Session
    structural: StructuralSignals = field(default_factory=StructuralSignals)
    semantic: SemanticSignals = field(default_factory=SemanticSignals)
    llm: LLMSignals = field(default_factory=LLMSignals)
    composite: float = 0.0

    def compute_composite(self):
        """
        Composite score (0–100) weighted across signal categories.

        Weights reflect how much each dimension contributes to genuine
        cognitive engagement:
          - Question framing / hypothesis: 25%  (most direct signal)
          - Agency:                        20%  (who drives the work)
          - Critical engagement:           20%  (do they evaluate output)
          - Learning / self-reliance:      20%  (do concepts stick)
          - Metacognition:                 10%  (self-reflection)
          - Structural ratios:              5%  (supporting evidence)
        """
        s = self.structural
        sem = self.semantic
        llm = self.llm

        # Structural (5%)
        structural_score = (
            min(s.question_ratio * 100, 100) * 0.4 +       # asks questions
            min(s.turn_count / 10 * 100, 100) * 0.3 +      # engages in dialogue
            min(s.message_ratio * 200, 100) * 0.3           # writes substantive messages
        )

        # Semantic (from Tier 2 embeddings, 70% of score)
        hypothesis_score   = sem.hypothesis_level_avg / 4 * 100   # 0-4 → 0-100
        agency_score       = sem.agency_score * 100                # 0-1 → 0-100
        critical_score     = sem.critical_engagement * 100
        self_reliance      = sem.self_reliance * 100
        metacognition      = sem.metacognition_score * 100

        # LLM refinement (Tier 3, adjusts when available)
        llm_adjustment = llm.confidence_adjustment if llm.ran else 0.0

        composite = (
            structural_score    * 0.05 +
            hypothesis_score    * 0.25 +
            agency_score        * 0.20 +
            critical_score      * 0.20 +
            self_reliance       * 0.15 +
            metacognition       * 0.10 +
            (sem.delegation_penalty * -20)  # penalty for heavy delegation
        )

        self.composite = max(0.0, min(100.0, composite + llm_adjustment))
        return self.composite


def score_session(session: Session, max_tier: int = 2) -> SessionScore:
    """Score a single session up to the specified analysis tier."""
    result = SessionScore(session=session)

    # Tier 1: always run
    result.structural = extract_structural(session)

    # Tier 2: embeddings (local, free)
    if max_tier >= 2:
        result.semantic = extract_semantic(session)

    # Tier 3: LLM classification (optional, cheap)
    if max_tier >= 3:
        result.llm = extract_llm(session)

    result.compute_composite()
    return result


def score_sessions(sessions: list[Session], max_tier: int = 2) -> list[SessionScore]:
    """Score a list of sessions."""
    results = []
    for session in sessions:
        results.append(score_session(session, max_tier))
    return results
