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

    def compute_composite(self, max_tier: int = 2):
        """
        Composite score (0–100) weighted across signal categories.

        Weights are redistributed based on which tiers ran, so Tier 1-only
        scores are still meaningful rather than collapsing to near-zero.

        Full weights (Tier 2+):
          - Hypothesis / question framing: 25%
          - Agency:                        20%
          - Critical engagement:           20%
          - Self-reliance:                 15%
          - Metacognition:                 10%
          - Structural ratios:              5%
          - Delegation penalty:           -20pts max

        Tier 1 only — structural signals carry 100% of weight:
          - Question ratio:    40%  (curiosity vs commands)
          - Turn count:        30%  (dialogue depth)
          - Message ratio:     20%  (how much the user writes)
          - Prior attempts:    10%  (showed effort before asking)
        """
        s = self.structural
        sem = self.semantic
        llm = self.llm

        # ── Tier 1: structural score (always computed) ──────────────────────
        structural_score = (
            min(s.question_ratio * 100, 100) * 0.40 +
            min(s.turn_count / 10 * 100, 100) * 0.30 +
            min(s.message_ratio * 200, 100)   * 0.20 +
            min(s.prior_attempt_count / max(s.turn_count, 1) * 100, 100) * 0.10
        )

        # ── Tier 1 only: rescale so structural carries full weight ──────────
        if max_tier == 1:
            # Apply delegation penalty even in Tier 1 (keyword-detectable)
            delegation_ratio = s.delegation_count / max(s.turn_count, 1)
            penalty = min(delegation_ratio * 30, 20)
            self.composite = max(0.0, min(100.0, structural_score - penalty))
            return self.composite

        # ── Tier 2+: full weighted composite ───────────────────────────────
        hypothesis_score = sem.hypothesis_level_avg / 4 * 100
        agency_score     = sem.agency_score * 100
        critical_score   = sem.critical_engagement * 100
        self_reliance    = sem.self_reliance * 100
        metacognition    = sem.metacognition_score * 100

        # LLM refinement (Tier 3, adjusts when available)
        llm_adjustment = llm.confidence_adjustment if llm.ran else 0.0

        composite = (
            structural_score * 0.05 +
            hypothesis_score * 0.25 +
            agency_score     * 0.20 +
            critical_score   * 0.20 +
            self_reliance    * 0.15 +
            metacognition    * 0.10 +
            (sem.delegation_penalty * -20)
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

    result.compute_composite(max_tier=max_tier)
    return result


def score_sessions(sessions: list[Session], max_tier: int = 2) -> list[SessionScore]:
    """Score a list of sessions."""
    results = []
    for session in sessions:
        results.append(score_session(session, max_tier))
    return results
