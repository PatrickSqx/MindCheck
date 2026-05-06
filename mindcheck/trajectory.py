"""
Cross-session learning trajectory — tracks how cognitive engagement
changes over time.

Groups scored sessions by time period (week / month), computes
per-period averages, and analyses trends to show whether the user
is improving, declining, or stable.

Trajectory snapshots are persisted to ~/.mindcheck/trajectory.json
so historical data survives even if old session files are deleted.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from mindcheck.scorer import SessionScore


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    """Aggregate stats for one time period."""
    period_label: str               # e.g. "2024-W12" or "2024-03"
    period_start: datetime
    session_count: int = 0
    avg_composite: float = 0.0
    avg_hypothesis: float = 0.0
    avg_ownership: float = 0.0
    avg_critical: float = 0.0
    avg_self_reliance: float = 0.0
    avg_metacognition: float = 0.0
    avg_delegation: float = 0.0
    type_counts: dict = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Overall and per-signal trend analysis."""
    direction: str = "stable"         # "improving" | "declining" | "stable"
    composite_slope: float = 0.0      # points per period
    period_count: int = 0
    signal_trends: dict = field(default_factory=dict)  # signal → slope
    best_period: Optional[TrajectoryPoint] = None
    worst_period: Optional[TrajectoryPoint] = None
    summary: str = ""


# ── Public API ───────────────────────────────────────────────────────────────

def compute_trajectory(
    results: list[SessionScore],
    period: str = "auto",
) -> list[TrajectoryPoint]:
    """Group scored sessions by time period and compute per-period averages.

    Args:
        results: Scored sessions to group.
        period:  "week", "month", or "auto" (auto picks month if span > 12 weeks).

    Returns:
        List of TrajectoryPoints sorted chronologically.
    """
    # Resolve "auto" period
    if period == "auto":
        dates = [_get_session_date(r) for r in results]
        dates = [d for d in dates if d is not None]
        if len(dates) >= 2:
            span_days = (max(dates) - min(dates)).days
            period = "month" if span_days > 84 else "week"   # 12 weeks
        else:
            period = "week"

    # Bucket sessions by period
    buckets: dict[str, list[SessionScore]] = defaultdict(list)
    bucket_starts: dict[str, datetime] = {}

    for r in results:
        dt = _get_session_date(r)
        if dt is None:
            continue

        if period == "week":
            iso = dt.isocalendar()
            label = f"{iso[0]}-W{iso[1]:02d}"
            period_start = datetime.fromisocalendar(iso[0], iso[1], 1)
        else:
            label = dt.strftime("%Y-%m")
            period_start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        buckets[label].append(r)
        if label not in bucket_starts:
            bucket_starts[label] = period_start

    # Compute per-period averages
    points = []
    for label in sorted(buckets.keys()):
        sessions = buckets[label]
        n = len(sessions)

        type_counts: dict[str, int] = defaultdict(int)
        for r in sessions:
            type_counts[r.session_type] += 1

        points.append(TrajectoryPoint(
            period_label=label,
            period_start=bucket_starts[label],
            session_count=n,
            avg_composite=sum(r.composite for r in sessions) / n,
            avg_hypothesis=sum(r.semantic.hypothesis_level_avg for r in sessions) / n,
            avg_ownership=sum(r.semantic.ownership_score for r in sessions) / n,
            avg_critical=sum(r.semantic.critical_engagement for r in sessions) / n,
            avg_self_reliance=sum(r.semantic.self_reliance for r in sessions) / n,
            avg_metacognition=sum(r.semantic.metacognition_score for r in sessions) / n,
            avg_delegation=sum(r.semantic.delegation_penalty for r in sessions) / n,
            type_counts=dict(type_counts),
        ))

    return points


def analyze_trend(points: list[TrajectoryPoint]) -> TrendAnalysis:
    """Analyse the trend across trajectory points using linear regression.

    Returns direction (improving / declining / stable), slopes per
    signal, best and worst periods, and a human-readable summary.
    """
    analysis = TrendAnalysis(period_count=len(points))

    if len(points) < 2:
        analysis.direction = "insufficient_data"
        analysis.summary = "Need at least 2 time periods to detect a trend."
        if points:
            analysis.best_period = points[0]
            analysis.worst_period = points[0]
        return analysis

    # Best / worst
    analysis.best_period = max(points, key=lambda p: p.avg_composite)
    analysis.worst_period = min(points, key=lambda p: p.avg_composite)

    # Simple linear regression: y = mx + b
    n = len(points)
    xs = list(range(n))

    def _slope(ys: list[float]) -> float:
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        return num / den if den else 0.0

    # Composite trend
    analysis.composite_slope = _slope([p.avg_composite for p in points])

    # Per-signal trends (all on 0–100 scale for comparability)
    signal_extractors = {
        "hypothesis":    lambda p: p.avg_hypothesis / 4 * 100,
        "ownership":     lambda p: p.avg_ownership * 100,
        "critical":      lambda p: p.avg_critical * 100,
        "self_reliance": lambda p: p.avg_self_reliance * 100,
        "metacognition": lambda p: p.avg_metacognition * 100,
        "delegation":    lambda p: p.avg_delegation * 100,
    }
    for name, extract in signal_extractors.items():
        analysis.signal_trends[name] = _slope([extract(p) for p in points])

    # Direction classification
    if analysis.composite_slope > 1.5:
        analysis.direction = "improving"
    elif analysis.composite_slope < -1.5:
        analysis.direction = "declining"
    else:
        analysis.direction = "stable"

    # Identify improving / declining signals
    improving = [s for s, sl in analysis.signal_trends.items()
                 if sl > 1.0 and s != "delegation"]
    declining = [s for s, sl in analysis.signal_trends.items()
                 if sl < -1.0 and s != "delegation"]

    # Delegation is inverted — rising delegation is bad
    deleg_sl = analysis.signal_trends.get("delegation", 0)
    if deleg_sl > 1.0:
        declining.append("delegation (rising)")
    elif deleg_sl < -1.0:
        improving.append("delegation (falling)")

    # Summary text
    if analysis.direction == "improving":
        detail = f"Score trending up (+{analysis.composite_slope:.1f} pts/period)."
        if improving:
            detail += f" Strongest growth: {', '.join(improving[:3])}."
    elif analysis.direction == "declining":
        detail = f"Score trending down ({analysis.composite_slope:.1f} pts/period)."
        if declining:
            detail += f" Watch: {', '.join(declining[:3])}."
    else:
        detail = "Score is stable across periods."
        if improving:
            detail += f" Improving: {', '.join(improving[:3])}."
        if declining:
            detail += f" Declining: {', '.join(declining[:3])}."

    analysis.summary = detail
    return analysis


# ── Persistence ──────────────────────────────────────────────────────────────

def save_trajectory_snapshot(points: list[TrajectoryPoint]) -> None:
    """Persist trajectory data to ~/.mindcheck/trajectory.json.

    Merges with existing data — newer snapshots for the same period
    label overwrite older ones, so history accumulates over time.
    """
    path = _trajectory_path()

    existing: dict[str, dict] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing = {p["period_label"]: p for p in data.get("points", [])}
        except Exception:
            existing = {}

    for point in points:
        existing[point.period_label] = {
            "period_label":      point.period_label,
            "period_start":      point.period_start.isoformat(),
            "session_count":     point.session_count,
            "avg_composite":     round(point.avg_composite, 1),
            "avg_hypothesis":    round(point.avg_hypothesis, 2),
            "avg_ownership":     round(point.avg_ownership, 3),
            "avg_critical":      round(point.avg_critical, 3),
            "avg_self_reliance": round(point.avg_self_reliance, 3),
            "avg_metacognition": round(point.avg_metacognition, 3),
            "avg_delegation":    round(point.avg_delegation, 3),
            "type_counts":       point.type_counts,
        }

    sorted_points = sorted(existing.values(), key=lambda p: p["period_label"])

    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps({"points": sorted_points,
                     "updated_at": datetime.now().isoformat()},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_trajectory_history() -> list[TrajectoryPoint]:
    """Load persisted trajectory history from ~/.mindcheck/trajectory.json."""
    path = _trajectory_path()
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        points = []
        for d in data.get("points", []):
            points.append(TrajectoryPoint(
                period_label=d["period_label"],
                period_start=datetime.fromisoformat(d["period_start"]),
                session_count=d["session_count"],
                avg_composite=d["avg_composite"],
                avg_hypothesis=d["avg_hypothesis"],
                avg_ownership=d["avg_ownership"],
                avg_critical=d["avg_critical"],
                avg_self_reliance=d["avg_self_reliance"],
                avg_metacognition=d["avg_metacognition"],
                avg_delegation=d["avg_delegation"],
                type_counts=d.get("type_counts", {}),
            ))
        return points
    except Exception:
        return []


def clear_trajectory() -> bool:
    """Delete persisted trajectory data. Returns True if file existed."""
    path = _trajectory_path()
    if path.exists():
        path.unlink()
        return True
    return False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _trajectory_path() -> Path:
    return Path.home() / ".mindcheck" / "trajectory.json"


def _get_session_date(result: SessionScore) -> Optional[datetime]:
    """Best available date for a scored session."""
    s = result.session
    if s.created_at:
        # Ensure timezone-naive for consistent comparison
        dt = s.created_at
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    try:
        return datetime.fromtimestamp(s.file_path.stat().st_mtime)
    except OSError:
        return None


def sparkline(values: list[float]) -> str:
    """Generate a Unicode sparkline from a list of values."""
    if not values:
        return ""
    if len(values) == 1 or max(values) == min(values):
        return "▅" * len(values)  # ▅
    blocks = "▁▂▃▄▅▆▇█"  # ▁▂▃▄▅▆▇█
    lo, hi = min(values), max(values)
    return "".join(
        blocks[int((v - lo) / (hi - lo) * (len(blocks) - 1))]
        for v in values
    )
