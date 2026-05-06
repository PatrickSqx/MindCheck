"""Quick end-to-end test for the trajectory feature."""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from mindcheck.parser import auto_discover_sessions
from mindcheck.scorer import score_sessions
from mindcheck.trajectory import (
    compute_trajectory, analyze_trend, save_trajectory_snapshot,
    load_trajectory_history, sparkline, _get_session_date,
)
from mindcheck.report import print_trajectory


def main():
    sessions = auto_discover_sessions(window="365d")
    print(f"Discovered {len(sessions)} sessions")

    results = score_sessions(sessions, max_tier=2)
    print(f"Scored {len(results)} sessions")

    # Check timestamps
    dated = [r for r in results if _get_session_date(r) is not None]
    print(f"Sessions with timestamps: {len(dated)}/{len(results)}")

    dates = sorted([_get_session_date(r) for r in results if _get_session_date(r)])
    if dates:
        d0 = dates[0].strftime("%Y-%m-%d")
        d1 = dates[-1].strftime("%Y-%m-%d")
        span = (dates[-1] - dates[0]).days
        print(f"Date range: {d0} to {d1}  ({span} days)")

    # Compute trajectory
    points = compute_trajectory(results)
    print(f"\nTrajectory: {len(points)} periods")
    for p in points:
        print(f"  {p.period_label}: {p.session_count} sessions, "
              f"score={p.avg_composite:.1f}, hyp={p.avg_hypothesis:.1f}")

    trend = analyze_trend(points)
    print(f"\nDirection: {trend.direction}")
    print(f"Slope: {trend.composite_slope:.2f}")
    print(f"Summary: {trend.summary}")

    # Save snapshot
    if points:
        save_trajectory_snapshot(points)
        loaded = load_trajectory_history()
        print(f"\nPersisted {len(points)} points, loaded back {len(loaded)} points")

    # Sparkline
    if points:
        scores = [p.avg_composite for p in points]
        print(f"Sparkline: {sparkline(scores)}")

    # Rich display
    print("\n--- Rich display ---\n")
    period_name = "month" if points and "-W" not in points[0].period_label else "week"
    print_trajectory(points, trend, period_name)


if __name__ == "__main__":
    main()
