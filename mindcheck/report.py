"""
Report generator — turns scored sessions into human-readable output.
"""

from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from mindcheck.scorer import SessionScore
from mindcheck.trajectory import (
    TrajectoryPoint, TrendAnalysis, compute_trajectory, analyze_trend, sparkline,
)

console = Console()

_TASK_LABELS = {
    "code":     "Code / debugging",
    "data":     "Data / analysis",
    "writing":  "Writing / editing",
    "research": "Research / concepts",
    "planning": "Planning / design",
    "config":   "Config / setup",
}

_TASK_LABELS_SHORT = {
    "code": "code", "data": "data", "writing": "writing",
    "research": "research", "planning": "planning", "config": "config",
}


def _personalized_band(score: float, task_breakdown: dict, sem=None,
                       is_rich: bool = True) -> str:
    """Generate a personalized score interpretation based on actual patterns."""
    # Score band label
    if score >= 70:
        label = ("Strong engagement", "green") if is_rich else ("Strong engagement", "")
    elif score >= 50:
        label = ("Moderate engagement", "yellow") if is_rich else ("Moderate engagement", "")
    elif score >= 30:
        label = ("Passive engagement", "yellow") if is_rich else ("Passive engagement", "")
    else:
        label = ("Heavy delegation", "red") if is_rich else ("Heavy delegation", "")

    # Categorize domains into strengths and weaknesses
    strengths = []
    weaknesses = []
    if task_breakdown:
        for domain, stats in task_breakdown.items():
            if stats["count"] < 3:
                continue
            name = _TASK_LABELS_SHORT.get(domain, domain)
            if stats["delegation_rate"] >= 0.5 or stats["hypothesis_avg"] < 1.5:
                weaknesses.append(name)
            elif stats["hypothesis_avg"] >= 2.0 and stats["delegation_rate"] < 0.4:
                strengths.append(name)

    # Build personalized message
    if is_rich:
        band_text = f"[{label[1]}]{label[0]}[/{label[1]}]"
    else:
        band_text = label[0]

    if strengths and weaknesses:
        s = " and ".join(strengths)
        w = " and ".join(weaknesses)
        if len(weaknesses) >= 3:
            detail = f"engaged on {s}, but delegating heavily across {w}."
        elif len(weaknesses) == 2:
            detail = f"engaged on {s}, but {w} need more independent thinking."
        else:
            detail = f"engaged on {s}, but leaning on AI too much for {w}."
    elif weaknesses:
        w = " and ".join(weaknesses)
        detail = f"heavy delegation across {w} -- try forming hypotheses before asking."
    elif strengths:
        s = " and ".join(strengths)
        detail = f"strong thinking on {s} -- keep pushing deeper."
    elif sem and sem.critical_engagement < 0.3:
        detail = "low critical engagement -- try questioning AI outputs more before accepting."
    elif sem and sem.hypothesis_level_avg < 1.5:
        detail = "try forming a hypothesis before asking -- even a rough guess helps."
    else:
        # Fallback to generic
        if score >= 70:
            detail = "you're driving, hypothesising, and thinking critically."
        elif score >= 50:
            detail = "solid in places, but room to push deeper before asking."
        elif score >= 30:
            detail = "leaning on AI for direction more than thinking it through first."
        else:
            detail = "most asks hand off the thinking entirely."

    return f"{band_text} -- {detail}"


def generate_report(results: list[SessionScore], period: str = "") -> str:
    """Generate a markdown report from scored sessions."""
    if not results:
        return "# MindCheck Report\n\nNo sessions to analyse.\n"

    period = period or datetime.now().strftime("%B %Y")
    avg_score = sum(r.composite for r in results) / len(results)

    best = max(results, key=lambda r: r.composite)
    worst = min(results, key=lambda r: r.composite)

    # Build aggregated task breakdown for personalized band
    agg_for_band: dict[str, dict] = {}
    for r in results:
        for domain, stats in r.semantic.task_breakdown.items():
            if domain not in agg_for_band:
                agg_for_band[domain] = {"count": 0, "hyp_sum": 0.0, "deleg_sum": 0.0}
            agg_for_band[domain]["count"]    += stats["count"]
            agg_for_band[domain]["hyp_sum"]  += stats["hypothesis_avg"] * stats["count"]
            agg_for_band[domain]["deleg_sum"] += stats["delegation_rate"] * stats["count"]
    band_breakdown = {}
    for domain, d in agg_for_band.items():
        band_breakdown[domain] = {
            "count": d["count"],
            "hypothesis_avg": d["hyp_sum"] / d["count"],
            "delegation_rate": d["deleg_sum"] / d["count"],
        }
    band_text = _personalized_band(avg_score, band_breakdown, is_rich=False)

    lines = [
        f"# MindCheck Report — {period}",
        "",
        f"**Cognitive Engagement Score: {avg_score:.0f}/100**",
        f"Sessions analysed: {len(results)}",
        f"*{band_text}*",
        "",
        "---",
        "",
        "## Session breakdown",
        "",
        "| Session | Tool | Type | T1 | T2 | Hypothesis | Ownership | Critical |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in sorted(results, key=lambda x: x.composite, reverse=True):
        s = r.session
        sem = r.semantic
        tag = " [archived]" if s.archived else ""
        lines.append(
            f"| {s.id[:40]}{tag} | {s.tool} | {r.session_type} "
            f"| {r.tier1_score:.0f} | {r.composite:.0f} "
            f"| {sem.hypothesis_level_avg:.1f}/4 "
            f"| {sem.ownership_score*100:.0f}% "
            f"| {sem.critical_engagement*100:.0f}% |"
        )

    # ── Task pattern aggregation across all sessions ──────────────────────────
    agg: dict[str, dict] = {}
    for r in results:
        for domain, stats in r.semantic.task_breakdown.items():
            if domain not in agg:
                agg[domain] = {"count": 0, "hyp_sum": 0.0, "deleg_sum": 0.0}
            agg[domain]["count"]    += stats["count"]
            agg[domain]["hyp_sum"]  += stats["hypothesis_avg"] * stats["count"]
            agg[domain]["deleg_sum"] += stats["delegation_rate"] * stats["count"]

    if agg:
        lines += ["", "---", "", "## Task patterns", "",
                  "| Task type | Messages | Avg hypothesis | Delegation |  |",
                  "|---|---|---|---|---|"]
        for domain, d in sorted(agg.items(), key=lambda x: x[1]["count"], reverse=True):
            label = _TASK_LABELS.get(domain, domain)
            hyp   = d["hyp_sum"]  / d["count"]
            deleg = d["deleg_sum"] / d["count"]
            flag  = "<< watch this" if deleg >= 0.5 or hyp < 1.5 else ""
            lines.append(
                f"| {label} | {d['count']} | {hyp:.1f}/4 | {deleg*100:.0f}% | {flag} |"
            )

    # ── Trajectory section (when sessions span multiple periods) ────────────
    traj_points = compute_trajectory(results)
    if len(traj_points) >= 2:
        traj_trend = analyze_trend(traj_points)
        period_name = "month" if "-W" not in traj_points[0].period_label else "week"

        dir_arrows = {"improving": "↑", "declining": "↓", "stable": "→"}
        dir_arrow = dir_arrows.get(traj_trend.direction, "")

        lines += ["", "---", "",
                  f"## Engagement trajectory {dir_arrow}", ""]

        lines.append(f"*{traj_trend.summary}*")
        lines.append("")

        lines.append(
            "| Period | Sessions | Score | Hypothesis | Ownership | Critical |"
        )
        lines.append("|---|---|---|---|---|---|")

        for p in traj_points:
            lines.append(
                f"| {p.period_label} | {p.session_count} "
                f"| {p.avg_composite:.0f} | {p.avg_hypothesis:.1f}/4 "
                f"| {p.avg_ownership*100:.0f}% | {p.avg_critical*100:.0f}% |"
            )

        if traj_trend.best_period and traj_trend.worst_period:
            lines += [
                "",
                f"**Best period:** {traj_trend.best_period.period_label} "
                f"({traj_trend.best_period.avg_composite:.0f}/100)  ",
                f"**Worst period:** {traj_trend.worst_period.period_label} "
                f"({traj_trend.worst_period.avg_composite:.0f}/100)",
            ]

    # ── Subtext / authenticity section ──────────────────────────────────────
    sessions_with_subtext = [r for r in results
                             if (r.subtext.contradictions_found > 0 or
                                 r.subtext.performative_count > 0 or
                                 r.subtext.say_then_contradict_candidates > 0 or
                                 r.subtext.passive_acceptance_streak >= 3)]
    if sessions_with_subtext:
        avg_auth = sum(r.subtext.authenticity_score for r in results) / len(results)
        total_contradictions = sum(r.subtext.contradictions_found for r in results)
        total_performative = sum(r.subtext.performative_count for r in results)

        lines += ["", "---", "", "## Subtext analysis", ""]
        lines.append(f"**Average authenticity: {avg_auth*100:.0f}%**  ")
        lines.append(f"Sessions with subtext patterns: {len(sessions_with_subtext)}/{len(results)}")
        total_stc_candidates = sum(r.subtext.say_then_contradict_candidates for r in results)
        if total_contradictions > 0:
            lines.append(f"Say-then-contradict (confirmed): {total_contradictions}  ")
        elif total_stc_candidates > 0:
            lines.append(f"Say-then-contradict candidates: {total_stc_candidates} (needs Tier 3)  ")
        if total_performative > 0:
            lines.append(f"Performative messages: {total_performative}  ")

    lines += [
        "",
        "---",
        "",
        "## Highlights",
        "",
        f"**Strongest session:** {best.session.id}  ",
        f"Score: {best.composite:.0f} - "
        f"Hypothesis: {best.semantic.hypothesis_level_avg:.1f}/4",
        "",
        f"**Weakest session:** {worst.session.id}  ",
        f"Score: {worst.composite:.0f} - "
        f"Hypothesis: {worst.semantic.hypothesis_level_avg:.1f}/4",
        "",
        "---",
        "",
        "_Generated by MindCheck — https://github.com/PatrickSqx/MindCheck_",
    ]

    return "\n".join(lines)


def print_report(report_text: str):
    """Print a markdown report to the terminal with rich formatting."""
    from rich.markdown import Markdown
    console.print(Markdown(report_text))


def print_session_score(result: SessionScore):
    """Print a single session's score breakdown."""
    s = result.session
    sem = result.semantic
    st = result.structural

    archived_tag = " [dim][archived][/dim]" if s.archived else ""
    table = Table(title=f"Session: {s.id}{archived_tag}", box=box.SIMPLE)
    table.add_column("Signal", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Notes", style="dim")

    table.add_row("Session type", f"{result.session_type}", "auto-detected")

    # Show T2 vs T3 scores when Tier 3 ran
    llm = result.llm
    if llm.ran and llm.messages_reclassified > 0:
        delta = result.composite - llm.t2_composite
        delta_str = f"+{delta:.0f}" if delta >= 0 else f"{delta:.0f}"
        table.add_row("Composite score (T2)", f"{llm.t2_composite:.0f}/100", "before LLM refinement")
        table.add_row("Composite score (T3)", f"{result.composite:.0f}/100",
                      f"after refinement ({delta_str} pts, {llm.messages_reclassified} msg reclassified)")
    else:
        table.add_row("Composite score (T2)", f"{result.composite:.0f}/100", "semantic signals")
    table.add_row("Structural score (T1)", f"{result.tier1_score:.0f}/100", "structure only")

    if llm.ran and llm.messages_reclassified == 0:
        table.add_row("Tier 3 (LLM)", "[dim]ran[/dim]",
                      "all messages already high-confidence")
    table.add_row("Hypothesis level",    f"{sem.hypothesis_level_avg:.1f}/4",
                  "0=no attempt, 4=tested hypothesis")
    table.add_row("Ownership",            f"{sem.ownership_score*100:.0f}%",
                  "% of turns where you drove direction")
    table.add_row("Critical engagement", f"{sem.critical_engagement*100:.0f}%",
                  "pushed back / caught mistakes")
    table.add_row("Self-reliance",       f"{sem.self_reliance*100:.0f}%",
                  "showed prior attempt before asking")
    table.add_row("Metacognition",       f"{sem.metacognition_score*100:.0f}%",
                  "reflected on your own approach")
    _dp = sem.delegation_penalty * 20
    _dp_note = (
        "heavy outsourcing detected" if _dp >= 10 else
        "some outsourcing detected"  if _dp >= 3  else
        "minimal outsourcing"
    )
    table.add_row("Delegation penalty",  f"-{_dp:.1f}pts", _dp_note)
    table.add_row("Question ratio",      f"{st.question_ratio*100:.0f}%",
                  "of your messages contained a question")
    table.add_row("Turn count",          str(st.turn_count), "exchanges in session")
    table.add_row("Message ratio",       f"{st.message_ratio:.2f}",
                  "your words / AI words (higher = more engaged)")

    # Subtext / authenticity signals
    sub = result.subtext
    if sub.contradictions_found > 0 or sub.performative_count > 0 or sub.passive_acceptance_streak >= 3 or sub.say_then_contradict_candidates > 0 or sub.llm_ran:
        table.add_row("", "", "")  # spacer
        auth_pct = sub.authenticity_score * 100
        if auth_pct >= 80:
            auth_style = "green"
        elif auth_pct >= 60:
            auth_style = "yellow"
        else:
            auth_style = "red"
        table.add_row("Authenticity",
                      f"[{auth_style}]{auth_pct:.0f}%[/{auth_style}]",
                      "how genuine the engagement appears")
        if sub.say_then_contradict > 0:
            table.add_row("  Say-then-contradict",
                          str(sub.say_then_contradict),
                          "LLM-confirmed contradictions")
        elif sub.say_then_contradict_candidates > 0:
            table.add_row("  STC candidates",
                          str(sub.say_then_contradict_candidates),
                          "[dim]needs Tier 3 to confirm[/dim]")
        if sub.empty_self_reliance > 0:
            table.add_row("  Empty self-reliance",
                          str(sub.empty_self_reliance),
                          "claimed effort without specifics")
        if sub.hypothesis_without_followup > 0:
            table.add_row("  Dropped hypotheses",
                          str(sub.hypothesis_without_followup),
                          "formed hypothesis, never tested it")
        if sub.passive_acceptance_streak >= 3:
            table.add_row("  Passive streak",
                          f"{sub.passive_acceptance_streak} msgs",
                          "consecutive uncritical acceptance")
        if sub.llm_ran:
            if sub.performative_hypothesis > 0:
                table.add_row("  Performative hypothesis",
                              str(sub.performative_hypothesis),
                              "LLM-detected fake hypothesis")
            if sub.fake_curiosity > 0:
                table.add_row("  Fake curiosity",
                              str(sub.fake_curiosity),
                              "LLM-detected surface-level questions")

    console.print(table)

    # Task domain breakdown
    breakdown = sem.task_breakdown
    if breakdown:
        task_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        task_table.add_column("Task type",      style="cyan",  min_width=20)
        task_table.add_column("Messages",       style="white", justify="right")
        task_table.add_column("Avg hypothesis", style="white", justify="right")
        task_table.add_column("Delegation",     style="white", justify="right")
        task_table.add_column("",               style="dim")

        for domain, stats in sorted(breakdown.items(),
                                    key=lambda x: x[1]["count"], reverse=True):
            label    = _TASK_LABELS.get(domain, domain)
            hyp      = stats["hypothesis_avg"]
            deleg    = stats["delegation_rate"]
            count    = stats["count"]
            flag = "[yellow]<< watch this[/yellow]" if deleg >= 0.5 or hyp < 1.5 else ""
            task_table.add_row(
                label, str(count),
                f"{hyp:.1f}/4",
                f"{deleg*100:.0f}%",
                flag,
            )

        console.print("  [bold]Task breakdown[/bold]")
        console.print(task_table)

    # Personalized interpretation
    console.print(f"\n  {_personalized_band(result.composite, sem.task_breakdown, sem)}")
    console.print(
        "  [dim]Hypothesis guide: 0 = dump the problem | 1 = describe symptom | "
        "2 = locate cause | 3 = form hypothesis | 4 = tested a hypothesis[/dim]\n"
    )


def print_trajectory(points: list[TrajectoryPoint], trend: TrendAnalysis,
                     period_name: str = "week"):
    """Print trajectory analysis to the terminal with rich formatting."""
    if not points:
        console.print("[yellow]No trajectory data available.[/yellow]")
        return

    # Direction label with colour
    dir_colours = {
        "improving": "green", "declining": "red", "stable": "yellow",
        "insufficient_data": "dim",
    }
    dir_colour = dir_colours.get(trend.direction, "white")
    dir_arrows = {
        "improving": "↑", "declining": "↓", "stable": "→",
        "insufficient_data": "—",
    }
    dir_arrow = dir_arrows.get(trend.direction, "")

    console.print(Panel(
        f"[bold]Cognitive Engagement Trajectory[/bold]  "
        f"[{dir_colour}]{dir_arrow} {trend.direction.capitalize()}[/{dir_colour}]"
    ))

    # Main trajectory table
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("Period",     style="cyan",  min_width=10)
    table.add_column("Sessions",   style="white", justify="right", min_width=8)
    table.add_column("Score",      style="white", justify="right", min_width=6)
    table.add_column("Hypothesis", style="white", justify="right", min_width=10)
    table.add_column("Ownership",  style="white", justify="right", min_width=9)
    table.add_column("Critical",   style="white", justify="right", min_width=8)
    table.add_column("Self-rel.",  style="white", justify="right", min_width=8)
    table.add_column("Meta.",      style="white", justify="right", min_width=6)
    table.add_column("",           style="dim",   min_width=3)

    prev_score = None
    for p in points:
        # Trend arrow vs previous period
        if prev_score is not None:
            delta = p.avg_composite - prev_score
            if delta > 2:
                arrow = "[green]↑[/green]"
            elif delta < -2:
                arrow = "[red]↓[/red]"
            else:
                arrow = "[dim]→[/dim]"
        else:
            arrow = ""
        prev_score = p.avg_composite

        # Colour the score
        if p.avg_composite >= 70:
            score_str = f"[green]{p.avg_composite:.0f}[/green]"
        elif p.avg_composite >= 50:
            score_str = f"[yellow]{p.avg_composite:.0f}[/yellow]"
        elif p.avg_composite >= 30:
            score_str = f"[yellow]{p.avg_composite:.0f}[/yellow]"
        else:
            score_str = f"[red]{p.avg_composite:.0f}[/red]"

        table.add_row(
            p.period_label,
            str(p.session_count),
            score_str,
            f"{p.avg_hypothesis:.1f}/4",
            f"{p.avg_ownership*100:.0f}%",
            f"{p.avg_critical*100:.0f}%",
            f"{p.avg_self_reliance*100:.0f}%",
            f"{p.avg_metacognition*100:.0f}%",
            arrow,
        )

    console.print(table)

    # Sparkline
    scores = [p.avg_composite for p in points]
    spark = sparkline(scores)
    console.print(f"  Score trend: [bold]{spark}[/bold]  ", end="")
    if trend.direction == "improving":
        console.print(f"[green]+{trend.composite_slope:.1f} pts/{period_name}[/green]")
    elif trend.direction == "declining":
        console.print(f"[red]{trend.composite_slope:.1f} pts/{period_name}[/red]")
    else:
        console.print(f"[dim]stable[/dim]")

    # Per-signal trend highlights
    improving = [s for s, sl in trend.signal_trends.items()
                 if sl > 1.0 and s != "delegation"]
    declining = [s for s, sl in trend.signal_trends.items()
                 if sl < -1.0 and s != "delegation"]
    deleg_sl = trend.signal_trends.get("delegation", 0)
    if deleg_sl > 1.0:
        declining.append("delegation")
    elif deleg_sl < -1.0:
        improving.append("delegation")

    if improving:
        console.print(f"  [green]Improving:[/green] {', '.join(improving)}")
    if declining:
        console.print(f"  [red]Watch:[/red] {', '.join(declining)}")

    # Best / worst
    if trend.best_period and trend.worst_period and trend.period_count >= 2:
        console.print(
            f"  Best: [green]{trend.best_period.period_label}[/green] "
            f"({trend.best_period.avg_composite:.0f})  |  "
            f"Worst: [red]{trend.worst_period.period_label}[/red] "
            f"({trend.worst_period.avg_composite:.0f})"
        )

    console.print(f"\n  [dim]{trend.summary}[/dim]\n")
