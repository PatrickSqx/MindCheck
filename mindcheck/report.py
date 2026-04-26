"""
Report generator — turns scored sessions into human-readable output.
"""

from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from mindcheck.scorer import SessionScore

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
        "| Session | Tool | T1 | T2 | Hypothesis | Ownership | Critical |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in sorted(results, key=lambda x: x.composite, reverse=True):
        s = r.session
        sem = r.semantic
        tag = " [archived]" if s.archived else ""
        lines.append(
            f"| {s.id[:40]}{tag} | {s.tool} | {r.tier1_score:.0f} | {r.composite:.0f} "
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

    table.add_row("Composite score (T2)", f"{result.composite:.0f}/100", "semantic signals")
    table.add_row("Structural score (T1)", f"{result.tier1_score:.0f}/100", "structure only")

    # Show Tier 3 status if it ran
    llm = result.llm
    if llm.ran:
        t3_note = (
            f"reclassified {llm.messages_reclassified} message(s) → "
            f"hypothesis now {llm.hypothesis_level_avg:.2f}/4"
            if llm.messages_reclassified > 0
            else "ran — all messages already confident"
        )
        table.add_row("Tier 3 (LLM)", "[green]active[/green]", t3_note)
    elif result.session.tool:  # only show if a session was actually scored
        table.add_row("Tier 3 (LLM)", "[dim]not triggered[/dim]",
                      "all messages were high-confidence")
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
