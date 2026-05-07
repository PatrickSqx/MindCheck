"""
CLI entry point — `mindcheck` command.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option()
def main():
    """MindCheck — analyse your AI conversation logs for cognitive engagement."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", "-o", default="report.md", help="Output file path")
@click.option("--tier", default=2, type=click.IntRange(1, 3),
              help="Max analysis tier (1=rules, 2=embeddings, 3=LLM)")
@click.option("--skip-archived", is_flag=True, help="Exclude archived sessions")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON (for piping to other tools)")
def analyze(path: str, output: str, tier: int, skip_archived: bool, as_json: bool):
    """Analyse sessions in PATH and generate a report."""
    import json
    from mindcheck.parser import discover_sessions
    from mindcheck.scorer import score_sessions
    from mindcheck.report import generate_report

    if not as_json:
        console.print(Panel(f"[bold]MindCheck[/bold] — analysing [cyan]{path}[/cyan]"))

    sessions = discover_sessions(Path(path), skip_archived=skip_archived)
    if not sessions:
        if as_json:
            click.echo(json.dumps({"error": "No sessions found", "sessions": []}, indent=2))
        else:
            console.print("[red]No sessions found.[/red]")
        return

    if not as_json:
        archived_count = sum(1 for s in sessions if s.archived)
        if archived_count:
            console.print(f"Found [cyan]{len(sessions)}[/cyan] sessions ([dim]{archived_count} archived[/dim])")
        else:
            console.print(f"Found [cyan]{len(sessions)}[/cyan] sessions")

    results = score_sessions(sessions, max_tier=tier)

    if as_json:
        avg_score = sum(r.composite for r in results) / len(results)
        json_output = {
            "path": str(path),
            "session_count": len(results),
            "average_score": round(avg_score, 1),
            "sessions": [r.to_dict() for r in results],
        }
        click.echo(json.dumps(json_output, indent=2))
    else:
        report = generate_report(results)
        Path(output).write_text(report, encoding="utf-8")
        console.print(f"[green]Report saved to {output}[/green]")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--tier", default=2, type=click.IntRange(1, 3))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON (for piping to other tools)")
def score(file: str, tier: int, as_json: bool):
    """Score a single session FILE."""
    import json
    from mindcheck.parser import parse_session
    from mindcheck.scorer import score_session
    from mindcheck.report import print_session_score

    session = parse_session(Path(file))
    if not session:
        if as_json:
            click.echo(json.dumps({"error": "Could not parse session"}, indent=2))
        else:
            console.print("[red]Could not parse session.[/red]")
        return

    result = score_session(session, max_tier=tier)
    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        print_session_score(result)


@main.command()
@click.option("--last", default="30d", help="Time window e.g. 7d, 30d, 90d")
@click.option("--tier", default=2, type=click.IntRange(1, 3))
@click.option("--skip-archived", is_flag=True, help="Exclude archived sessions")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON (for piping to other tools)")
def report(last: str, tier: int, skip_archived: bool, as_json: bool):
    """Generate a report from auto-discovered sessions."""
    import json
    from mindcheck.parser import auto_discover_sessions
    from mindcheck.scorer import score_sessions
    from mindcheck.report import generate_report, print_report

    sessions = auto_discover_sessions(window=last, skip_archived=skip_archived)
    if not sessions:
        if as_json:
            click.echo(json.dumps({"error": "No sessions found", "sessions": []}, indent=2))
        else:
            console.print("[yellow]No sessions found in known directories.[/yellow]")
            console.print("Try: mindcheck analyze <path>")
        return

    if not as_json:
        archived_count = sum(1 for s in sessions if s.archived)
        msg = f"Found [cyan]{len(sessions)}[/cyan] sessions across the last [cyan]{last}[/cyan]"
        if archived_count:
            msg += f" ([dim]{archived_count} archived[/dim])"
        console.print(msg)

    results = score_sessions(sessions, max_tier=tier)

    if as_json:
        avg_score = sum(r.composite for r in results) / len(results)
        output = {
            "period": last,
            "session_count": len(results),
            "average_score": round(avg_score, 1),
            "sessions": [r.to_dict() for r in results],
        }
        click.echo(json.dumps(output, indent=2))
    else:
        report_text = generate_report(results)
        print_report(report_text)


@main.command("import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--format", "fmt", default="auto",
              type=click.Choice(["auto", "claude_chat", "chatgpt"]),
              help="Export format (auto-detected by default)")
@click.option("--tier", default=2, type=click.IntRange(1, 3))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def import_cmd(file: str, fmt: str, tier: int, as_json: bool):
    """Score conversations from a Claude Chat or ChatGPT export file."""
    import json as json_mod
    from mindcheck.parser import parse_export
    from mindcheck.scorer import score_sessions
    from mindcheck.report import generate_report, print_report, print_session_score

    if not as_json:
        console.print(Panel(f"[bold]MindCheck[/bold] -- importing [cyan]{file}[/cyan]"))

    sessions = parse_export(Path(file), format=fmt)
    if not sessions:
        if as_json:
            click.echo(json_mod.dumps({"error": "No conversations found"}, indent=2))
        else:
            console.print("[red]No conversations found.[/red]")
            console.print("[dim]Supported formats: Claude Chat export, ChatGPT export[/dim]")
        return

    # Filter: require at least 3 meaningful user messages
    sessions = [s for s in sessions
                if sum(1 for m in s.user_messages if len(m.content.strip()) >= 12) >= 3]

    if not sessions:
        if as_json:
            click.echo(json_mod.dumps({"error": "No conversations with enough messages"}, indent=2))
        else:
            console.print("[yellow]No conversations with enough messages to score.[/yellow]")
        return

    if not as_json:
        console.print(f"Found [cyan]{len(sessions)}[/cyan] conversations to score")

    results = score_sessions(sessions, max_tier=tier)

    if as_json:
        avg_score = sum(r.composite for r in results) / len(results)
        output = {
            "source": str(file),
            "format": fmt if fmt != "auto" else results[0].session.tool,
            "session_count": len(results),
            "average_score": round(avg_score, 1),
            "sessions": [r.to_dict() for r in results],
        }
        click.echo(json_mod.dumps(output, indent=2))
    elif len(results) == 1:
        print_session_score(results[0])
    else:
        report_text = generate_report(results)
        print_report(report_text)


@main.command()
def scan():
    """Show all auto-discovered session directories on this machine."""
    from mindcheck.parser import get_known_directories

    dirs = get_known_directories()
    console.print(Panel("[bold]Known session directories[/bold]"))
    for tool, paths in dirs.items():
        for p in paths:
            exists = p.exists()
            status = "[green]found[/green]" if exists else "[dim]not found[/dim]"
            console.print(f"  {tool:<15} {status}  {p}")


@main.command("config")
@click.option("--key",        default=None, help="API key (provider auto-detected from prefix)")
@click.option("--provider",   default=None,
              type=click.Choice(["anthropic", "openai", "gemini", "ollama"]),
              help="Set provider explicitly")
@click.option("--model",      default=None, help="Model to use (see --show for options)")
@click.option("--ollama-url", default=None, help="Ollama base URL (default: http://localhost:11434)")
@click.option("--show",       is_flag=True, help="Show current config and available models")
def config_cmd(key, provider, model, ollama_url, show):
    """Configure Tier 3 LLM provider, API key, and model.

    \b
    Examples:
      mindcheck config --key sk-ant-API_KEY                   Anthropic (auto-detected)
      mindcheck config --key sk-ant-API_KEY --model claude-haiku-4-5
      mindcheck config --key sk-API_KEY                       OpenAI (auto-detected)
      mindcheck config --key sk-API_KEY --model gpt-4.1-nano
      mindcheck config --key AIza...                          Gemini (auto-detected)
      mindcheck config --key AIza... --model gemini-2.5-flash
      mindcheck config --provider ollama                      Local Ollama (free)
      mindcheck config --provider ollama --model mistral
      mindcheck config --show                                 Show config + model list
    """
    from mindcheck.config import (load_config, save_config, set_key, set_provider,
                                   masked_key, is_tier3_configured, AVAILABLE_MODELS,
                                   _DEFAULT_MODELS)

    changed = False

    if key:
        cfg = set_key(key)
        console.print(f"[green]✓ Key saved.[/green] Provider: [cyan]{cfg['tier3_provider']}[/cyan]")
        changed = True

    if provider:
        cfg = set_provider(provider, url=ollama_url, model=model)
        console.print(f"[green]✓ Provider set to:[/green] [cyan]{provider}[/cyan]")
        changed = True

    if model:
        cfg = load_config()
        cfg["tier3_model"] = model
        save_config(cfg)
        console.print(f"[green]✓ Model set to:[/green] [cyan]{model}[/cyan]")
        changed = True

    if ollama_url and not provider:
        cfg = load_config()
        cfg["tier3_ollama_url"] = ollama_url
        save_config(cfg)
        console.print(f"[green]✓ Ollama URL set to:[/green] [cyan]{ollama_url}[/cyan]")
        changed = True

    # Always show status after a change, or when --show / no args
    if show or not changed:
        cfg = load_config()
        current_provider = cfg.get("tier3_provider")
        current_model    = cfg.get("tier3_model")
        active_model     = current_model or _DEFAULT_MODELS.get(current_provider, "—")

        console.print(Panel("[bold]Tier 3 configuration[/bold]"))
        console.print(f"  Provider : [cyan]{current_provider or '[dim]not set'}[/cyan]")
        console.print(f"  Key      : {masked_key(cfg)}")
        console.print(f"  Model    : [cyan]{active_model}[/cyan]"
                      + (" [dim](default)[/dim]" if not current_model else ""))
        console.print(f"  Status   : "
                      f"{'[green]ready — run with --tier 3[/green]' if is_tier3_configured(cfg) else '[yellow]not configured[/yellow]'}")

        # Show learned prototype count
        from pathlib import Path as _Path
        import json as _json
        learned_path = _Path.home() / ".mindcheck" / "learned_prototypes.json"
        learned_count = 0
        if learned_path.exists():
            try:
                learned_count = len(_json.loads(learned_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        console.print(f"  Learned  : [cyan]{learned_count}[/cyan] prototype(s) accumulated"
                      + (" [dim](grows as Tier 3 runs)[/dim]" if learned_count == 0 else ""))

        # Show available models for current or all providers
        providers_to_show = [current_provider] if current_provider else list(AVAILABLE_MODELS)
        console.print()
        for p in providers_to_show:
            if p not in AVAILABLE_MODELS:
                continue
            console.print(f"  [bold]{p.capitalize()} models:[/bold]")
            for m_name, m_desc in AVAILABLE_MODELS[p]:
                marker = " ◀" if m_name == active_model else ""
                console.print(f"    [cyan]{m_name:<30}[/cyan] {m_desc}{marker}")
        console.print()
        console.print("  [dim]Set model: mindcheck config --model <name>[/dim]")


@main.command()
@click.option("--last", default="90d", help="Time window e.g. 30d, 90d, 365d")
@click.option("--period", default="auto",
              type=click.Choice(["auto", "week", "month"]),
              help="Group by week or month (auto picks based on window)")
@click.option("--tier", default=2, type=click.IntRange(1, 3))
@click.option("--skip-archived", is_flag=True, help="Exclude archived sessions")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def trajectory(last: str, period: str, tier: int, skip_archived: bool, as_json: bool):
    """Show how your cognitive engagement changes over time."""
    import json
    from mindcheck.parser import auto_discover_sessions
    from mindcheck.scorer import score_sessions
    from mindcheck.report import print_trajectory
    from mindcheck.trajectory import (
        compute_trajectory, analyze_trend, save_trajectory_snapshot,
    )

    sessions = auto_discover_sessions(window=last, skip_archived=skip_archived)
    if not sessions:
        if as_json:
            click.echo(json.dumps({"error": "No sessions found"}, indent=2))
        else:
            console.print("[yellow]No sessions found in known directories.[/yellow]")
            console.print("Try: mindcheck analyze <path>")
        return

    if not as_json:
        console.print(f"Found [cyan]{len(sessions)}[/cyan] sessions across the last [cyan]{last}[/cyan]")

    results = score_sessions(sessions, max_tier=tier)

    points = compute_trajectory(results, period=period)
    trend = analyze_trend(points)

    # Persist snapshot so historical data accumulates
    if points:
        save_trajectory_snapshot(points)

    period_name = "month" if points and "-W" not in points[0].period_label else "week"

    if as_json:
        output = {
            "period": last,
            "grouping": period_name,
            "direction": trend.direction,
            "composite_slope": round(trend.composite_slope, 2),
            "summary": trend.summary,
            "signal_trends": {k: round(v, 2) for k, v in trend.signal_trends.items()},
            "points": [
                {
                    "period": p.period_label,
                    "sessions": p.session_count,
                    "score": round(p.avg_composite, 1),
                    "hypothesis": round(p.avg_hypothesis, 2),
                    "ownership": round(p.avg_ownership, 3),
                    "critical": round(p.avg_critical, 3),
                    "self_reliance": round(p.avg_self_reliance, 3),
                    "metacognition": round(p.avg_metacognition, 3),
                    "delegation": round(p.avg_delegation, 3),
                    "types": p.type_counts,
                }
                for p in points
            ],
        }
        if trend.best_period:
            output["best_period"] = trend.best_period.period_label
        if trend.worst_period:
            output["worst_period"] = trend.worst_period.period_label
        click.echo(json.dumps(output, indent=2))
    else:
        print_trajectory(points, trend, period_name)


@main.command()
@click.option("--clear", is_flag=True, help="Delete all cached scores")
def cache(clear: bool):
    """Show cache stats or clear the cache."""
    from mindcheck.cache import cache_stats, clear_cache, get_cache_path

    if clear:
        n = clear_cache()
        console.print(f"[green]Cleared {n} cached session(s).[/green]")
        return

    stats = cache_stats()
    path = get_cache_path()
    size_kb = stats["size_bytes"] / 1024

    console.print(Panel("[bold]MindCheck cache[/bold]"))
    console.print(f"  Location : {path}")
    console.print(f"  Size     : {size_kb:.1f} KB")
    console.print(f"  Sessions : {stats['total']} cached")
    for tier, count in sorted(stats["by_tier"].items()):
        console.print(f"    Tier {tier} : {count} session(s)")


@main.group()
def validate():
    """Validate subtext detection accuracy against ground truth."""
    pass


@validate.command("sample")
@click.option("--window", default="90d", help="Time window e.g. 7d, 30d, 90d")
@click.option("--max-per-session", default=5, type=int,
              help="Max flags to sample per session")
@click.option("--max-total", default=50, type=int,
              help="Max total new entries to add")
@click.option("--tool", multiple=True, help="Filter by tool (claude, codex, etc.)")
@click.option("--skip-archived", is_flag=True, default=True)
def validate_sample(window, max_per_session, max_total, tool, skip_archived):
    """Sample flagged message pairs from real sessions into ground truth."""
    from mindcheck.validate import sample as do_sample, load_ground_truth

    console.print(f"Scanning sessions from the last [cyan]{window}[/cyan]...")
    tools_list = list(tool) if tool else None
    new_count = do_sample(
        window=window,
        max_per_session=max_per_session,
        max_total=max_total,
        skip_archived=skip_archived,
        tools=tools_list,
    )

    existing = load_ground_truth()
    total = len(existing)
    unlabeled = sum(1 for e in existing if e["label"] == "UNLABELED")
    console.print(f"[green]Added {new_count} new entries.[/green]")
    console.print(f"Ground truth: {total} total, {unlabeled} unlabeled")
    console.print("[dim]Next: mindcheck validate label[/dim]")


@validate.command("label")
@click.option("--batch", default=10, type=int,
              help="Number of entries to label per session")
def validate_label(batch):
    """Interactively label flagged pairs as CORRECT or FALSE_POSITIVE."""
    from mindcheck.validate import label_interactive
    label_interactive(batch_size=batch)


@validate.command("measure")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def validate_measure(as_json):
    """Measure subtext detection accuracy against ground truth."""
    import json as json_mod
    from mindcheck.validate import measure as do_measure

    result = do_measure()
    if result is None:
        console.print("[yellow]No labeled data found. Run `mindcheck validate label` first.[/yellow]")
        return

    if as_json:
        click.echo(json_mod.dumps({
            "total_labeled": result.total_labeled,
            "accuracy": round(result.accuracy, 3),
            "fp_rate": round(result.fp_rate, 3),
            "true_positives": result.true_positives,
            "false_positives": result.false_positives,
            "fixed_false_positives": result.fixed_false_positives,
            "per_flag_type": result.per_flag_type,
        }, indent=2))
        return

    console.print()
    console.print(f"{'='*60}")
    console.print(f"  Subtext Detection Accuracy Report")
    console.print(f"{'='*60}")
    console.print(f"  Labeled entries:     {result.total_labeled}")
    console.print(f"    CORRECT:           {result.correct_labels}")
    console.print(f"    FALSE_POSITIVE:    {result.false_positive_labels}")
    console.print()
    console.print(f"  True positives:      {result.true_positives}")
    console.print(f"  False positives:     {result.false_positives}")
    console.print(f"  Fixed (no longer):   {result.fixed_false_positives}")
    acc_pct = f"{result.accuracy:.1%}"
    fp_pct = f"{result.fp_rate:.1%}"
    console.print(f"  Accuracy:            {acc_pct}")
    console.print(f"  FP rate:             {fp_pct}")
    console.print()
    console.print(f"  {'Flag Type':<30} {'TP':>4} {'FP':>4} {'Fixed':>6} {'Total':>6}")
    console.print(f"  {'-'*52}")
    for ft, counts in sorted(result.per_flag_type.items()):
        console.print(
            f"  {ft:<30} {counts['tp']:>4} {counts['fp']:>4} "
            f"{counts['fixed']:>6} {counts['total']:>6}"
        )
    console.print(f"{'='*60}")
