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
def analyze(path: str, output: str, tier: int):
    """Analyse sessions in PATH and generate a report."""
    from mindcheck.parser import discover_sessions
    from mindcheck.scorer import score_sessions
    from mindcheck.report import generate_report

    console.print(Panel(f"[bold]MindCheck[/bold] — analysing [cyan]{path}[/cyan]"))

    sessions = discover_sessions(Path(path))
    if not sessions:
        console.print("[red]No sessions found.[/red]")
        return

    console.print(f"Found [cyan]{len(sessions)}[/cyan] sessions")

    results = score_sessions(sessions, max_tier=tier)
    report = generate_report(results)

    Path(output).write_text(report, encoding="utf-8")
    console.print(f"[green]Report saved to {output}[/green]")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--tier", default=2, type=click.IntRange(1, 3))
def score(file: str, tier: int):
    """Score a single session FILE."""
    from mindcheck.parser import parse_session
    from mindcheck.scorer import score_session
    from mindcheck.report import print_session_score

    session = parse_session(Path(file))
    if not session:
        console.print("[red]Could not parse session.[/red]")
        return

    result = score_session(session, max_tier=tier)
    print_session_score(result)


@main.command()
@click.option("--last", default="30d", help="Time window e.g. 7d, 30d, 90d")
@click.option("--tier", default=2, type=click.IntRange(1, 3))
def report(last: str, tier: int):
    """Generate a report from auto-discovered sessions."""
    from mindcheck.parser import auto_discover_sessions
    from mindcheck.scorer import score_sessions
    from mindcheck.report import generate_report, print_report

    sessions = auto_discover_sessions(window=last)
    if not sessions:
        console.print("[yellow]No sessions found in known directories.[/yellow]")
        console.print("Try: mindcheck analyze <path>")
        return

    results = score_sessions(sessions, max_tier=tier)
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
