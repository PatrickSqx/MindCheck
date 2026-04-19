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


@main.command("config")
@click.option("--key",        default=None, help="API key (provider auto-detected from prefix)")
@click.option("--provider",   default=None,
              type=click.Choice(["anthropic", "openai", "ollama"]),
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
      mindcheck config --key sk-API_KEY --model gpt-4o
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
