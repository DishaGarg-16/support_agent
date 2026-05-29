"""CLI entry point for the support triage agent."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pyfiglet import Figlet
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from agent import SupportTriageAgent
import llm_client

console = Console()

def print_banner() -> None:
    f = Figlet(font="slant")
    ascii_art = f.renderText("SUPA")
    panel = Panel(
        ascii_art.rstrip("\n"),
        title="[bold cyan]Support Triage Pipeline[/bold cyan]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)
    console.print()

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "support_tickets" / "support_tickets.csv"
    output_path = repo_root / "support_tickets" / "output.csv"

    print_banner()

    # Load environment variables (.env)
    load_dotenv(repo_root / ".env")

    start = time.perf_counter()

    with console.status("[bold green]Initializing Support Triage Agent...", spinner="dots"):
        agent = SupportTriageAgent(repo_root)

    # Show LLM mode status
    if llm_client.is_available():
        console.print(f"[bold green]*[/bold green] LLM mode: [green]Active[/green] (Groq / {llm_client.MODEL})")
    else:
        console.print("[bold yellow]*[/bold yellow] LLM mode: [yellow]Disabled[/yellow] (deterministic fallback)")

    console.print(f"[bold cyan]*[/bold cyan] Reading tickets from [yellow]{input_path.relative_to(repo_root).as_posix()}[/yellow]")
    
    total_tickets = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Processing tickets...", total=None)
        
        for processed, total in agent.process_csv(input_path, output_path):
            if progress.tasks[task].total is None:
                progress.update(task, total=total)
                total_tickets = total
            progress.update(task, completed=processed)

    elapsed = time.perf_counter() - start

    console.print(f"[bold green]Success:[/bold green] Wrote output to [yellow]{output_path.relative_to(repo_root).as_posix()}[/yellow]")
    console.print()

    # Summary table
    table = Table(title="[bold]Execution Summary[/bold]", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Value", justify="right")

    table.add_row("Total Tickets", str(total_tickets))
    table.add_row("Elapsed Time", f"{elapsed:.2f}s")
    if total_tickets > 0:
        table.add_row("Average Latency", f"{(elapsed / total_tickets):.2f}s / ticket")
    table.add_row("LLM Mode", "Active" if llm_client.is_available() else "Disabled")

    console.print(table)
    console.print()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
