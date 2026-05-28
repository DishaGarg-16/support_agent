"""CLI entry point for the support triage agent."""

from __future__ import annotations

from pathlib import Path
import sys
import time

from agent import SupportTriageAgent


def print_banner() -> None:
    print("SUPA")
    print("Support triage pipeline")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "support_tickets" / "support_tickets.csv"
    output_path = repo_root / "support_tickets" / "output.csv"

    start = time.perf_counter()
    print_banner()
    print(f"Reading tickets from {input_path.relative_to(repo_root).as_posix()}")

    agent = SupportTriageAgent(repo_root)
    agent.process_csv(input_path, output_path)

    elapsed = time.perf_counter() - start
    print(f"Wrote {output_path.relative_to(repo_root).as_posix()}")
    print(f"Completed in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

