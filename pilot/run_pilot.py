"""Run the pilot: every setting x {feasible, impossible} x episodes.

Usage:
  python -m pilot.run_pilot --out logs/run1                 # real model (see pilot/agent.py)
  python -m pilot.run_pilot --out logs/mock --mock follower # scripted agents, no GPU

Each episode is written to OUT/<setting>__<feasibility>__seed<k>.jsonl when it
finishes. Episodes whose file already exists are skipped, so a run can be resumed.
An episode in which an agent's model requests kept failing is saved as .failed
and retried on the next run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .analyze import report
from .episode import DEFAULT_SETTINGS, SETTINGS, Condition, Episode
from .mock_llm import POLICIES, MockClient


async def _run(args: argparse.Namespace) -> None:
    if args.mock:
        client = MockClient(args.mock)
    else:
        from .agent import OpenAIClient

        client = OpenAIClient()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(args.concurrency)

    async def one(condition: Condition, seed: int) -> str:
        path = out / f"{condition.name}__seed{seed:03d}.jsonl"
        if path.exists():
            return "skipped"
        async with limit:
            events = await Episode(condition, seed, client, args.max_turns).run()
        text = "".join(json.dumps(event) + "\n" for event in events)
        if events[-1]["failed_agents"]:
            # Keep the log for inspection, but not as a finished episode, so
            # the next run retries it and the analysis ignores it.
            path.with_suffix(".failed").write_text(text)
            print(f"FAILED {path.name}: model requests failed", flush=True)
            return "failed"
        partial = path.with_suffix(".partial")
        partial.write_text(text)
        partial.rename(path)
        path.with_suffix(".failed").unlink(missing_ok=True)  # a retried episode succeeded
        print(f"done {path.name}", flush=True)
        return "run"

    jobs = [
        one(Condition(setting, feasible), seed)
        for setting in args.settings
        for feasible in (True, False)
        for seed in range(args.episodes)
    ]
    results = await asyncio.gather(*jobs)
    print(
        f"\n{results.count('run')} episodes run, {results.count('skipped')} already present, "
        f"{results.count('failed')} failed.\n"
    )
    print(report(out))
    if "failed" in results:
        raise SystemExit("Some episodes failed because the model endpoint did not respond.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="directory for episode logs")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per cell (default 10)")
    parser.add_argument(
        "--settings", nargs="+", default=list(DEFAULT_SETTINGS), choices=list(SETTINGS),
        help=f"settings to run (default: {' '.join(DEFAULT_SETTINGS)})",
    )
    parser.add_argument("--max-turns", type=int, default=20, help="turns per agent (default 20)")
    parser.add_argument("--concurrency", type=int, default=8, help="episodes run at once")
    parser.add_argument("--mock", choices=POLICIES, help="use a scripted policy, not a model")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
