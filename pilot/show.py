"""Print one or more episode logs in reading order.

Usage: python -m pilot.show LOG.jsonl [LOG2.jsonl ...] [--full]

Reasoning and replies are printed in full. Tool results (file contents) are cut
to one line unless --full is given.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .analyze import load_episode


def _one_line(text: object, limit: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def render(events: list[dict], full: bool = False) -> str:
    lines: list[str] = []
    for e in events:
        t = e["type"]
        if t == "episode_start":
            kind = "feasible" if e["feasible"] else "impossible"
            lines.append(f"# {e['setting']} {kind} seed={e['seed']} order={' '.join(e['order'])}")
            for agent, task in e["tasks"].items():
                lines.append(f"  {agent}: account {task['account']}, answer {task['answer']}")
        elif t == "board_post":
            tag = "scripted" if e["scripted"] else "post"
            lines.append(f"\n## BOARD ({tag}) {e['author']}: {e['text']}")
        elif t == "board_delivery":
            lines.append(f"  [board -> {e['agent']}: posts {e['post_idxs']}]")
        elif t == "model_call":
            finish = e.get("finish_reason") or "?"
            lines.append(
                f"\n[{e['agent']} round {e['round']} finish={finish} tool_calls={e['n_tool_calls']}]"
            )
            if e["reasoning"]:
                lines.append(f"  <reasoning> {e['reasoning']}")
            if e["content"]:
                lines.append(f"  <content> {e['content']}")
        elif t == "model_error":
            lines.append(f"\n[{e['agent']} round {e['round']} MODEL ERROR] {e['error']}")
        elif t == "tool":
            path = f" [{e['path_class']}]" if e.get("path_class") else ""
            error = f" ERROR={e['error']}" if e["error"] else ""
            result = e["result"] if full else _one_line(e["result"])
            lines.append(f"  -> {e['name']}({json.dumps(e['args'])}){path}{error} => {result}")
        elif t == "submit":
            notes = f" ({e['notes']})" if e.get("notes") else ""
            lines.append(f"  == SUBMIT {e['agent']}: {e['answer']!r} -> {e['verdict']}{notes}")
        elif t == "no_submission":
            lines.append(f"  == {e['agent']}: no submission")
        elif t == "debrief":
            lines.append(f"\n[{e['agent']} debrief]")
            if e["reasoning"]:
                lines.append(f"  <reasoning> {e['reasoning']}")
            lines.append(f"  <content> {e['content']}")
        elif t == "episode_end" and e["failed_agents"]:
            lines.append(f"\nfailed agents: {' '.join(e['failed_agents'])}")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    full = "--full" in argv
    paths = [a for a in argv if a != "--full"]
    if not paths:
        sys.exit(__doc__)
    for path in paths:
        print(render(load_episode(Path(path)), full))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
