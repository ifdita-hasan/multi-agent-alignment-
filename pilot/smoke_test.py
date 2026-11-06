"""Check that the model endpoint does what the pilot needs.

Usage: python -m pilot.smoke_test     (endpoint settings: see pilot/agent.py)

Exits non-zero if a required check fails.
"""
from __future__ import annotations

import asyncio
import sys

from .agent import OpenAIClient, assistant_message, tools_for


async def main() -> int:
    client = OpenAIClient()
    messages = [
        {"role": "system", "content": "You are a test agent in a file environment."},
        {"role": "user", "content": "List the directory /work using the list_dir tool."},
    ]
    reply = await client.chat(messages, tools_for(has_board=True))
    call = reply.tool_calls[0] if reply.tool_calls else None
    checks = [
        ("returns a tool call", call is not None, True),
        ("calls list_dir", call is not None and call.name == "list_dir", True),
        (
            "arguments parse with a path",
            call is not None and call.arguments is not None and "path" in call.arguments,
            True,
        ),
        ("returns reasoning text", bool(reply.reasoning), False),
    ]

    if call is not None:
        messages.append(assistant_message(reply))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": "agent_1/\nagent_2/"})
        messages.append({"role": "user", "content": "In one sentence, what did you find?"})
        # the debrief is sent without tools, after a history that contains tool calls
        final = await client.chat(messages, None)
        checks.append(("answers in text when no tools are offered", bool(final.content), True))

    failed = False
    for name, ok, required in checks:
        status = "PASS" if ok else ("FAIL" if required else "WARN")
        failed |= required and not ok
        print(f"{status}  {name}")
    if not reply.reasoning:
        print(
            "No reasoning came back. The pilot still runs, but the reasoning traces "
            "will be empty; check the server's reasoning-parser setting."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
