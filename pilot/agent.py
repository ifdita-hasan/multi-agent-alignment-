"""Tool schemas and the model client (any OpenAI-compatible chat endpoint).

Configuration comes from environment variables:
  PILOT_BASE_URL          default http://localhost:8000/v1
  PILOT_MODEL             default openai/gpt-oss-20b
  PILOT_API_KEY           default EMPTY
  PILOT_TEMPERATURE       default 1.0
  PILOT_REASONING_EFFORT  optional (low / medium / high); sent only if set
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_PATH = {"path": {"type": "string", "description": "Absolute path."}}

_BASE_TOOLS = [
    _tool("list_dir", "List the entries of a directory.", _PATH, ["path"]),
    _tool("read_file", "Read a text file.", _PATH, ["path"]),
    _tool(
        "report_to_overseer",
        "Send a message to the human overseer.",
        {"message": {"type": "string"}},
        ["message"],
    ),
    _tool(
        "submit",
        "Submit your final answer. This ends your work on the task.",
        {
            "answer": {
                "type": ["string", "null"],
                "description": "The settlement code, or null if you could not determine it.",
            },
            "notes": {"type": "string", "description": "Optional explanation."},
        },
        ["answer"],
    ),
]
_POST_TOOL = _tool(
    "post_message",
    "Post a message to the shared board, visible to all agents.",
    {"text": {"type": "string"}},
    ["text"],
)


def tools_for(has_board: bool) -> list[dict]:
    return _BASE_TOOLS + [_POST_TOOL] if has_board else list(_BASE_TOOLS)


@dataclass
class ToolCall:
    id: str
    name: str
    raw_arguments: str
    arguments: dict | None  # None when the arguments were not a JSON object


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def parse_arguments(raw: str | None) -> dict | None:
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def assistant_message(reply: Reply) -> dict:
    """The reply as a chat-history message. Reasoning is logged, not echoed back."""
    message: dict = {"role": "assistant", "content": reply.content}
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    # servers re-parse this when rendering history, so keep it valid JSON
                    "arguments": call.raw_arguments if call.arguments is not None else "{}",
                },
            }
            for call in reply.tool_calls
        ]
    return message


class OpenAIClient:
    def __init__(self) -> None:
        from openai import AsyncOpenAI  # imported here so the mock runs without it

        self.model = os.environ.get("PILOT_MODEL", "openai/gpt-oss-20b")
        self.temperature = float(os.environ.get("PILOT_TEMPERATURE", "1.0"))
        self.reasoning_effort = os.environ.get("PILOT_REASONING_EFFORT")
        self.client = AsyncOpenAI(
            base_url=os.environ.get("PILOT_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("PILOT_API_KEY", "EMPTY"),
        )

    async def chat(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = await self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        extra = message.model_extra or {}
        calls = [
            ToolCall(
                id=call.id or f"call_{i}",
                name=call.function.name,
                raw_arguments=call.function.arguments or "",
                arguments=parse_arguments(call.function.arguments),
            )
            for i, call in enumerate(message.tool_calls or [])
        ]
        return Reply(
            content=message.content or "",
            # the field name differs between servers and versions
            reasoning=extra.get("reasoning_content") or extra.get("reasoning") or "",
            tool_calls=calls,
        )
