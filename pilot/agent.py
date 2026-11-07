"""Tool schemas and the model client (any OpenAI-compatible chat endpoint).

Configuration comes from environment variables:
  PILOT_BASE_URL          default http://localhost:8000/v1
  PILOT_MODEL             default openai/gpt-oss-20b
  PILOT_API_KEY           default EMPTY
  PILOT_TEMPERATURE       default 1.0
  PILOT_REASONING_EFFORT  optional (low / medium / high); sent only if set
  PILOT_MAX_TOKENS        optional cap on the tokens of one reply; sent only if set
  PILOT_ECHO_REASONING    "1": send the model's reasoning back as `reasoning` on its
                          own history messages (gpt-oss keeps its chain of thought
                          across tool calls that way). Default: logged, not echoed.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

_SPECIAL_TOKEN = re.compile(r"<\|[A-Za-z0-9_]+\|>")


def scrub(text: str) -> str:
    """Remove harmony-style special tokens such as <|end|> from text that goes
    back to the model. gpt-oss sometimes writes them into replies and board
    posts, and the server then rejects the whole request."""
    return _SPECIAL_TOKEN.sub("", text)


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
    finish_reason: str = ""  # "stop", "tool_calls", "length", ... as reported by the server


def parse_arguments(raw: str | None) -> dict | None:
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def assistant_message(reply: Reply) -> dict:
    """The reply as a chat-history message.

    Reasoning is logged, not echoed back, unless PILOT_ECHO_REASONING=1: then it
    is sent as `reasoning`, which vLLM renders into gpt-oss's analysis channel.
    """
    message: dict = {"role": "assistant", "content": scrub(reply.content)}
    if reply.reasoning and os.environ.get("PILOT_ECHO_REASONING") == "1":
        message["reasoning"] = scrub(reply.reasoning)
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": scrub(call.name),  # a garbled parse once put <|end|> in the name
                    # servers re-parse this when rendering history, so keep it valid JSON
                    "arguments": scrub(call.raw_arguments) if call.arguments is not None else "{}",
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
        max_tokens = os.environ.get("PILOT_MAX_TOKENS")
        self.max_tokens = int(max_tokens) if max_tokens else None
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
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            # in the JSON body directly, so it does not depend on the SDK version
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        raw = message.model_dump()  # includes fields the server adds, such as `reasoning`
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
            reasoning=raw.get("reasoning_content") or raw.get("reasoning") or "",
            tool_calls=calls,
            finish_reason=choice.finish_reason or "",
        )
