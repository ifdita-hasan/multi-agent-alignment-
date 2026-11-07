"""One episode: identical agents take turns in a shared environment.

`Episode.run()` returns the event log as a list of dicts. Every event has a
`seq` number (its position in the log) and a `type`:

  episode_start   condition, seed, agents, tasks, turn order
  board_post      author, text, scripted (True for the seeded post)
  board_delivery  agent, post_idxs shown to it this turn
  model_call      agent, round, content, reasoning, n_tool_calls, finish_reason
  model_error     agent, round, error
  tool            agent, round, call_seq, name, args, error, path, path_class, result
  submit          agent, round, answer, notes, verdict
  no_submission   agent (ran out of turns)
  debrief         agent, content, reasoning
  episode_end     failed_agents (agents whose model requests kept failing)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts
from .agent import Reply, ToolCall, assistant_message, scrub, tools_for
from .env import Board, Task, VirtualFS, build_world, classify_path, normalize, score

# setting name -> (number of model-driven agents, whether the board starts with the scripted post)
SETTINGS: dict[str, tuple[int, bool]] = {
    "solo": (1, False),
    "group2": (2, False),
    "group2_seed": (2, True),
    "group4": (4, False),
    "group4_seed": (4, True),
    "group8": (8, False),  # larger swarm; opt in with --settings
    "group8_seed": (8, True),
}
DEFAULT_SETTINGS = ("solo", "group2", "group2_seed", "group4", "group4_seed")

_REQUIRED_ARGS = {
    "list_dir": "path",
    "read_file": "path",
    "post_message": "text",
    "report_to_overseer": "message",
    "submit": "answer",
}
_MODEL_CALL_ATTEMPTS = 3


@dataclass(frozen=True)
class Condition:
    setting: str
    feasible: bool

    @property
    def n_agents(self) -> int:
        return SETTINGS[self.setting][0]

    @property
    def seeded(self) -> bool:
        return SETTINGS[self.setting][1]

    @property
    def name(self) -> str:
        return f"{self.setting}__{'feasible' if self.feasible else 'impossible'}"


@dataclass
class _Agent:
    task: Task
    messages: list[dict]
    board_cursor: int = 0
    done: bool = False
    failed: bool = False

    @property
    def id(self) -> str:
        return self.task.agent_id

    def add_user(self, text: str) -> None:
        """Append a user message, merging with a preceding user message."""
        text = scrub(text)  # board posts by other agents may carry special tokens
        if self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] += "\n\n" + text
        else:
            self.messages.append({"role": "user", "content": text})


@dataclass
class Episode:
    condition: Condition
    seed: int
    client: object  # anything with `async chat(messages, tools) -> Reply`
    max_turns: int = 20
    events: list[dict] = field(default_factory=list)

    def _log(self, type_: str, **data: object) -> int:
        seq = len(self.events)
        self.events.append({"seq": seq, "type": type_, **data})
        return seq

    async def run(self) -> list[dict]:
        cond = self.condition
        agent_ids = [f"agent_{i + 1}" for i in range(cond.n_agents)]
        scripted_peer = f"agent_{cond.n_agents + 1}" if cond.seeded else None
        self.fs: VirtualFS
        self.fs, tasks = build_world(
            agent_ids, cond.feasible, self.seed, [scripted_peer] if scripted_peer else None
        )
        self.board = Board()
        self.has_board = cond.n_agents > 1
        self.tools = tools_for(self.has_board)
        told_agents = cond.n_agents + (1 if cond.seeded else 0)

        order = list(agent_ids)
        random.Random(f"order-{self.seed}").shuffle(order)
        agents = [
            _Agent(
                tasks[a],
                [
                    {"role": "system", "content": prompts.system_prompt(tasks[a], told_agents)},
                    {"role": "user", "content": prompts.task_prompt(tasks[a])},
                ],
            )
            for a in order
        ]

        self._log(
            "episode_start",
            setting=cond.setting,
            feasible=cond.feasible,
            seed=self.seed,
            max_turns=self.max_turns,
            order=order,
            scripted_peer=scripted_peer,
            tasks={a: {"account": t.account, "answer": t.answer} for a, t in tasks.items()},
            system_prompt_example=agents[0].messages[0]["content"],
        )
        if scripted_peer:
            self.board.post(scripted_peer, prompts.SEED_POST)
            self._log("board_post", author=scripted_peer, text=prompts.SEED_POST, scripted=True)

        for round_ in range(1, self.max_turns + 1):
            active = [a for a in agents if not a.done]
            if not active:
                break
            for agent in active:
                await self._turn(agent, round_)

        for agent in agents:
            if not agent.done:
                self._log("no_submission", agent=agent.id)
        for agent in agents:
            if not agent.failed:
                await self._debrief(agent)
        self._log("episode_end", failed_agents=[a.id for a in agents if a.failed])
        return self.events

    async def _call_model(self, agent: _Agent, round_: int, tools: list[dict] | None) -> Reply | None:
        for _ in range(_MODEL_CALL_ATTEMPTS):
            try:
                return await self.client.chat(agent.messages, tools)
            except Exception as error:  # network or server failure: log and retry
                self._log("model_error", agent=agent.id, round=round_, error=repr(error))
        agent.done = agent.failed = True
        return None

    async def _turn(self, agent: _Agent, round_: int) -> None:
        unread = self.board.unread(agent.id, agent.board_cursor)
        agent.board_cursor = len(self.board.messages)
        if unread:
            agent.add_user(prompts.board_update(unread))
            self._log("board_delivery", agent=agent.id, post_idxs=[m.idx for m in unread])

        reply = await self._call_model(agent, round_, self.tools)
        if reply is None:
            return
        call_seq = self._log(
            "model_call",
            agent=agent.id,
            round=round_,
            content=reply.content,
            reasoning=reply.reasoning,
            n_tool_calls=len(reply.tool_calls),
            finish_reason=reply.finish_reason,
        )
        agent.messages.append(assistant_message(reply))
        if not reply.tool_calls:
            agent.add_user(prompts.NUDGE)
            return
        for call in reply.tool_calls:
            if agent.done:
                result = "Not executed: you have already submitted."
            else:
                result = self._run_tool(agent, call, call_seq, round_)
            agent.messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": scrub(result)}
            )

    def _run_tool(self, agent: _Agent, call: ToolCall, call_seq: int, round_: int) -> str:
        args = call.arguments
        event: dict = {"agent": agent.id, "round": round_, "call_seq": call_seq, "name": call.name}
        required = _REQUIRED_ARGS.get(call.name)

        if required is None or (call.name == "post_message" and not self.has_board):
            error, result = "unknown_tool", f"Error: unknown tool {call.name!r}."
        elif args is None:
            error, result = "bad_arguments", "Error: arguments must be a JSON object."
        elif required not in args:
            error, result = "bad_arguments", f"Error: missing argument {required!r}."
        else:
            error, result = None, self._execute(agent, call.name, args, event, round_)

        self._log(
            "tool",
            **event,
            args=args if args is not None else call.raw_arguments,
            error=error,
            result=result,
        )
        return result

    def _execute(self, agent: _Agent, name: str, args: dict, event: dict, round_: int) -> str:
        if name in ("list_dir", "read_file"):
            path = normalize(args["path"])
            event["path"] = path
            event["path_class"] = classify_path(agent.id, path)
            if name == "list_dir":
                entries = self.fs.list_dir(path)
                found = entries is not None
                result = "\n".join(entries) if found else f"Error: no such directory: {path}"
            else:
                content = self.fs.read(path)
                found = content is not None
                result = content if found else f"Error: no such file: {path}"
            event["found"] = found
            return result
        if name == "post_message":
            text = str(args["text"])
            self.board.post(agent.id, text)
            self._log("board_post", author=agent.id, text=text, scripted=False)
            return "Posted to the shared board."
        if name == "report_to_overseer":
            return "Your message was sent to the overseer."
        # submit
        agent.done = True
        self._log(
            "submit",
            agent=agent.id,
            round=round_,
            answer=args["answer"],
            notes=args.get("notes"),
            verdict=score(agent.task, args["answer"]),
        )
        return "Submission recorded."

    async def _debrief(self, agent: _Agent) -> None:
        agent.add_user(prompts.DEBRIEF)
        reply = await self._call_model(agent, 0, None)
        if reply is not None:
            self._log("debrief", agent=agent.id, content=reply.content, reasoning=reply.reasoning)
