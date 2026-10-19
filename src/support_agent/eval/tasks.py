"""τ³-bench banking tasks: the customer's scenario, the documents a correct answer needs, and one
reference trajectory of tool calls (``evaluation_criteria.actions``)."""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field
from functools import lru_cache

from ..kb.store import TOOL_NAME, data_dir

BOILERPLATE = re.compile(
    r"(You are (?:currently )?(?:playing the role of|on the line with)[^.]*\.|Never respond as[^.]*\.|"
    r"Only provide additional details[^.]*\.|Don'?t dump all[^.]*\.|You are playing the role of the customer\.|"
    r"Do NOT volunteer this information[^.]*\.|ONLY MENTION THIS[^.]*\.)", re.I)


@dataclass
class Action:
    name: str
    arguments: dict
    requestor: str  # assistant | user

    @property
    def tool(self) -> str | None:
        """The discoverable tool an action unlocks, gives or calls (None for base tools)."""
        for key in ("agent_tool_name", "discoverable_tool_name", "tool_name"):
            if isinstance(self.arguments.get(key), str):
                return self.arguments[key]
        return self.name if TOOL_NAME.fullmatch(self.name) else None


@dataclass
class Task:
    id: str
    instructions: str
    required_documents: list[str]
    actions: list[Action]
    reward_basis: list[str]
    initial_state: dict | None = None
    user_tools: list[str] = field(default_factory=list)

    @property
    def split(self) -> str:
        return "dev" if zlib.crc32(self.id.encode()) % 2 == 0 else "test"

    @property
    def gold_tools(self) -> list[str]:
        return list(dict.fromkeys(a.tool for a in self.actions if a.tool))

    def opening(self) -> str:
        """What the customer says first: a scripted opening line if the scenario has one, else the situation."""
        text = self.instructions
        m = re.search(r"(?:Start naturally|Opening|start (?:the conversation )?(?:by|with)[^:\"]*)[^\"\n]*[:\-]?\s*\"([^\"]{20,})\"",
                      text, re.I)
        if m:
            return m.group(1)
        m = re.search(r"\*\*Your situation:?\*\*:?\s*(.+?)(?:\n\n|$)", text, re.S)
        if m:
            return BOILERPLATE.sub("", m.group(1)).strip()
        return self.scenario()[:600]

    def scenario(self) -> str:
        """The scenario without role-play instructions."""
        text = re.sub(r"(?m)^#+ .*$", "", self.instructions)
        return re.sub(r"\s+", " ", BOILERPLATE.sub("", text)).strip()


def _actions(raw: list[dict]) -> list[Action]:
    out = []
    for a in raw or []:
        args = a.get("arguments") or {}
        if isinstance(args.get("arguments"), str):
            try:
                args = {**args, "arguments": json.loads(args["arguments"])}
            except json.JSONDecodeError:
                pass
        out.append(Action(a["name"], args, a.get("requestor", "assistant")))
    return out


@lru_cache(maxsize=1)
def load_tasks() -> tuple[Task, ...]:
    tasks = []
    for f in sorted((data_dir() / "tasks").glob("task_*.json")):
        raw = json.loads(f.read_text(encoding="utf-8"))
        ec = raw.get("evaluation_criteria") or {}
        tasks.append(Task(raw["id"], raw["user_scenario"]["instructions"], raw.get("required_documents") or [],
                          _actions(ec.get("actions")), ec.get("reward_basis") or [], raw.get("initial_state"),
                          raw.get("user_tools") or []))
    return tuple(tasks)


def task_by_id(task_id: str) -> Task:
    for t in load_tasks():
        if t.id == task_id:
            return t
    raise KeyError(task_id)
