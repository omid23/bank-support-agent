"""The bank's knowledge base: 698 policy and product documents from τ³-bench (``banking_knowledge``).

Documents are short (median ~900 characters) and structured with ``##`` headings. Many describe a
procedure and name the internal tool that performs it (``file_credit_card_transaction_dispute_4829``);
the agent can only use such a tool after finding the document that names it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

TOOL_NAME = re.compile(r"\b([a-z][a-z_]*_\d{4})\b")


def data_dir() -> Path:
    return Path(str(resources.files("support_agent") / "data" / "rho_bank"))


@dataclass
class Section:
    doc_id: str
    heading: str
    text: str


@dataclass
class Document:
    id: str
    title: str
    content: str
    category: str = ""
    sections: list[Section] = field(default_factory=list)

    @property
    def tools(self) -> list[str]:
        """Discoverable tool names this document mentions, in order of appearance."""
        return list(dict.fromkeys(TOOL_NAME.findall(self.content)))

    def snippet(self, n: int = 600) -> str:
        return self.content[:n] + ("…" if len(self.content) > n else "")


def _sections(doc_id: str, content: str) -> list[Section]:
    parts = re.split(r"(?m)^(#{2,3} .+)$", content)
    out: list[Section] = []
    if parts[0].strip():
        out.append(Section(doc_id, "", parts[0].strip()))
    for i in range(1, len(parts) - 1, 2):
        out.append(Section(doc_id, parts[i].lstrip("# ").strip(), parts[i + 1].strip()))
    return out or [Section(doc_id, "", content)]


def _category(doc_id: str) -> str:
    m = re.match(r"doc_([a-z]+(?:_[a-z]+)?)_", doc_id)
    return m.group(1) if m else ""


class KnowledgeBase:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.by_id = {d.id: d for d in documents}

    def __len__(self) -> int:
        return len(self.documents)

    def get(self, doc_id: str) -> Document | None:
        return self.by_id.get(doc_id)

    def docs_mentioning(self, tool_name: str) -> list[str]:
        return [d.id for d in self.documents if tool_name in d.content]

    @classmethod
    def load(cls, path: str | Path | None = None) -> KnowledgeBase:
        root = Path(path) if path else data_dir() / "documents"
        docs = []
        for f in sorted(root.glob("*.json")):
            raw = json.loads(f.read_text(encoding="utf-8"))
            docs.append(Document(raw["id"], raw.get("title", ""), raw.get("content", ""), _category(raw["id"]),
                                 _sections(raw["id"], raw.get("content", ""))))
        return cls(docs)


@lru_cache(maxsize=1)
def default_kb() -> KnowledgeBase:
    return KnowledgeBase.load()


@lru_cache(maxsize=1)
def tool_catalog() -> dict:
    """Discoverable tool signatures, generated from tau2's tool docstrings (see scripts/build_catalog.py)."""
    return json.loads((data_dir() / "tool_catalog.json").read_text(encoding="utf-8"))


def policy_text() -> str:
    return (data_dir() / "policy.md").read_text(encoding="utf-8")
