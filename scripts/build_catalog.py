"""Regenerate ``data/rho_bank/tool_catalog.json`` from tau2-bench's banking tools (needs the tau2 extra).

The catalog lets the agent show tool signatures and the guard know which discoverable tools exist,
without importing tau2 at runtime.

    python scripts/build_catalog.py            # write
    python scripts/build_catalog.py --check    # exit 1 if the shipped catalog differs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src/support_agent/data/rho_bank/tool_catalog.json"


def build() -> dict:
    from tau2.domains.banking_knowledge.environment import get_environment
    from tau2.domains.banking_knowledge.tools import (
        KnowledgeTools,
        KnowledgeUserTools,
        parse_discoverable_tool_docstring,
    )
    from tau2.environment.toolkit import DISCOVERABLE_ATTR, TOOL_TYPE_ATTR

    def discoverable(cls) -> dict:
        out = {}
        for name in sorted(dir(cls)):
            fn = getattr(cls, name, None)
            if callable(fn) and getattr(fn, DISCOVERABLE_ATTR, False):
                info = parse_discoverable_tool_docstring(fn)
                out[name] = {"name": name, **info, "tool_type": str(getattr(fn, TOOL_TYPE_ATTR, ""))}
        return out

    env = get_environment(retrieval_variant="no_knowledge")
    return {
        "agent_discoverable": discoverable(KnowledgeTools),
        "user_discoverable": discoverable(KnowledgeUserTools),
        "base_agent_tools": [t.openai_schema for t in env.get_tools()],
        "base_user_tools": [t.openai_schema for t in env.get_user_tools()],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    cat = build()
    if a.check:
        shipped = json.loads(OUT.read_text(encoding="utf-8"))
        diff = [k for k in cat if json.dumps(cat[k], sort_keys=True) != json.dumps(shipped.get(k), sort_keys=True)]
        print("catalog up to date" if not diff else f"differs: {', '.join(diff)}")
        return 1 if diff else 0
    OUT.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(cat['agent_discoverable'])} agent tools, {len(cat['user_discoverable'])} user tools)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
