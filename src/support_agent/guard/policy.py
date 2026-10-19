"""Pre-execution policy guard: every tool call the agent wants to make is checked before it runs.

The bank's policy (``data/rho_bank/policy.md``) and the knowledge base set rules a model breaks under
pressure. The guard enforces the ones that can be checked from state, deterministically and in
microseconds, so a voice agent can afford to run it on every call:

* ``known_tool``         the discoverable tool exists (models invent plausible names)
* ``documented_tool``    a document the agent actually read in this conversation names the tool
* ``unlock_first``       an agent tool is unlocked before it is called
* ``identity``           customer data is touched only after that customer's identity was verified
                         (``log_verification`` succeeded for that user_id)
* ``ownership``          every account, card and transaction in the arguments belongs to the verified customer
                         (or to a minor whose verified parent is acting for them)
* ``no_duplicate_give``  a user tool is given once (repeats write extra rows and break the audit trail)

Rules that need judgement (did the customer agree to a transfer?) go to an optional decision model.
A blocked call never reaches the bank: the agent gets the reason and how to fix it instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..kb.store import KnowledgeBase, default_kb, tool_catalog
from .ownership import OwnershipIndex

CUSTOMER_DATA_TOOLS = {"get_referrals_by_user", "get_credit_card_transactions_by_user",
                       "get_credit_card_accounts_by_user", "change_user_email"}
VERIFICATION_LOOKUPS = {"get_user_information_by_id", "get_user_information_by_name", "get_user_information_by_email"}
TOOL_ARG = {"unlock_discoverable_agent_tool": "agent_tool_name", "call_discoverable_agent_tool": "agent_tool_name",
            "give_discoverable_user_tool": "discoverable_tool_name"}
# Discoverable tools that need no customer context (product information, generic transfers).
NO_CUSTOMER = {"initial_transfer_to_human_agent_0218", "initial_transfer_to_human_agent_1822",
               "emergency_credit_bureau_incident_transfer_1114", "example_agent_tool_0000"}


@dataclass
class Verdict:
    allowed: bool
    rule: str = ""
    message: str = ""
    advisory: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "rule": self.rule, "message": self.message, "advisory": self.advisory}


@dataclass
class ConversationState:
    verified: set[str] = field(default_factory=set)
    read_docs: set[str] = field(default_factory=set)
    unlocked: set[str] = field(default_factory=set)
    given: set[str] = field(default_factory=set)
    user_turns: list[str] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"verified": sorted(self.verified), "read_docs": sorted(self.read_docs), "unlocked": sorted(self.unlocked),
                "given": sorted(self.given)}


def parse_args(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class PolicyGuard:
    def __init__(self, ownership: OwnershipIndex, kb: KnowledgeBase | None = None, decision_model=None,
                 rules: set[str] | None = None):
        self.ownership = ownership
        self.kb = kb or default_kb()
        cat = tool_catalog()
        self.agent_tools = set(cat["agent_discoverable"])
        self.user_tools = set(cat["user_discoverable"])
        self.decision_model = decision_model
        self.rules = rules  # None = all

    def _on(self, rule: str) -> bool:
        return self.rules is None or rule in self.rules

    def check(self, name: str, arguments: dict, state: ConversationState) -> Verdict:
        args = arguments or {}
        advisory: list[str] = []
        if name in TOOL_ARG:
            tool = str(args.get(TOOL_ARG[name], ""))
            known = self.user_tools if name == "give_discoverable_user_tool" else self.agent_tools
            if self._on("known_tool") and tool not in known:
                other = "a user tool" if tool in self.user_tools else "an agent tool" if tool in self.agent_tools else None
                return Verdict(False, "known_tool",
                               f"'{tool}' is not an available {'user' if name == 'give_discoverable_user_tool' else 'agent'} "
                               f"tool" + (f" (it is {other})" if other else "") + ". Use the exact name a knowledge-base "
                               "document gives.")
            if self._on("documented_tool") and not any(tool in self.kb.by_id[d].content for d in state.read_docs
                                                       if d in self.kb.by_id):
                return Verdict(False, "documented_tool",
                               f"No document you have read names '{tool}'. Search the knowledge base for the procedure "
                               "that describes it and follow that document.")
            if name == "call_discoverable_agent_tool" and self._on("unlock_first") and tool not in state.unlocked:
                return Verdict(False, "unlock_first", f"Unlock '{tool}' with unlock_discoverable_agent_tool before calling it.")
            if name == "give_discoverable_user_tool" and self._on("no_duplicate_give") and tool in state.given:
                return Verdict(False, "no_duplicate_give", f"'{tool}' was already given to the customer.")
            inner = parse_args(args.get("arguments"))
            if name in ("call_discoverable_agent_tool",) and tool not in NO_CUSTOMER:
                v = self._customer_scope(inner, state, advisory)
                if v is not None:
                    return v
        elif name in CUSTOMER_DATA_TOOLS:
            v = self._customer_scope(args, state, advisory)
            if v is not None:
                return v
        elif name == "transfer_to_human_agents" and self.decision_model is not None:
            agreed = self.decision_model.ask("Did the customer ask for, or agree to, a transfer to a human agent?",
                                             state.user_turns[-6:])
            if agreed is False:
                return Verdict(False, "transfer_consent", "Ask the customer whether they want to be transferred first.")
            if agreed is None:
                advisory.append("could not confirm the customer agreed to a transfer")
        return Verdict(True, advisory=advisory)

    def _customer_scope(self, args: dict, state: ConversationState, advisory: list[str] | None = None) -> Verdict | None:
        owners = self.ownership.owners_in(args)
        if not owners:
            return None
        for ref, owner in owners.items():
            if owner is None:
                # Not in the records the guard can see: possibly created earlier in this conversation (a new account
                # or card). The bank's own tools reject ids that do not exist, so this is noted, not blocked.
                if advisory is not None:
                    advisory.append(f"{ref} is not in the known records")
                continue
            if self._on("identity") and not (self.ownership.guardians(owner) & state.verified):
                if not state.verified:
                    return Verdict(False, "identity",
                                   "Verify the customer's identity first: confirm 2 of date of birth, email, phone number "
                                   "and address against their record, then call log_verification.")
                return Verdict(False, "ownership", f"{ref} belongs to a different customer than the one verified in "
                                                   "this conversation.")
        return None

    def observe(self, name: str, arguments: dict, result: str, state: ConversationState) -> None:
        """Update conversation state from a tool result (only successful effects count)."""
        ok = not str(result).lstrip().lower().startswith(("error", "failed"))
        args = arguments or {}
        if name == "log_verification" and ok and args.get("user_id"):
            state.verified.add(str(args["user_id"]))
        elif name == "unlock_discoverable_agent_tool" and ok:
            state.unlocked.add(str(args.get("agent_tool_name")))
        elif name == "give_discoverable_user_tool" and ok:
            state.given.add(str(args.get("discoverable_tool_name")))
        state.log.append({"tool": name, "ok": ok})

    @staticmethod
    def saw_documents(state: ConversationState, doc_ids: list[str]) -> None:
        state.read_docs.update(doc_ids)
