"""Who owns what: maps account, card and transaction ids to the customer they belong to."""

from __future__ import annotations

import copy
import json
from functools import lru_cache

from ..kb.store import data_dir

# argument name -> tables that can hold the id it refers to
ID_ARGS = {
    "user_id": ("users",),
    "account_id": ("accounts", "credit_card_accounts"),
    "checking_account_id": ("accounts",),
    "from_account_id": ("accounts",), "to_account_id": ("accounts",),
    "source_account_id": ("accounts",), "destination_account_id": ("accounts",),
    "credit_card_account_id": ("credit_card_accounts",),
    "card_id": ("debit_cards",),
    "transaction_id": ("credit_card_transaction_history", "bank_account_transaction_history"),
}


@lru_cache(maxsize=1)
def base_db() -> dict:
    return json.loads((data_dir() / "db.json").read_text(encoding="utf-8"))


def merge_initial_state(db: dict, initial_state: dict | None) -> dict:
    """Apply a task's ``initial_state.initialization_data.agent_data`` the way tau2 does (table rows added)."""
    out = copy.deepcopy(db)
    data = ((initial_state or {}).get("initialization_data") or {}).get("agent_data") or {}
    for table, payload in data.items():
        rows = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(rows, dict):
            tbl = out.setdefault(table, {"data": {}})
            target = tbl["data"] if isinstance(tbl, dict) and "data" in tbl else tbl
            target.update(rows)
    return out


def _rows(db: dict, table: str) -> dict:
    t = db.get(table) or {}
    return t.get("data", t) if isinstance(t, dict) else {}


class OwnershipIndex:
    def __init__(self, db: dict):
        self.db = db
        self._account_owner = {k: v.get("user_id") for k, v in _rows(db, "accounts").items()}

    def owner(self, arg: str, value: str) -> str | None:
        """The user_id that owns the object an argument names, or None if unknown."""
        if not isinstance(value, str) or not value:
            return None
        if arg == "user_id":
            return value if value in _rows(self.db, "users") else None
        for table in ID_ARGS.get(arg, ()):
            row = _rows(self.db, table).get(value)
            if row is None:
                continue
            if row.get("user_id"):
                return row["user_id"]
            if row.get("account_id") in self._account_owner:
                return self._account_owner[row["account_id"]]
        return None

    def owners_in(self, args: dict) -> dict[str, str | None]:
        """Owner of every id-like argument: {"card_id=dbc_1": "user_1", ...}."""
        out = {}
        for k, v in (args or {}).items():
            if k in ID_ARGS and isinstance(v, str):
                out[f"{k}={v}"] = self.owner(k, v)
        return out

    def user(self, user_id: str) -> dict | None:
        return _rows(self.db, "users").get(user_id)

    def guardians(self, user_id: str) -> set[str]:
        """Who may act for this customer: the customer, and a parent for a minor's accounts."""
        row = self.user(user_id) or {}
        return {user_id} | ({row["parent_user_id"]} if row.get("parent_user_id") else set())
