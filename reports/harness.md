# Reference trajectories through the τ³-bench harness

Each task's reference solution is played through the guarded agent runtime inside tau2's orchestrator, and graded by tau2's evaluator (final database state).

| | tasks at reward 1.0 |
|---|---:|
| guard off | 97 / 97 |
| guard on | 95 / 97 |

Graded on the database only (their natural-language assertions need an LLM judge): task_102.

Tasks that pass without the guard but not with it:

| task | blocked call |
|---|---|
| task_005 | change_user_email: Blocked by policy guard (ownership): user_id=6680a37184 belongs to a different customer than the one verified in this co |
| task_083 | unlock_discoverable_agent_tool: Blocked by policy guard (documented_tool): No document you have read names 'get_all_user_accounts_by_user_id_3847'. Sear; unlock_discoverable_agent_tool: Blocked by pol |
