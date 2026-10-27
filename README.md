# bank-support-agent

A customer-support agent for a bank, over chat or a phone line. It answers from a 698-document policy and procedure base, carries out procedures with the bank's tools, and checks every tool call against policy before it runs.

It is built and measured on Sierra's [τ³-bench](https://github.com/sierra-research/tau2-bench) `banking_knowledge` domain (Rho-Bank): 97 tasks, 698 documents, 60 tools, graded on the bank's final database state.

![Console: the guard judging a task's reference solution with one call altered](docs/console.png)

## The problem

Support automation fails in three ways that τ³-Banking captures well:

- **The procedure is buried in the documents.** Most actions are "discoverable" tools such as `file_credit_card_transaction_dispute_4829`: the agent only learns a tool exists by finding and reading the document that names it, then has to unlock it before calling it. A missed document means the right action never happens.
- **Models break policy under pressure.** They act on an account before verifying the caller, touch another customer's account, invent a plausible tool name, or repeat a write.
- **Voice adds a latency budget.** Every check that runs between the caller finishing a sentence and the agent answering is heard as silence.

## How it works

```
caller ── chat, or voice: Silero VAD → Deepgram STT → turn detection ──┐
                                                                      ▼
                         ┌──────────── SupportAgent (agent/runtime.py) ────────────┐
                         │ model (Anthropic / Vertex AI / OpenAI-compat, httpx)     │
                         │   ├─ search_knowledge ─▶ hybrid retrieval (kb/search.py) │
                         │   ├─ read_document    ─▶ full text + tool signatures     │
                         │   └─ bank tool call   ─▶ PolicyGuard.check (≈8 µs)       │
                         │                           ├─ blocked: reason back to model│
                         │                           └─ allowed: execute            │
                         └──────────────────────────────────┬───────────────────────┘
                                                            ▼
                τ³ bank: tau2's own environment, in-process or inside tau2's orchestrator
                                                            │
caller ◀── Cartesia TTS ◀── sentence chunks, filler on slow turns, barge-in ◀──┘
```

**Retrieval** (`kb/`)
- Documents are split into sections. BM25 scores each section with the document title prepended, plus a share of the whole document's score, so a long procedure isn't hidden behind one generic heading.
- A dense leg (TF-IDF + truncated SVD, 128 dimensions) catches paraphrases, such as "charged twice" for a duplicate-transaction dispute. The two rankings are fused with reciprocal-rank fusion.
- Everything is local and builds in seconds: no embedding API and no vector database. Search results tell the model which internal tools each document names.

**Policy guard** (`guard/`)
Every bank call goes through the guard before it runs. The rules it enforces:
- `known_tool`: the discoverable tool exists. It also catches a user tool used as an agent tool.
- `documented_tool`: a document the agent actually read in this conversation names the tool.
- `unlock_first`: an agent tool is unlocked before it is called.
- `identity`: customer data is touched only after `log_verification` succeeded for that customer.
- `ownership`: every account, card and transaction in the arguments belongs to the verified customer. A verified parent may act for a minor's accounts.
- `no_duplicate_give`: a user tool is given to the customer only once.
- `transfer_consent` (optional): a decision model is asked whether the caller agreed to a transfer to a human.

A blocked call never reaches the bank. The model gets the rule and how to fix it ("verify the customer first…"). Ids the guard has never seen, such as an account created earlier in the call, are noted as advisory rather than blocked, because the bank's own tools reject ids that don't exist.

**Model calls** (`agent/llm.py`)
- An agent turn makes several model calls, and each one resends the policy, the system prompt and every tool schema (about 4K tokens before the conversation starts). The Anthropic adapter caches them: a breakpoint on the system block covers tools and system, and automatic caching moves forward with the conversation, so each call reads the previous call's prefix and only processes the new tail.
- `usage` in a session trace splits input into uncached, cache-read and cache-written tokens, for both adapters (OpenAI caches long prefixes on its own and reports `cached_tokens`).
- Claude on Google Cloud Vertex AI (`vertex/claude-sonnet-5-5`) sends the same request, caching included, to Vertex's `rawPredict` endpoint with a Google OAuth token from Application Default Credentials. It reads `ANTHROPIC_VERTEX_PROJECT_ID` and `CLOUD_ML_REGION` (default `global`), the same variables as Anthropic's SDK. With a Vertex project set and no `ANTHROPIC_API_KEY`, chat and voice use Vertex by default.
- `AnthropicLLM(cache=False)` sends plain requests, for gateways that reject `cache_control`. Retries follow the server's `retry-after` (capped at 30 s) before falling back to exponential backoff.

**τ³-bench integration** (`agent/tau2_plugin.py`)
- The agent is a tau2 `HalfDuplexAgent`, registered as `guarded_rag_agent`, so tau2's own runner, user simulator and grader run it unchanged.
- Ownership checks read the live tau2 database, so they cover rows the task seeds and rows created during the conversation.

**Voice** (`voice/`, Pipecat 1.12)
`AgentProcessor` takes the place of the LLM service in a cascaded pipeline:
- When the caller connects, the bot writes the model's prompt cache (a `max_tokens: 0` request) while the greeting plays, so the first turn doesn't wait for a cold read of the policy and tools.
- It collects final transcripts for a turn and waits a short grace period for a late one.
- It runs the agent off the event loop, so audio keeps flowing while the agent searches and calls tools.
- If the turn takes longer than 800 ms, it says a filler ("One moment while I check that.").
- When the caller barges in, it drops the reply that is still in flight.
- It strips markdown and pushes the reply one sentence at a time, so TTS starts on the first sentence.
- `LatencyTracker` records the time from the end of the caller's speech to the first text and to the first audio.

**Console** (`web/`) has four views:
- Guard: pick any task, watch the guard judge its reference solution, or switch to a broken copy.
- Knowledge: search the policy base.
- Results: the eval reports.
- Chat: talk to the agent when a model key is set.

## Results

### Read this first

- **No end-to-end model pass rate is reported here.** `support-agent tau2-run` runs the real benchmark (tau2's LLM user simulator plus this agent with your model), but it needs API keys and was not run for this README. The numbers below are model-free. They measure the parts this project adds: retrieval, the guard, the plumbing into tau2, and the voice layer's own latency.
- **Retrieval** used a fixed dev/test split (crc32 of the task id: 47 dev, 50 test). Settings were tuned on dev only, and test was run once at the end.
- **The guard** was not split that way. Its rules were refined while looking at all 97 reference solutions, for example the advisory handling of unseen ids and the parent/minor rule. Its false-block rate is therefore an in-sample number. The broken-copy catch rates are less exposed to this, because each broken copy targets a rule by construction.
- **The voice numbers** exclude STT, TTS and the network, which need live provider keys.

### Reference solutions through tau2's own harness

Each task's reference solution is played through this agent's runtime inside tau2's `Orchestrator`, with a scripted customer, and graded by tau2's evaluator on the final database state. This checks that the runtime, the knowledge reads, the tool plumbing and the guard can reproduce a correct solution end to end.

| | tasks at reward 1.0 |
|---|---:|
| guard off | **97 / 97** |
| guard on | **95 / 97** |

The guard costs two tasks:
- **task_005:** the reference solution changes the email of a user other than the one verified, which `ownership` blocks. This is arguably what the policy asks for.
- **task_083:** the reference solution uses `get_all_user_accounts_by_user_id_3847`, a tool that none of the task's required documents names. Two more tools in task_051 have the same gap, but they are read-only lookups, so that task keeps its reward.

task_102 also has a natural-language assertion that needs an LLM judge, so it was graded on its database state only.

### Policy guard

Replaying all 97 reference solutions call by call, then a broken copy of each solution with one unsafe change:

| broken copy | caught at the altered call |
|---|---:|
| skip identity verification | 67 / 67 |
| act on another customer's account, card or transaction | 67 / 67 |
| invented tool name (changed digits) | 76 / 76 |
| tool no document read so far names | 76 / 76 |
| call a tool before unlocking it | 67 / 67 |

- 3 of 97 correct solutions were blocked (task_005, task_051, task_083), for the reasons above.
- 833 checks took 8.3 µs each on average, which makes the guard free in a voice latency budget.
- A broken copy counts only if the block lands on the altered call itself.

### Retrieval

The query is either the customer's scripted opening line or the whole scenario. A live agent searches several times as the call unfolds, so these one-shot numbers are lower bounds. The baseline is tau2's own BM25 retriever: `rank_bm25` over whole documents with whitespace tokens.

Recall@10 is the share of a task's required documents found in the top 10. Tool docs@10 is the share of the reference solution's discoverable tools that some top-10 document names.

| test split (50 tasks), held out | recall@10 | tool docs@10 | MRR |
|---|---:|---:|---:|
| tau2 BM25, opening line | 6.3% | 4.7% | 0.258 |
| **hybrid, opening line** | **24.6%** | **31.0%** | **0.529** |
| tau2 BM25, scenario | 17.0% | 27.5% | 0.361 |
| **hybrid, scenario** | **37.0%** | **56.7%** | **0.727** |

Full tables, including BM25 alone and dense alone, are in [`reports/retrieval_test.md`](reports/retrieval_test.md) and [`reports/retrieval_dev.md`](reports/retrieval_dev.md). A query takes about 25 ms on one core.

### Voice turn latency (offline)

This measures the time from the end of the caller's speech to the first sentence handed to TTS. All 97 opening lines are spoken into the Pipecat processors, and the agent runtime runs for real: search and guard. The model is a stand-in with a fixed delay per call, two calls per turn.

| model latency per call | first output p50 | p95 | agent turn p50 | fillers |
|---:|---:|---:|---:|---:|
| 0 ms | 280 ms | 295 ms | 28 ms | 0 / 97 |
| 400 ms | 1052 ms | 1054 ms | 832 ms | 97 / 97 |

- With an instant model, the pipeline adds about 30 ms on top of the 250 ms end-of-turn grace.
- With a realistic model, the caller hears the filler at 0.8 s rather than silence until about 1.1 s.
- To get real p50/p95 numbers with STT and TTS, `support-agent voice` logs both latency trackers when the call ends.

## Run it

Python 3.10+ for the core; tau2-bench needs 3.12+.

```bash
pip install -e ".[dev]"                      # core: retrieval, guard, console, evals
bash scripts/install_tau2.sh                 # tau2-bench at the pinned commit, from source (3.12+)
pip install -e ".[voice]"                    # Pipecat with Deepgram, Cartesia, Silero, WebRTC
pip install -e ".[vertex]"                   # Claude on Google Cloud Vertex AI (google-auth)

support-agent search "I was charged twice at a hotel"
support-agent eval-retrieval --split dev --out reports/retrieval_dev
support-agent eval-guard --out reports/guard
support-agent eval-harness --workers 6 --out reports/harness      # needs tau2
support-agent bench-voice --out reports/voice_latency            # needs voice
support-agent serve                                               # console on http://127.0.0.1:8090

# with keys (see .env.example)
support-agent chat --task task_026 --trace                        # terminal chat against the τ³ bank
support-agent voice --task task_026                               # browser call over WebRTC
support-agent tau2-run --llm anthropic/claude-sonnet-5-5 --trials 4   # the real benchmark

# Claude on Vertex AI: gcloud auth application-default login, then
ANTHROPIC_VERTEX_PROJECT_ID=my-project support-agent chat --llm vertex/claude-sonnet-5-5
```

- `--task` seeds the bank with that task's customers, so you can play the customer.
- `scripts/build_catalog.py --check` confirms the shipped tool catalog still matches tau2's tool definitions.
- `docker build -t bank-support-agent .` builds an image that serves the console.

## Layout

```
src/support_agent/
  kb/          documents, sections, hybrid search, tau2 BM25 baseline
  guard/       ownership index, policy guard
  agent/       model adapters, runtime, backends (tau2 bank / scripted), tau2 plugin and oracle
  voice/       Pipecat processors, pipeline, offline latency bench
  eval/        tasks and split, retrieval, guard and harness evals
  web/         console
  data/rho_bank/   τ³-bench banking data (MIT, Sierra), pinned commit in SOURCE
tests/         65 tests; tau2 and voice tests skip when those extras are missing
reports/       the numbers above
```

## Limits

- No model-in-the-loop pass rate has been measured yet (see above). That number is the real test, and the guard's value shows up as policy violations avoided per task.
- `transfer_consent` is the only rule that needs judgement, and it is optional. A decision model, such as Jev or a small classifier, plugs in through `decision_model.ask(question, recent_turns)`.
- The guard enforces rules that can be checked from state. It cannot tell whether the agent chose the right procedure, only that the procedure is documented, unlocked and scoped to the right customer.
- The dense leg is LSA, which runs fully offline. A neural embedding model would likely lift recall further, at the cost of an API call or a local model.
