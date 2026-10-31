## Multi-agent examples: Customer Support, as ADK 2.0 graph workflows

This folder reimplements every example in [`../multi-support-agent`](../multi-support-agent)
with ADK 2.0's [graph-based workflows](https://adk.dev/graphs/): a `Workflow` of **nodes**
(LLM agents, plain functions) connected by **edges** (optionally conditional on a route).

Same use case (ShopSmart, mock data, same tools), same scenarios, so the two folders can be
compared side by side. The main idea throughout: **move control flow out of prompts and
into the graph.** LLM nodes do language work (classify, plan, investigate, write). Code
nodes do the deciding (route, verify, gate, execute, loop).

```
multi-support-graph-agent/
├── support_runner.py      # CLI runner: tool calls, routes taken, human-input prompts
├── shared/
│   ├── model.py           # get_model(): gemini / doubleword / openai_compatible via MODEL_PROVIDER
│   ├── mock_data.py       # customers, products, orders, shipments, policies (in-memory)
│   ├── tools.py           # the same 11 tools as ../multi-support-agent
│   └── graph.py           # conversation memory, identity check, reply helpers for graphs
├── support_single/        # START -> one chat-mode agent
├── support_routing/       # LLM classifies, code routes to specialists
├── support_sequential/    # propose -> check -> pause for the customer -> execute in code
├── support_parallel/      # LLM plans, code researches concurrently, LLM replies
└── support_loop/          # draft -> code fact check -> LLM review -> revise (a graph cycle)
```

### Running

From this directory (`.env` is loaded from `src/.env`). Requires `google-adk >= 2.0`.

```bash
python support_runner.py support_single      # CLI chat
python support_runner.py support_routing     # ⤷ lines show the route each code node chose
python support_runner.py support_sequential  # pauses with "asks:" for a yes/no
python support_runner.py support_parallel    # ◆ lines show research nodes finishing together
python support_runner.py support_loop        # every draft, check, and review is shown dimmed
python -m support_parallel.benchmark         # fan-out vs one-by-one research timing
adk web                                      # pick a support_* app in the dev UI
```

Each `agent.py` exports `root_agent` (the `Workflow`) and `app` (an `App` wrapping it).
`adk web` loads `app`, which matters for `support_sequential`: its human-input pause needs
`ResumabilityConfig(is_resumable=True)`.

---

### Things to know about graph workflows (learned building these)

These were verified against `google-adk` 2.12 and shaped every design below.

1. **LLM nodes have no memory by default.** An `LlmAgent` used as a node runs in
   `single_turn` mode: it sees its node input and nothing else, not even its own previous
   turns. A `mode="chat"` agent keeps history, but ADK only allows it **directly after
   `START`**, because it reads session history instead of a node input.
   So every multi-node graph here keeps memory explicitly in state:
   `remember_customer` appends each message to `state["conversation"]`, `record_reply` /
   `reply()` append the agent's answer, and LLM nodes read `{conversation}` in their
   instructions (`shared/graph.py`).
2. **Routes are just values.** A function node returns `Event(output=..., route="x")`, and
   edges like `(node, {"x": a, "y": b})` pick the next node. A route can also be a list
   (multi-route fan-out).
3. **Function nodes get state as parameters.** Besides `ctx` and `node_input`, any
   parameter name is read from `ctx.state`. State updates are returned with
   `Event(state={...})` so they're recorded in the event history.
4. **`output_schema` turns an LLM into a typed node.** Its output reaches the next node as
   a dict (`RouteDecision`, `ResearchPlan`, `Review`), which code can validate and branch on.
5. **A `JoinNode` waits for every incoming branch.** If a branch might not run, the join
   never fires, so researchers always run and return "not needed" instead of being skipped.
6. **Human input is a pause, not a message.** A node yields `RequestInput(...)`, the
   workflow stops, and the client sends back a function response named `adk_request_input`
   with the same id. The node (with `rerun_on_resume=True`) runs again with the answer in
   `ctx.resume_inputs`. `adk web` renders this natively; `support_runner.py` does it in the CLI.
7. **Cycles need a routed edge.** An unconditional cycle is rejected when the graph is
   built, so every loop has an explicit exit route.

---

### Single (`support_single/`)

```
START -> support_agent (mode="chat", all 11 tools)
```

The smallest possible workflow, equivalent to the single `LlmAgent` baseline. Because the
agent comes directly after `START`, it can use `mode="chat"` and keep the conversation
natively. Everything else is up to the model, including when "yes" means yes.

### Routing (`support_routing/`)

```
START -> remember_customer -> verify_identity -> classify -> dispatch
dispatch --order-----------> order_agent ----------\
         --returns_refunds-> returns_refunds_agent -+
         --product_policy--> product_policy_agent --+-> record_reply
         --escalation------> escalation_agent -----/
         --verify----------> ask_for_email
         --unclear---------> ask_to_clarify
```

| | `../multi-support-agent` (LLM transfers) | graph |
|---|---|---|
| Who routes | each agent's LLM via `transfer_to_agent` | `classify` (LLM) labels, `dispatch` (code) routes |
| Routing table | agent `description`s | the graph's edges |
| Verification | triage prompt + tool errors bouncing back | `verify_identity` (code) + a `verify` route enforced by `dispatch` |
| "yes" / follow-ups | the active agent keeps the conversation | `follow_up` label → `last_route` in state (sticky in code) |
| Mid-turn handoffs | specialists transfer to peers | none: one classification per turn |

**Tested:** "Where is my rain jacket?" before giving an email went to `ask_for_email`. The
email turn was labeled `follow_up` and went to the order specialist. "Yes please" stayed with
the returns specialist, and a stock question switched to the product specialist.

**Watch out:** the transcript holds only text, not earlier tool results. In testing, the
returns specialist first called `initiate_return` with an invented SKU on the "yes" turn,
got an error, and recovered by looking the order up again.

### Sequential (`support_sequential/`)

```
START -> remember_customer -> verify_identity -> start_turn -> resolver -> after_resolve
after_resolve --no_action--> responder
              --review-----> policy_checker -> after_review
after_review  --rejected---> responder
              --approved---> confirm_and_execute  ⏸ RequestInput("... Shall I go ahead? (yes/no)")
confirm_and_execute -------> responder -> record_reply
```

The biggest upgrade over the `SequentialAgent` version:
- **Confirmation is a real human-in-the-loop pause.** `confirm_and_execute` yields a
  `RequestInput`, and the workflow stops until the customer answers. No LLM decides whether
  "yes please escalate it" counts as confirmation; code checks the answer.
- **Execution is code.** After a yes, `confirm_and_execute` runs the stored, approved
  proposal itself. No LLM in this graph holds a write tool.
- **Branching replaces skip callbacks.** The checker only runs on the `review` route, and
  the pause only on `approved`.
- The responder has no tools. It only describes what happened.

**Tested:**
- Alice's return: proposed, approved, paused for confirmation; "yes" → RMA created in code
  (that turn took 1.9s, with no LLM until the reply).
- Priya's $249 damaged watch: the refund was rejected (over $100), and the responder offered
  escalation. When she asked, a ticket was proposed, approved, and paused. "No, wait" →
  nothing was created.

**Fixed along the way:** the checker applied the $100 direct-refund limit to a *return*. The
responder claimed "I've escalated this" when it had only been proposed, and promised
follow-ups. Each is now a specific rule in the prompts. Note that none of them could
cause a wrong *action*, because execution is gated in code.

### Parallel (`support_parallel/`)

```
START -> remember_customer -> verify_identity -> load_orders -> planner
planner -> (research_order, research_policy, research_product)     # fan-out, concurrent
        -> JoinNode research_done                                   # fan-in
        -> compile_brief -> responder -> record_reply
```

- **The researchers are plain functions, not LLMs.** Looking up an order, a policy, or a
  product is deterministic, so only planning and replying use a model. That's cheaper,
  faster, and gives the same result every time.
- `planner` returns a typed `ResearchPlan` (`output_schema`). `load_orders` (code) first
  gives it the customer's orders, so it can resolve "my jacket" to an id without tools.
- Fan-out is `(planner, (a, b, c))`, and fan-in is a `JoinNode` that hands
  `{"research_order": ..., "research_policy": ..., "research_product": ...}` to `compile_brief`.

**Measured** (`python -m support_parallel.benchmark`; each researcher simulates 1.0s of
backend latency, because the mock tools are instant):

| research as | research phase | whole turn |
|---|---|---|
| one after another | 3.0s | 9.5–10.0s |
| fan-out + `JoinNode` | 1.0s | 6.8–10.4s |

The research phase is deterministic: the slowest branch, not the sum. Whole-turn time is
dominated by the two LLM calls and varies run to run.

### Loop (`support_loop/`)

```
START -> remember_customer -> verify_identity -> case_agent -> start_drafting -> drafter -> fact_check
fact_check --passed--> reviewer -> decide
           --failed------------> decide          # skips the LLM review
decide --approved--> publish
       --revise----> drafter                     # the cycle
       --give_up---> fallback                    # after 3 drafts; flags needs_human_review
```

- **The loop is an edge**, `(decide, drafter, "revise")`. ADK rejects unconditional cycles,
  so the exit condition is always explicit.
- **`decide` is code.** The LLM reviewer only fills in a structured `Review`
  (`complete`, `policy_correct`, `honest`, `tone`, `issues`), and code approves only if all
  four are true. The draft count and max-drafts limit are code too.
- **The fact check is a node.** When the draft contains an id, amount, or date that no tool
  returned, the graph goes straight to `decide` → `revise` without spending a reviewer call.
- `publish` and `fallback` are function nodes; no custom `BaseAgent` is needed.

**Tested with Bob's multi-part complaint** (watch outside the return window → warranty;
backordered keyboard → restock date; gift card → whole-order cancellation only; "this is
ridiculous" → acknowledge frustration):
- One run: draft 1 wrote "the return window closed on 2026-10-11", a date the model
  calculated. `fact_check → failed`, revised without an LLM review, and draft 2 was approved.
- Follow-up "Yes, open the warranty ticket": the reviewer rejected "it will ship as soon as
  it's back in stock" as an unbacked promise, and the revision was approved.
- Another run passed on draft 1. LLM results vary; the code checks don't.

---

### Original vs graph: summary

| | `../multi-support-agent` | this folder |
|---|---|---|
| Building blocks | `LlmAgent`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`, custom `BaseAgent` | one `Workflow` per example: LLM nodes + function nodes + edges |
| Control flow | mostly in prompts and callbacks | in edges and routes, visible in one place |
| Conversation memory | automatic (shared session history) | explicit `{conversation}` in state (except `support_single`) |
| Confirm before acting | LLM judges "yes" | `support_sequential`: `RequestInput` pause + code |
| Deterministic work | inside tools | its own nodes (`verify_identity`, researchers, `fact_check`, `decide`) |
| Cost of being explicit | — | more nodes and state keys to manage; memory plumbing; no mid-turn handoffs |

Data is in-memory: restart the process to reset returns, refunds, and tickets.
