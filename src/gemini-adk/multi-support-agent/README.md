## Multi-agent examples: Customer Support

These examples all start from one use case, customer support for **ShopSmart**,
a mock online retailer, and grow it from a single agent into multi-agent designs.

```
multi-support-agent/
├── support_runner.py      # interactive CLI runner (prints tool calls)
├── shared/
│   ├── model.py           # get_model(): gemini / doubleword / openai_compatible via MODEL_PROVIDER
│   ├── mock_data.py       # customers, products, orders, shipments, policies (in-memory)
│   └── tools.py           # 7 read tools + 4 write tools, reused by every example
├── support_single/        # baseline: one LlmAgent with every tool
├── support_routing/       # triage agent hands off to specialists with narrow tool sets
├── support_sequential/    # resolver -> policy checker -> responder on every turn
├── support_parallel/      # intake -> concurrent researchers -> responder
└── support_loop/          # draft -> review -> revise until the reply passes
```

### Running

From this directory (`.env` is loaded from `src/.env`):

```bash
python support_runner.py support_single     # CLI chat, shows tool calls
python support_runner.py support_single -v  # also dumps every event as JSON
python support_runner.py support_routing    # shows ⇢ transfers between agents
python support_runner.py support_sequential # internal agents' notes are shown dimmed
python support_runner.py support_parallel   # tool calls are timestamped to show overlap
python -m support_parallel.benchmark        # parallel vs sequential research timing
python support_runner.py support_loop       # every draft and review verdict is shown dimmed
adk web                                     # pick a support_* app in the dev UI
```

### The tools

| Tool | Type | Notes |
|---|---|---|
| `lookup_customer(email)` | read | Verifies the customer; stores `customer_id` / `customer_name` in session state |
| `list_customer_orders()` | read | Orders for the verified customer |
| `get_order_details(order_id)` | read | Line items, totals, refunded amount |
| `track_shipment(order_id)` | read | Carrier, shipment status, scan history |
| `search_products(query)` | read | Catalog search (stock, restock date, warranty) |
| `search_policies(query)` | read | Store policy KB: returns, refunds, shipping, cancellations, warranty, price match, loyalty |
| `check_return_eligibility(order_id, sku)` | read | 30-day window, non-returnable items, return fee (free for gold tier) |
| `initiate_return(order_id, sku, reason)` | write | Creates an RMA + label |
| `cancel_order(order_id)` | write | Only while `processing` |
| `issue_refund(order_id, amount, reason)` | write | Auto-approved up to **$100**; larger amounts must be escalated |
| `create_support_ticket(subject, description, priority, order_id)` | write | Escalation to humans; gold tier is bumped to high priority |

**Guardrail in code:** every order-level tool calls `_verified_order()`, which
rejects the call unless `lookup_customer` has verified a customer and the order
belongs to them. The prompt asks for verification as well, but the code enforces it,
so a prompt-injected or confused model still can't read or refund someone else's order.

### Mock scenarios to try

| Customer (email) | Order | Scenario |
|---|---|---|
| alice.chen@example.com (gold) | ORD-5001 | Delivered 10 days ago: return the headphones (free return shipping) |
| | ORD-5002 | Shipped, **delayed** by weather |
| bob.martinez@example.com (standard) | ORD-5003 | Smartwatch delivered 47 days ago: **outside** the return window |
| | ORD-5004 | Still processing: **cancel**; includes a backordered keyboard and a non-returnable gift card |
| priya.patel@example.com (silver) | ORD-5005 | $249 watch arrived cracked: refund exceeds $100, so it **escalates** |

Sample prompts:
- "Where is my rain jacket? My email is alice.chen@example.com"
- "I want to return the headphones, they hurt my ears" → confirm → RMA
- "My smartwatch arrived cracked, I'm priya.patel@example.com, order ORD-5005" → ticket
- "When will the TypePro keyboard be back in stock?" (no verification needed)
- "Cancel ORD-5004" while verified as Priya → refused (not her order)

Data is in-memory, so restart the process to reset returns, refunds, and tickets.
Dates are relative to today, so the return-window scenarios always hold.

### Single agent (`support_single/`)

One `LlmAgent` with all 11 tools and one long instruction covering identity,
policies, confirmation-before-write, refund limits, and escalation. It works, and
it's the baseline for the multi-agent versions. Its weak points are what those
versions address:

- **One prompt does everything.** Rules for returns, refunds, and escalation all
  compete for attention, and adding a new domain means growing a single prompt.
- **Every tool is always visible.** A product question still exposes `issue_refund`.
  More tools means more chances to pick the wrong one.
- **No separation of duties.** The agent that decides to refund also issues the refund.
  No second agent reviews the action.
- **Hard to evaluate in pieces.** You can only test the whole agent end to end.

### Routing (`support_routing/`)

```
triage_agent            tools: lookup_customer
├── order_agent             list_customer_orders, get_order_details, track_shipment, cancel_order
├── returns_refunds_agent   get_order_details, check_return_eligibility, initiate_return, issue_refund
├── product_policy_agent    search_products, search_policies
└── escalation_agent        get_order_details, create_support_ticket
```

The triage agent verifies the customer if needed, then hands off. It doesn't solve
anything itself.

**How it works in ADK:**
- Passing `sub_agents=[...]` gives every agent in the tree a built-in
  `transfer_to_agent` tool. The model picks a target from each agent's
  **`description`**, so descriptions are the routing contract and need to be precise.
- After a transfer, **the new agent stays active** for the next user turns until it
  transfers again. This is a handoff, not a call-and-return.
- Specialists can transfer to **peers** as well as back to the parent. In testing, the
  order agent passed "return my headphones" straight to the returns agent.
- All agents share **one session**. The `customer_id` that triage writes to state is what
  the specialists' tools check, and every prompt shows `{customer_name?}`.
- If a specialist's tool fails with "not verified", it transfers back to triage,
  which asks for the email and then routes back.

**What improved over `support_single/`:**
- Each specialist sees 2–4 tools instead of 11. The product agent can't issue a refund.
- Each prompt is short and domain-specific. Shared rules live in `COMMON_RULES`.
- The refund limit is enforced twice: `issue_refund` rejects amounts over $100 in code,
  and only the escalation agent can create tickets.

**Trade-offs to discuss:**
- **Domain rules must live with the specialist that applies them.** An early version
  of the returns agent treated a damaged item as a normal return and kept a $6.99
  shipping fee. Its prompt didn't mention the damaged-on-arrival rule, and it has no
  access to `search_policies`.
- **More LLM calls and latency.** Every handoff is an extra model turn.
- **Routing errors are a new kind of failure.** Evaluate the routing itself:
  "given this message, which agent should take it?"
- Handoffs can **bounce** (specialist → triage → specialist) when context is missing.

### Sequential (`support_sequential/`)

```
SequentialAgent support_pipeline       (runs all three, in order, on every customer turn)
├── resolver_agent        read-only tools + propose_action
├── policy_checker_agent  get_order_details, check_return_eligibility, search_policies + record_verdict
└── responder_agent       execute_approved_action, discard_proposal
```

This version adds **separation of duties**. The agent that decides what to do can't do
it, the agent that approves never sees the customer, and the agent that executes can
only run what was approved.

**The propose, review, execute contract** (`support_sequential/workflow_tools.py`):
1. The resolver investigates and calls `propose_action(...)`, which writes
   `state["proposal"]`. It has no write tools.
2. The checker reviews that proposal and calls `record_verdict(approved, reason)`, which
   writes `proposal["review"]`.
3. The responder replies to the customer. When the customer confirms, it calls
   `execute_approved_action()`, which **takes no arguments**. It looks up the stored
   proposal and runs the underlying write tool with exactly the approved args. The model
   can't change the amount or run something that wasn't approved.

**ADK features shown:**
- `SequentialAgent`: a fixed pipeline with no LLM deciding the order. Compare with the
  LLM-driven handoffs in `support_routing/`.
- **State as the handoff channel:** `output_key="resolver_findings"`, plus
  `{proposal?}` / `{resolver_findings?}` templates in the next agents' instructions.
- `include_contents="none"` on the checker: it doesn't see the chat history, so a pushy
  or prompt-injecting customer can't argue it into approving. It judges only the
  proposal and facts it looks up itself.
- **Callbacks to skip work:** if there's no new proposal, the checker's model call is
  replaced with a canned reply (`before_model_callback` returns an `LlmResponse`).
  The skip decision is made once in `before_agent_callback` and stored in a
  `temp:` key, because the checker makes several model calls per run. Note that
  returning content from `before_agent_callback` would **end the whole invocation**
  and skip the responder.

**Try it:**
- Alice returns her headphones: the checker approves and the responder asks for a "yes".
  Only after the customer confirms does `execute_approved_action` create the RMA.
- Priya's $249 damaged watch: the resolver proposes a full refund, the checker
  **rejects** it as over $100 and says to escalate, and the responder offers escalation.
  When she agrees, a ticket is proposed, approved, and created.

**Trade-offs to discuss:**
- **Cost and latency:** at least 3 agents run every turn, even for "where's my order?".
  The skip callback saves the checker's model call, but the resolver and responder always run.
- **The confirmation gate is still an LLM judgment.** In testing, "yes please escalate it"
  was treated as confirming the offered escalation, so the ticket was created in the same
  turn. That's reasonable, but the code doesn't enforce it. The approval gate is enforced in code.
- **The checker is an LLM too.** Hard limits (refund at most $100, ownership) are also
  enforced in the write tools, so the checker is defense in depth, not the only gate.
  A natural next step is a code-only checker for the rules that can be written as code.
- **Fixed order:** the pipeline can't skip the resolver or loop back. The loop example
  will address that.

### Parallel (`support_parallel/`)

```
SequentialAgent support_parallel
├── intake_agent             lookup_customer, list_customer_orders, record_intake
├── ParallelAgent research_team          (concurrent)
│   ├── order_researcher       get_order_details, track_shipment, check_return_eligibility -> order_research
│   ├── policy_researcher      search_policies                                          -> policy_research
│   └── product_researcher     search_products                                          -> product_research
└── responder_agent          initiate_return, cancel_order, issue_refund, create_support_ticket
```

Most support questions need context from several places: "my jacket is late, can I
still return it, and do I pay shipping?" needs the order, tracking, *and* the returns
policy. These lookups are independent, so they can run **at the same time**.

**How it works in ADK:**
- `ParallelAgent` runs each sub-agent in its **own branch**. Branches share session
  state but don't see each other's messages, so each researcher writes to its **own
  `output_key`** and the responder reads all three through `{order_research?}` etc.
  Two branches writing the same key would race.
- The **intake agent plans the fan-out.** `record_intake` writes the plan
  (`intake_order_id`, `intake_policy_query`, `intake_product_query`) to state, and each
  researcher has a `before_model_callback` (`skip_unless(key)`) that skips its LLM call
  when its part of the plan is empty. A price-match question runs only the policy
  researcher. A "yes" confirmation runs none of them.
- `reset_turn_state` (intake's `before_agent_callback`) clears last turn's plan and
  research, so stale results can't leak into this turn's reply.

**Measured** (`python -m support_parallel.benchmark`, a prompt that needs all three researchers,
mock tools, GLM-5.3-Flash via `openai_compatible`):

| research_team as | research phase | whole turn |
|---|---|---|
| `SequentialAgent` | 9.8–18.6s | 17.0–27.4s |
| `ParallelAgent` | 4.0–6.7s | 10.8–16.2s |

(Four runs per mode across two sessions. LLM latency varies, but parallel was 2–3x
faster on the research phase every time.)

The parallel phase takes as long as the **slowest** researcher, not the sum of all three.
With real APIs instead of instant mock tools, the gap would be wider.

**Trade-offs to discuss:**
- **Parallel only helps independent work.** The order researcher can't start until intake
  has found the order id, which is why intake is a sequential step in front.
- **Combining results is a job in itself.** The responder must reconcile notes that can
  overlap or disagree. In testing, the policy researcher noted that worn headphones may
  not count as "unused", and the responder had to decide how to present that caveat.
- **Token cost isn't lower,** only latency. Every active researcher re-reads the
  conversation. Skipping researchers that aren't needed is what saves tokens.
- **The plan is a single point of failure.** If intake forgets to set `policy_query`,
  the policy researcher never runs and the responder works without policy facts.

### Loop (`support_loop/`)

```
SequentialAgent support_loop
├── case_agent              all tools; investigates, runs only actions confirmed last turn
├── LoopAgent reply_refiner     (max_iterations=3)
│   ├── drafter_agent       writes / revises the reply                 -> state["draft"]
│   └── reviewer_agent      code fact check + LLM rubric; search_policies,
│                           approve_draft() exits the loop, request_revision() loops again
└── support_agent           custom BaseAgent (no LLM): publishes the approved reply,
                            or a safe fallback that sets needs_human_review
```

**Scenario: Bob's multi-part complaint.** Every part has a trap:

> "This is ridiculous. I'm bob.martinez@example.com. My PulseFit watch stopped working,
> I want a refund. And where is my keyboard? Just cancel the gift card, I don't want it anymore."

| Request | Correct answer | Trap |
|---|---|---|
| Watch refund (ORD-5003) | Outside the 30-day window; offer a **warranty ticket** | Promising a refund, or a flat "no" |
| Where's the keyboard (ORD-5004) | **Backordered**, restock date, ships after | "Still processing" with no date |
| Cancel the gift card | Cancellation is **whole order only**; offer to cancel ORD-5004 ($114) or keep it | "Done!", or refunding just the $25 |
| Tone | Acknowledge "this is ridiculous" once | Generic "thanks for reaching out" |

Follow up with *"Yes, open the warranty ticket. Keep the order."* The ticket is created
once, by the case agent, outside the loop.

**The reviewer combines code and an LLM:**
- `fact_check.py` (code, runs in the reviewer's `before_agent_callback`): every id, $ amount,
  and date in the draft must appear in a tool result or the policies. The case agent's
  `after_tool_callback` logs every tool result to `state["tool_facts"]`. This reliably
  catches invented ids ("TKT-3001" before any ticket exists), amounts, and dates.
- The LLM rubric covers what code can't: completeness, policy correctness (checked with its
  own `search_policies` calls), honesty about actions, and tone.
- `approve_draft(complete, policy_correct, honest, tone)` makes the reviewer give **a
  verdict per item**, and **refuses** if any is false or the code fact check failed.
  Approving sets `escalate=True`, which exits the `LoopAgent` but not the outer
  `SequentialAgent`.

**What the loop caught in testing:**
- Draft 1 said "I'm sorry about the trouble" to a customer who wrote "this is ridiculous".
  Rejected for tone, fixed in draft 2.
- A draft offered to refund just the $25 gift card. Rejected as policy-incorrect.
- A draft promised "a $249 refund will follow" if the warranty claim is approved. The case
  agent invented that, and an early reviewer that trusted the case notes approved it. Once
  the reviewer checked the policy text itself, it was rejected.

**ADK features shown:**
- `LoopAgent(max_iterations=3)`, exited early by a tool setting `tool_context.actions.escalate`.
- `include_contents="none"` on the drafter and reviewer: they work from state (`customer_message`,
  `case_notes`, `draft`, `review_feedback`), so earlier drafts don't pile up in their context.
- Callbacks: `before_agent_callback` (reset the turn, count drafts, run the fact check) and
  `after_tool_callback` (log tool facts).
- A **custom `BaseAgent`** (`ReplyPublisher`): plain Python in a pipeline that emits the
  final event. If the loop hits the limit without an approval, it sends a safe holding
  message and flags `needs_human_review` instead of an unverified draft.

**Lessons from building it:**
- **The reviewer is only as good as its reference.** When it judged policy against the case
  notes, it approved the case agent's mistakes. Give it an independent source.
- **Strict reviewers produce false positives.** It once rejected a ticket's "1 business day"
  response time as unbacked by policy. Telling it which details are tool facts and which are
  policy claims fixed that.
- **Better tools beat longer prompts.** The first run never mentioned the keyboard backorder,
  because the case agent didn't look up stock. Adding stock and restock data to
  `get_order_details` fixed it for every example.

**Trade-offs to discuss:**
- **Cost and latency grow with each revision:** about 25s with one revision, about 33s
  with three drafts, versus about 8s when draft 1 passes.
- **Quality varies run to run:** the same message may pass on draft 1 or need all three.
  The code fact check gives the same result every run; the LLM rubric doesn't.
- **What happens at max_iterations** is a product decision: send the last draft, a holding
  message, or hand off to a human. This example picks the safe option.
