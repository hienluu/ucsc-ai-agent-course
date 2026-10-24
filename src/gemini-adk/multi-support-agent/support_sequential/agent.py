"""Sequential: resolver -> policy checker -> responder, on every customer turn.

    SequentialAgent support_pipeline
    ├── resolver_agent        investigates with read-only tools, may PROPOSE one action
    ├── policy_checker_agent  reviews the proposal against policy, without the chat history
    └── responder_agent       talks to the customer, EXECUTES only an approved,
                              customer-confirmed action

Separation of duties: the agent that decides what to do can't do it, the agent
that approves never sees the customer's pressure, and the agent that executes
can only run the exact action that was approved (see workflow_tools.py).

Agents pass work along through session state:
    resolver  -> output_key "resolver_findings" + state["proposal"]
    checker   -> state["proposal"]["review"]
    responder -> reads both, writes the customer-facing reply
"""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from shared.model import get_model
from shared.tools import (
    check_return_eligibility,
    get_order_details,
    list_customer_orders,
    lookup_customer,
    search_policies,
    search_products,
    track_shipment,
)

from .workflow_tools import (
    discard_proposal,
    execute_approved_action,
    propose_action,
    record_verdict,
)

MODEL = get_model()

RESOLVER_INSTRUCTION = """\
You are the resolver for ShopSmart customer support. You work behind the scenes:
the customer never sees your output. Another agent writes the reply.

Verified customer (from session state): {customer_name?}
Pending proposal (from session state): {proposal?}

Your job on each turn:
1. Understand what the customer wants from the conversation.
2. Gather facts with your read-only tools. Never invent facts.
   - Anything about a specific order needs a verified customer. If the customer is not
     verified and hasn't given an email, note that the responder must ask for it. If they
     gave an email, call `lookup_customer`.
   - Product and policy questions don't need verification.
3. If the customer wants something changed (return, cancel, refund, escalate to a human),
   call `propose_action` with exactly ONE action. You cannot execute it.
   - Returns: call `check_return_eligibility` first; include the estimated refund in the summary.
   - Damaged or defective on arrival: propose `issue_refund` for the full item price
     (no return needed). If the refund would be too large, the policy checker will say so.
   - A human, a complaint, a legal or safety issue, or a policy checker rejection that says to
     escalate: propose `create_support_ticket` with full context.
4. If the latest message just confirms (yes) or declines (no) the pending proposal,
   do NOT call any tools or propose anything new. Write "CUSTOMER CONFIRMED" or
   "CUSTOMER DECLINED".

Output: short internal notes for the responder covering the facts found (ids, dates,
amounts, statuses), what was proposed, and what is still missing.
"""

CHECKER_INSTRUCTION = """\
You are the policy checker for ShopSmart. You review ONE proposed action before it
can be executed. You don't talk to the customer and you don't see the conversation;
judge only the proposal and the facts you look up yourself.

Proposal to review:
{proposal?}

Check, using your tools, that:
- The order exists and belongs to the verified customer (`get_order_details`).
- The action matches policy (`search_policies`, e.g. "returns", "refunds", "cancellations").
- Returns: the item is eligible (`check_return_eligibility`).
- Damaged-on-arrival items: these get a full refund with no return and no shipping fee,
  so reject an `initiate_return` that would charge a fee for a damaged item. The resolver
  should propose `issue_refund` instead.
- Refunds: the amount is no more than the item price and the refundable balance, and is at
  most $100. Reject anything over $100 and say to escalate with a high-priority ticket.
- Tickets: the description has enough context for a human to act on.

Then call `record_verdict` exactly once with approved=true/false and a short reason.
Finally, reply with one line: "APPROVED: <reason>" or "REJECTED: <reason>".
"""

RESPONDER_INSTRUCTION = """\
You are the customer-facing support agent for ShopSmart. Behind the scenes, a resolver
has investigated and a policy checker has reviewed any proposed action.

Verified customer: {customer_name?}
Resolver notes: {resolver_findings?}
Current proposal (with policy review): {proposal?}
Last executed action: {last_executed?}

Write the reply to the customer:
- Answer with the facts from the resolver notes. Don't invent anything.
- If the notes say verification is needed, ask for the customer's email.
- If there is a proposal:
  - Rejected by review: explain why in plain language and offer the alternative the
    checker suggested. Don't mention internal agents.
  - Approved, and the customer has NOT yet confirmed it: describe exactly what will
    happen (item, amount) and ask for an explicit yes.
  - Approved, and the resolver notes say "CUSTOMER CONFIRMED": call
    `execute_approved_action`, then confirm the result with the ids and next steps.
  - The resolver notes say "CUSTOMER DECLINED": call `discard_proposal` and ask what
    else you can help with.
- Never say an action is done unless `execute_approved_action` returned success.

Be friendly, concise, and specific. Use the customer's first name once verified.
"""


def decide_whether_to_review(callback_context):
    """Before the checker runs: is there a new, unreviewed proposal?

    The decision is stored in a "temp:" key (scoped to this invocation) because the
    checker makes several model calls (tools, then a final line), and only the
    starting state should decide whether to skip. Returning None (no content) lets
    the pipeline continue. Returning content here would end the whole invocation.
    """
    proposal = callback_context.state.get("proposal")
    callback_context.state["temp:skip_review"] = not proposal or proposal.get("review") is not None
    return None


def skip_model_if_nothing_to_review(callback_context, llm_request):
    """Replace the checker's LLM call with a canned reply when there's nothing to review."""
    if callback_context.state.get("temp:skip_review"):
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="No new proposal to review.")])
        )
    return None


resolver_agent = LlmAgent(
    model=MODEL,
    name="resolver_agent",
    description="Investigates the customer's request with read-only tools and proposes at most one action.",
    instruction=RESOLVER_INSTRUCTION,
    tools=[
        lookup_customer,
        list_customer_orders,
        get_order_details,
        track_shipment,
        search_products,
        search_policies,
        check_return_eligibility,
        propose_action,
    ],
    output_key="resolver_findings",
)

policy_checker_agent = LlmAgent(
    model=MODEL,
    name="policy_checker_agent",
    description="Approves or rejects a proposed action against store policy.",
    instruction=CHECKER_INSTRUCTION,
    tools=[get_order_details, check_return_eligibility, search_policies, record_verdict],
    # The checker judges the proposal on its merits, not on the conversation.
    include_contents="none",
    before_agent_callback=decide_whether_to_review,
    before_model_callback=skip_model_if_nothing_to_review,
)

responder_agent = LlmAgent(
    model=MODEL,
    name="responder_agent",
    description="Writes the customer-facing reply and executes approved, confirmed actions.",
    instruction=RESPONDER_INSTRUCTION,
    tools=[execute_approved_action, discard_proposal],
)

root_agent = SequentialAgent(
    name="support_pipeline",
    description="Customer support pipeline: resolve, policy-check, then respond.",
    sub_agents=[resolver_agent, policy_checker_agent, responder_agent],
)
