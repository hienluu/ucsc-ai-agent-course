"""Propose -> check -> confirm -> execute, as a graph with a human-in-the-loop pause.

    START -> remember_customer -> verify_identity -> resolver -> after_resolve
    after_resolve --"no_action"--> responder
                  --"review"-----> policy_checker -> after_review
    after_review  --"rejected"---> responder
                  --"approved"---> confirm_and_execute   (pauses: RequestInput)
    confirm_and_execute ---------> responder -> record_reply

Compared with the SequentialAgent version in ../multi-support-agent:
- Branching is explicit: the checker only runs when there's a proposal, and the
  confirmation step only when the checker approved. No skip callbacks needed.
- Confirmation is a real pause. `confirm_and_execute` yields a RequestInput and the
  workflow stops until the customer answers. No LLM decides whether "yes" means yes.
- Execution is code. `confirm_and_execute` runs the stored, approved proposal itself;
  no LLM ever holds the write tools.
- The responder only writes the reply. It has no tools.

Needs a resumable App (`ResumabilityConfig(is_resumable=True)`) so the paused
workflow can continue when the answer arrives.
"""

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import FunctionNode, Workflow

from shared.graph import record_reply, remember_customer, verify_identity
from shared.model import get_model
from shared.tools import (
    check_return_eligibility,
    get_order_details,
    list_customer_orders,
    search_policies,
    search_products,
    track_shipment,
)

from .workflow_tools import ACTIONS, propose_action, record_verdict

MODEL = get_model()


def start_turn(ctx: Context, node_input):
    """Clear last turn's proposal and results, then pass the message on."""
    return Event(
        output=node_input,
        state={"proposal": None, "action_result": "", "resolver_notes": ""},
    )


# ---------------------------------------------------------------------------
# Resolver (LLM, read-only tools + propose_action)
# ---------------------------------------------------------------------------

resolver = LlmAgent(
    model=MODEL,
    name="resolver",
    description="Investigates the request with read-only tools and proposes at most one action.",
    instruction="""\
You are the resolver for ShopSmart customer support. You work behind the scenes; another
agent writes the reply.

Verified customer: {customer_name?}
Identity note: {identity_note?}

Conversation so far (latest last):
{conversation}

1. Work out what the customer wants in their latest message.
2. Gather facts with your read-only tools. Never invent facts. Anything about a specific
   order needs a verified customer; if there isn't one, note that the reply must ask for
   their email.
3. If they want something changed (return, cancel, refund, escalate to a human), call
   `propose_action` with exactly ONE action. You cannot execute it; the customer will be
   asked to confirm.
   - Returns: call `check_return_eligibility` first; put the estimated refund in the summary.
   - Damaged on arrival: propose `issue_refund` for the full item price (no return needed).
   - Human, complaint, legal or safety issue, or a refund the checker said to escalate:
     propose `create_support_ticket` with full context.
4. Output short notes for the reply writer: facts found (ids, dates, amounts, statuses),
   what you proposed, and anything still missing.
""",
    tools=[
        list_customer_orders,
        get_order_details,
        track_shipment,
        search_products,
        search_policies,
        check_return_eligibility,
        propose_action,
    ],
    output_key="resolver_notes",
)


def after_resolve(ctx: Context, node_input):
    proposal = ctx.state.get("proposal")
    return Event(output=proposal or {}, route="review" if proposal else "no_action")


# ---------------------------------------------------------------------------
# Policy checker (LLM, judges the proposal only)
# ---------------------------------------------------------------------------

policy_checker = LlmAgent(
    model=MODEL,
    name="policy_checker",
    description="Approves or rejects the proposed action against store policy.",
    instruction="""\
You are the policy checker for ShopSmart. You review ONE proposed action. You don't see
the conversation; judge only the proposal and the facts you look up yourself.

Proposal:
{proposal}

Check with your tools that:
- The order exists and belongs to the verified customer (`get_order_details`).
- The action matches policy (`search_policies`: returns, refunds, cancellations, warranty).
- Returns: the item is eligible (`check_return_eligibility`).
- Damaged-on-arrival items get a full refund with no return and no fee, so reject an
  `initiate_return` that charges a fee for a damaged item.
- `issue_refund` (a direct refund, no return): at most the item price and refundable balance,
  and at most $100. Reject anything over $100 and say to escalate with a high-priority ticket.
  The $100 limit applies ONLY to `issue_refund`. A return's refund (`initiate_return`) is
  issued after the item comes back and has no amount limit.
- Tickets: enough context for a human to act on.

Call `record_verdict` exactly once, then reply with one line: "APPROVED: <reason>" or
"REJECTED: <reason>".
""",
    tools=[get_order_details, check_return_eligibility, search_policies, record_verdict],
)


def after_review(ctx: Context, node_input):
    proposal = ctx.state.get("proposal") or {}
    review = proposal.get("review") or {}
    return Event(output=proposal, route="approved" if review.get("approved") else "rejected")


# ---------------------------------------------------------------------------
# Human-in-the-loop confirmation + execution (code)
# ---------------------------------------------------------------------------

YES = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "go ahead", "please do", "confirm"}


async def confirm_and_execute(ctx: Context, node_input: dict):
    """Pause for the customer's yes/no, then run exactly the approved proposal."""
    proposal = ctx.state.get("proposal")
    interrupt_id = f"confirm_{proposal['id']}"

    if interrupt_id not in (ctx.resume_inputs or {}):
        yield RequestInput(
            interrupt_id=interrupt_id,
            message=f"{proposal['summary']} Shall I go ahead? (yes/no)",
            payload=proposal,
        )
        return

    raw_answer = str(ctx.resume_inputs[interrupt_id]).strip()
    answer = raw_answer.lower().rstrip("!.")
    if answer not in YES and not answer.startswith("yes"):
        yield Event(
            output="declined",
            state={
                "proposal": None,
                "action_result": (
                    f"NOT DONE. We asked the customer to confirm: {proposal['summary']} "
                    f"They answered {raw_answer!r}, which is not a yes, so nothing was changed."
                ),
            },
        )
        return

    func, _ = ACTIONS[proposal["action"]]
    result = func(**proposal["args"], tool_context=ctx)
    outcome = "Done" if result.get("status") == "success" else "Failed"
    yield Event(
        output=result,
        state={"proposal": None, "action_result": f"{outcome}: {proposal['summary']} Result: {result}"},
    )


# ---------------------------------------------------------------------------
# Responder (LLM, no tools)
# ---------------------------------------------------------------------------

responder = LlmAgent(
    model=MODEL,
    name="responder",
    description="Writes the customer-facing reply.",
    instruction="""\
You write ShopSmart's reply to the customer. You have no tools: use only the facts below.

Verified customer: {customer_name?}
Identity note: {identity_note?}

Conversation so far (latest last):
{conversation}

Resolver notes: {resolver_notes?}
Proposal and policy review (if any): {proposal?}
Action result (if any): {action_result?}

- Answer with the facts from the notes. Don't invent anything.
- If the notes say verification is needed, ask for the customer's email.
- If the proposal was rejected, explain why in plain language and OFFER the alternative the
  review suggested (e.g. "Would you like me to escalate this to our support team?"). Don't
  say you will do it, and don't mention internal agents or reviews.
- Never promise future follow-ups; only describe what has happened or what you can offer now.
- Never say an action is done, started, submitted, or escalated unless the action result
  starts with "Done". A proposal alone has NOT happened.
- If there is an action result, confirm it with the ids and next steps. If it says NOT DONE,
  say nothing was changed, and respond to what the customer said instead.
- Be friendly, concise, and specific. Use the customer's first name.
""",
)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

confirm_node = FunctionNode(func=confirm_and_execute, rerun_on_resume=True)

root_agent = Workflow(
    name="support_sequential",
    description="Resolve, policy-check, ask the customer, then execute in code.",
    edges=[
        ("START", remember_customer, verify_identity, start_turn, resolver, after_resolve),
        (after_resolve, {"no_action": responder, "review": policy_checker}),
        (policy_checker, after_review),
        (after_review, {"rejected": responder, "approved": confirm_node}),
        (confirm_node, responder),
        (responder, record_reply),
    ],
)

app = App(
    name="support_sequential",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
