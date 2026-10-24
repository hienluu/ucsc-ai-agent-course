"""Loop: draft the reply, review it, revise until it passes (max 3 drafts).

    SequentialAgent support_loop
    ├── case_agent             investigates with tools; runs actions the customer confirmed last turn
    ├── LoopAgent reply_refiner    (max_iterations=3)
    │   ├── drafter_agent      writes or revises the reply             -> state["draft"]
    │   └── reviewer_agent     code fact-check + LLM rubric review;
    │                          approve_draft() exits the loop, request_revision() loops again
    └── support_agent          (custom BaseAgent, no LLM) publishes the approved reply,
                               or a safe fallback if no draft passed

The loop only *writes text*. Actions (tickets, cancellations) are offered for the
customer to confirm, and run on the next turn by case_agent, outside the loop.
So a loop that runs three times can't create three tickets.

Try Bob's multi-part complaint (see README): the watch is outside the return window
(the warranty path applies), the keyboard is backordered, and the gift card can't be
cancelled without cancelling the whole order.
"""

import json
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.events import Event, EventActions
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from shared.model import get_model
from shared.tools import ALL_TOOLS, search_policies

from .fact_check import check_draft

MODEL = get_model()
MAX_DRAFTS = 3


# ---------------------------------------------------------------------------
# Case agent: one pass of investigation (and confirmed actions) per turn
# ---------------------------------------------------------------------------

def start_turn(callback_context):
    """Reset per-turn loop state and save the customer's message for the loop agents."""
    user_content = callback_context.user_content
    callback_context.state["customer_message"] = "".join(
        p.text for p in (user_content.parts if user_content else []) if p.text
    )
    callback_context.state["draft"] = ""
    callback_context.state["review_feedback"] = ""
    callback_context.state["final_reply"] = ""
    callback_context.state["fact_check"] = ""
    callback_context.state["draft_count"] = 0
    return None


def log_tool_fact(tool, args, tool_context, tool_response):
    """Record every tool result so the fact checker can verify the draft against it.

    Kept for the whole session (not just this turn), so ids from earlier turns, like an
    RMA created yesterday, still count as grounded.
    """
    facts = list(tool_context.state.get("tool_facts") or [])
    facts.append(f"{tool.name}: {json.dumps(tool_response, default=str)}")
    tool_context.state["tool_facts"] = facts
    return None


CASE_INSTRUCTION = """\
You are the case agent for ShopSmart customer support. You investigate. You don't
write the reply to the customer; a drafter does that from your notes.

Verified customer (from session state): {customer_name?}

1. Verify the customer with `lookup_customer` when they give an email. Anything about
   a specific order needs verification.
2. Address EVERY request in the customer's latest message. For each one, gather the facts
   with tools: orders (`list_customer_orders`, `get_order_details`, `track_shipment`),
   eligibility (`check_return_eligibility`), products (`search_products`), and policies
   (`search_policies`). Look up the policy for each kind of request (returns, cancellations,
   warranty, ...).
3. Do NOT take actions unless the customer's latest message explicitly confirms an action
   offered in the previous reply ("yes, open the warranty claim"). Then run exactly that
   action with the write tools.
4. Write notes for the drafter: a numbered list of the customer's requests, and for each
   one, the facts (ids, amounts, dates, statuses), the policy-correct resolution, and any
   action to OFFER (not take). Then list any actions you took, with their ids.
"""

case_agent = LlmAgent(
    model=MODEL,
    name="case_agent",
    description="Investigates every part of the customer's request and runs actions they confirmed.",
    instruction=CASE_INSTRUCTION,
    tools=ALL_TOOLS,
    output_key="case_notes",
    before_agent_callback=start_turn,
    after_tool_callback=log_tool_fact,
)


# ---------------------------------------------------------------------------
# Refinement loop: drafter <-> reviewer
# ---------------------------------------------------------------------------

def count_draft(callback_context):
    callback_context.state["draft_count"] = (callback_context.state.get("draft_count") or 0) + 1
    return None


DRAFTER_INSTRUCTION = """\
You write ShopSmart's reply to the customer, based only on the case notes.

Customer's latest message:
{customer_message}

Case notes:
{case_notes}

Your previous draft (empty on the first attempt):
{draft?}

Reviewer feedback to fix (empty on the first attempt):
{review_feedback?}

Requirements:
- Answer every request in the customer's latest message, in order, specifically. For
  example, if an item is backordered, give the restock date.
- Use only ids, amounts, and dates that appear in the case notes. Don't do math to make
  new amounts, and don't add dates.
- Never say an action is done unless the case notes list it under actions taken. Offer
  actions and ask for an explicit yes.
- If something isn't possible under policy, say so plainly and offer the policy-correct
  alternative.
- If the customer is frustrated, acknowledge it once, briefly, then get to the answers.
- Under 200 words. Use the customer's first name.

If there is feedback, revise the previous draft to fix every point and keep what was fine.
Output ONLY the reply text, with no preamble.
"""

drafter_agent = LlmAgent(
    model=MODEL,
    name="drafter_agent",
    description="Writes or revises the customer reply from the case notes and reviewer feedback.",
    instruction=DRAFTER_INSTRUCTION,
    output_key="draft",
    before_agent_callback=count_draft,
    # The drafter works from the case notes. It doesn't need the transcript, and
    # leaving it out stops earlier drafts from piling up in the context.
    include_contents="none",
)


def run_fact_check(callback_context):
    """Before the reviewer's LLM runs, check the draft against the tool results in code."""
    issues = check_draft(
        callback_context.state.get("draft") or "",
        callback_context.state.get("tool_facts") or [],
    )
    callback_context.state["fact_check"] = (
        "PASSED: every id, amount, and date is backed by a tool result."
        if not issues
        else "FAILED:\n" + "\n".join(f"- {i}" for i in issues)
    )
    return None


def approve_draft(
    complete: bool,
    policy_correct: bool,
    honest: bool,
    tone: bool,
    tool_context: ToolContext,
) -> dict:
    """Approve the current draft and end the review loop.

    You must evaluate each rubric item. The draft is approved only if all are true.

    Args:
        complete: Every request in the customer's message is answered specifically.
        policy_correct: Every answer matches policy and the case notes.
        honest: No action is described as done unless it was; offered actions ask for a yes.
        tone: Frustration (if any) is acknowledged once; the reply is clear and concise.

    Returns:
        Confirmation, or an error naming what failed.
    """
    failed = [
        name for name, ok in
        {"complete": complete, "policy_correct": policy_correct, "honest": honest, "tone": tone}.items()
        if not ok
    ]
    if tool_context.state.get("fact_check", "").startswith("FAILED"):
        failed.append("grounded (automated fact check)")
    if failed:
        return {
            "status": "error",
            "error_message": f"Can't approve, failed: {', '.join(failed)}. Call request_revision instead.",
        }
    tool_context.state["final_reply"] = tool_context.state.get("draft", "")
    tool_context.actions.escalate = True  # exits the LoopAgent, not the outer pipeline
    tool_context.actions.skip_summarization = True
    return {"status": "success"}


def request_revision(issues: list[str], tool_context: ToolContext) -> dict:
    """Send the draft back to the drafter with specific issues to fix.

    Args:
        issues: Specific, actionable problems, e.g. "Doesn't answer where the keyboard is."

    Returns:
        Confirmation that the feedback was recorded.
    """
    feedback = list(issues)
    fact_check = tool_context.state.get("fact_check", "")
    if fact_check.startswith("FAILED"):
        feedback.append(fact_check)
    tool_context.state["review_feedback"] = "\n".join(f"- {i}" for i in feedback)
    return {"status": "success", "draft_number": tool_context.state.get("draft_count")}


REVIEWER_INSTRUCTION = """\
You are the quality reviewer for ShopSmart customer replies. Be strict: a reply goes
to the customer only if it passes every item. A revision is cheap; a wrong reply isn't.

Customer's latest message:
{customer_message}

Case notes (the facts and policy-correct resolutions):
{case_notes}

Draft reply:
{draft}

Automated fact check (code, always correct):
{fact_check}

Rubric (every item must pass):
1. Complete: every request in the customer's message is answered specifically, using the
   relevant facts from the case notes (e.g. "where is X?" gets a status AND an expected date
   if one is known, not just "processing").
2. Policy-correct: every policy claim or promise in the draft is backed by the policy text.
   Don't trust the case notes for this, since they can be wrong: call `search_policies` for
   each topic the draft touches (returns, refunds, warranty, cancellations, ...). Nothing is
   promised that policy doesn't state (e.g. a refund outside the return window, a refund as
   the outcome of a warranty claim, removing a single item from an order).
   Facts from tool results in the case notes (ids, amounts, dates, statuses, a ticket's
   expected response time) are NOT policy claims; the fact check covers those.
3. Honest: no action is described as done unless the case notes list it as taken. Offered
   actions ask for an explicit yes.
4. Grounded: the automated fact check PASSED.
5. Tone: if the customer sounds frustrated (e.g. "this is ridiculous"), the draft explicitly
   acknowledges that frustration once, without over-apologizing. A generic "thanks for
   reaching out" does not count. The reply is clear and under ~200 words.

Check each item against the customer's message, the case notes, and the draft. If every item
passes, call `approve_draft` with your per-item verdicts. Otherwise call `request_revision`
with a short list of specific issues, naming the rubric item for each. Then reply with one
line: "APPROVED" or "REVISE: <issues>".
"""

reviewer_agent = LlmAgent(
    model=MODEL,
    name="reviewer_agent",
    description="Reviews the draft with a code fact-check and an LLM rubric; approves or requests a revision.",
    instruction=REVIEWER_INSTRUCTION,
    tools=[search_policies, approve_draft, request_revision],
    before_agent_callback=run_fact_check,
    # Like the drafter, the reviewer judges from state: the customer's message, the case
    # notes, and the draft. It doesn't need the transcript.
    include_contents="none",
)

reply_refiner = LoopAgent(
    name="reply_refiner",
    description="Drafts and reviews the reply until it passes, up to 3 drafts.",
    sub_agents=[drafter_agent, reviewer_agent],
    max_iterations=MAX_DRAFTS,
)


# ---------------------------------------------------------------------------
# Publisher: a custom (non-LLM) agent that emits the final customer-facing reply
# ---------------------------------------------------------------------------

FALLBACK_REPLY = (
    "Thanks for your patience. I want to make sure everything I tell you is accurate, "
    "so I've flagged your conversation for a member of our support team to review. "
    "They'll follow up with you shortly."
)


class ReplyPublisher(BaseAgent):
    """Publishes the approved draft, or a safe fallback if no draft passed review."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        reply = state.get("final_reply")
        delta = {}
        if not reply:
            # The loop hit max_iterations without an approval. Don't send an unverified
            # draft; flag the session for a human instead.
            reply = FALLBACK_REPLY
            delta["needs_human_review"] = True
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=reply)]),
            actions=EventActions(state_delta=delta),
        )


support_agent = ReplyPublisher(
    name="support_agent",
    description="Publishes the reviewed reply to the customer.",
)

root_agent = SequentialAgent(
    name="support_loop",
    description="Customer support with a draft-review-revise loop on every reply.",
    sub_agents=[case_agent, reply_refiner, support_agent],
)
