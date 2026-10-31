"""Draft -> check -> review -> revise, as an explicit cycle in the graph.

    START -> remember_customer -> verify_identity -> case_agent -> start_drafting -> drafter
    drafter -> fact_check
    fact_check --"passed"--> reviewer -> decide
               --"failed"----------------> decide      (skips the LLM review)
    decide --"approved"--> publish
           --"revise"----> drafter                     (the loop: a routed back-edge)
           --"give_up"---> fallback                    (after MAX_DRAFTS)

Compared with the LoopAgent version in ../multi-support-agent:
- The loop is just an edge back to `drafter`. ADK requires every cycle to have a routed
  edge, so the exit condition is always explicit.
- `decide` is code: it approves only if every rubric item passed, counts drafts, and
  picks revise / give up. The LLM reviewer only fills in a structured `Review`.
- The code fact check is its own node. When it fails, the graph skips the LLM reviewer
  and goes straight to revision, so invented details never cost a review call.
- `publish` and `fallback` are plain function nodes. No custom BaseAgent needed.

Try Bob's multi-part complaint (see README).
"""

import json

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.workflow import Workflow
from pydantic import BaseModel, Field

from shared import mock_data as db
from shared.graph import content_text, remember_customer, reply, verify_identity
from shared.model import get_model
from shared.tools import ALL_TOOLS

from .fact_check import check_draft

MODEL = get_model()
MAX_DRAFTS = 3
POLICIES = "\n".join(f"- {topic}: {text}" for topic, text in db.KNOWLEDGE_BASE.items())


# ---------------------------------------------------------------------------
# Case agent (LLM): one pass of investigation per turn
# ---------------------------------------------------------------------------

def log_tool_fact(tool, args, tool_context, tool_response):
    """Record every tool result, so the fact check can verify drafts against them."""
    facts = list(tool_context.state.get("tool_facts") or [])
    facts.append(f"{tool.name}: {json.dumps(tool_response, default=str)}")
    tool_context.state["tool_facts"] = facts
    return None


case_agent = LlmAgent(
    model=MODEL,
    name="case_agent",
    description="Investigates every part of the request and runs actions the customer confirmed.",
    instruction="""\
You are the case agent for ShopSmart customer support. You investigate. You don't write
the reply to the customer; a drafter does that from your notes.

Verified customer: {customer_name?} (loyalty tier: {customer_tier?})
Identity note: {identity_note?}

Conversation so far (latest last):
{conversation}

1. Anything about a specific order needs a verified customer. If there isn't one, note that
   the reply must ask for their email.
2. Address EVERY request in the customer's latest message. For each one, gather the facts with
   tools: orders (`list_customer_orders`, `get_order_details`, `track_shipment`), eligibility
   (`check_return_eligibility`), products (`search_products`), and policies (`search_policies`).
3. Do NOT take actions unless the customer's latest message explicitly confirms an action
   offered in the previous reply ("yes, open the warranty claim"). Then run exactly that action.
4. Write notes for the drafter: a numbered list of the customer's requests, and for each one,
   the facts (ids, amounts, dates, statuses), the policy-correct resolution, and any action to
   OFFER (not take). Then list any actions you took, with their ids.
""",
    tools=ALL_TOOLS,
    output_key="case_notes",
    after_tool_callback=log_tool_fact,
)


def start_drafting(ctx: Context, node_input):
    """Reset the loop's state for this turn."""
    return Event(
        output="Write the first draft.",
        state={"draft_count": 0, "draft": "", "review_feedback": "", "fact_check": ""},
    )


# ---------------------------------------------------------------------------
# The loop: drafter (LLM) -> fact_check (code) -> reviewer (LLM) -> decide (code)
# ---------------------------------------------------------------------------

drafter = LlmAgent(
    model=MODEL,
    name="drafter",
    description="Writes or revises the customer reply.",
    instruction="""\
You write ShopSmart's reply to the customer, based only on the case notes.

Customer: {customer_name?}

Customer's latest message:
{customer_message}

Case notes:
{case_notes}

Your previous draft (empty on the first attempt):
{draft?}

Reviewer feedback to fix (empty on the first attempt):
{review_feedback?}

Requirements:
- Answer every request in the customer's latest message, in order, specifically. For example,
  if an item is backordered, give the restock date.
- Use only ids, amounts, and dates that appear in the case notes. Don't do math to make new
  amounts, and don't add dates.
- Never say an action is done unless the case notes list it under actions taken. Offer actions
  and ask for an explicit yes.
- If something isn't possible under policy, say so plainly and offer the policy-correct
  alternative.
- If the customer is frustrated, acknowledge it once, briefly, then get to the answers.
- Don't promise future follow-ups; describe what has happened or what you can do now.
- Under 200 words. Use the customer's first name.

If there is feedback, revise the previous draft to fix every point and keep what was fine.
Output ONLY the reply text, with no preamble.
""",
)


def fact_check(ctx: Context, node_input):
    """Code check: every id, $ amount, and date in the draft must come from a tool result."""
    draft = content_text(node_input)
    count = (ctx.state.get("draft_count") or 0) + 1
    issues = check_draft(draft, ctx.state.get("tool_facts") or [])
    state = {"draft": draft, "draft_count": count, "fact_check": "PASSED" if not issues else "FAILED"}
    if issues:
        return Event(output={"checked_by": "fact_check", "issues": issues}, route="failed", state=state)
    return Event(output=draft, route="passed", state=state)


class Review(BaseModel):
    complete: bool = Field(description="Every request in the customer's message is answered specifically.")
    policy_correct: bool = Field(description="Every policy claim or promise is backed by the store policies.")
    honest: bool = Field(description="No action is described as done unless the case notes list it as taken.")
    tone: bool = Field(description="Frustration acknowledged once (if any); clear; under ~200 words.")
    issues: list[str] = Field(default_factory=list, description="Specific, actionable problems. Empty if all pass.")


reviewer = LlmAgent(
    model=MODEL,
    name="reviewer",
    description="Reviews the draft against the rubric.",
    instruction=f"""\
You are the quality reviewer for ShopSmart customer replies. Be strict: a reply goes to the
customer only if it passes every item. A revision is cheap; a wrong reply isn't.

Customer's latest message:
{{customer_message}}

Case notes (facts from tools):
{{case_notes}}

Store policies (the source of truth for policy claims):
{POLICIES}
- Refunds over $100 can't be issued directly; they need escalation with a support ticket.

The draft to review is the input message.

Rubric:
- complete: every request is answered specifically using the facts (e.g. "where is X?" gets a
  status AND an expected date if one is known).
- policy_correct: every policy claim or promise is backed by the store policies above. Nothing
  is promised that policy doesn't state (e.g. a refund outside the return window, a refund as
  the outcome of a warranty claim, removing a single item from an order, or a promise to
  follow up later). When suggesting a fix, don't suggest new promises. Facts from tool
  results (ids, amounts, dates, a ticket's response time) are NOT policy claims.
- honest: no action is described as done unless the case notes list it as taken. Offered
  actions ask for an explicit yes.
- tone: if the customer sounds frustrated ("this is ridiculous"), the draft explicitly
  acknowledges it once without over-apologizing ("thanks for reaching out" doesn't count).
  Clear and under ~200 words.
List specific issues for every failed item, naming the item.
""",
    output_schema=Review,
)


def decide(ctx: Context, node_input: dict):
    """Approve, revise, or give up. The loop's exit condition lives here, in code."""
    if node_input.get("checked_by") == "fact_check":
        approved = False
        issues = [f"Grounding: {i}" for i in node_input["issues"]]
    else:
        review = Review(**node_input)
        approved = review.complete and review.policy_correct and review.honest and review.tone
        issues = review.issues or ["The reviewer failed an item without explaining; re-check every rubric item."]

    draft = ctx.state.get("draft", "")
    if approved:
        return Event(output=draft, route="approved")
    feedback = "\n".join(f"- {i}" for i in issues)
    if ctx.state.get("draft_count", 0) >= MAX_DRAFTS:
        return Event(output=feedback, route="give_up", state={"review_feedback": feedback})
    return Event(output=f"Revise the draft. Fix:\n{feedback}", route="revise", state={"review_feedback": feedback})


# ---------------------------------------------------------------------------
# Exits (code)
# ---------------------------------------------------------------------------

FALLBACK_REPLY = (
    "Thanks for your patience. I want to make sure everything I tell you is accurate, so I've "
    "flagged your conversation for a member of our support team to review. They'll follow up "
    "with you shortly."
)


def publish(ctx: Context, node_input: str):
    return reply(ctx, node_input)


def fallback(ctx: Context, node_input: str):
    # No draft passed: don't send an unverified one. Flag the session for a human instead.
    return reply(ctx, FALLBACK_REPLY, needs_human_review=True)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

root_agent = Workflow(
    name="support_loop",
    description="Investigate, then draft, fact-check, and review the reply until it passes.",
    edges=[
        ("START", remember_customer, verify_identity, case_agent, start_drafting, drafter, fact_check),
        (fact_check, {"passed": reviewer, "failed": decide}),
        (reviewer, decide),
        (decide, {"approved": publish, "revise": drafter, "give_up": fallback}),
    ],
)

app = App(name="support_loop", root_agent=root_agent)
