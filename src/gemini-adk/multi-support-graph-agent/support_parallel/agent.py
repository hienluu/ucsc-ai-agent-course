"""Parallel research, as a graph: an LLM plans, code researches concurrently, an LLM replies.

    START -> remember_customer -> verify_identity -> load_orders -> planner
    planner -> ( research_order, research_policy, research_product )     fan-out
            -> JoinNode "research_done"                                  fan-in
            -> compile_brief -> responder -> record_reply

Compared with the ParallelAgent version in ../multi-support-agent:
- Fan-out and fan-in are edges: a tuple of nodes runs concurrently, and a JoinNode
  waits for all of them. No branch-isolation rules to remember.
- The researchers are plain functions, not LLMs. Looking up an order, a policy, or a
  product is deterministic, so it doesn't need a model. Only planning and replying do.
  That's faster, cheaper, and gives the same result every time.
- The planner returns a structured ResearchPlan (`output_schema`), and code validates it.
- A JoinNode waits for EVERY incoming branch, so researchers always run and return
  "not needed" when the plan doesn't involve them (instead of being skipped).

The mock tools answer instantly, so each researcher adds SIMULATED_API_LATENCY to stand
in for real backend calls (order service, CMS, catalog). See benchmark.py.
"""

import asyncio
import json
from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.workflow import JoinNode, Workflow
from pydantic import BaseModel, Field

from shared.graph import record_reply, remember_customer, verify_identity
from shared.model import get_model
from shared.tools import (
    cancel_order,
    check_return_eligibility,
    create_support_ticket,
    get_order_details,
    initiate_return,
    issue_refund,
    list_customer_orders,
    search_policies,
    search_products,
    track_shipment,
)

MODEL = get_model()
SIMULATED_API_LATENCY = 1.0  # seconds per researcher; mock tools are instant
NOT_NEEDED = "Not needed for this request."

PolicyTopic = Literal["returns", "refunds", "shipping", "cancellations", "warranty", "price_match", "loyalty"]


# ---------------------------------------------------------------------------
# Plan (LLM, structured output)
# ---------------------------------------------------------------------------

def load_orders(ctx: Context, node_input: str):
    """Give the planner the verified customer's orders, so it can resolve "my jacket" to an id."""
    orders = "Customer not verified."
    if ctx.state.get("customer_id"):
        result = list_customer_orders(tool_context=ctx)
        orders = json.dumps(result.get("orders", []))
    return Event(output=node_input, state={"customer_orders": orders})


class ResearchPlan(BaseModel):
    intent: str = Field(description="What the customer wants this turn, in a short phrase.")
    order_id: str = Field("", description="Order to research, from the customer's orders. Empty if none.")
    check_return_eligibility: bool = Field(False, description="True for returns, refunds, or damaged items.")
    policy_topics: list[PolicyTopic] = Field(default_factory=list, description="Policies relevant to the request.")
    product_query: str = Field("", description="Product search keywords. Empty if no product question.")


planner = LlmAgent(
    model=MODEL,
    name="planner",
    description="Decides what to research for the customer's latest message.",
    instruction="""\
You plan the research for ShopSmart customer support. You don't reply to the customer.

Verified customer: {customer_name?} (loyalty tier: {customer_tier?})
The customer's orders: {customer_orders?}

Conversation so far (latest last):
{conversation}

Plan the research for the customer's LATEST message:
- order_id: the order it's about. If they describe it ("my jacket", "my last order"), pick the
  matching id from their orders. Leave empty if not verified or no order is involved.
- check_return_eligibility: true for returns, refunds, or damaged items.
- policy_topics: every policy that matters. Most order problems involve one.
- product_query: keywords if they ask about a product, stock, or warranty.
If the message just confirms or declines something, describe that in intent and leave the
rest empty.
""",
    output_schema=ResearchPlan,
    output_key="plan",
)


# ---------------------------------------------------------------------------
# Research (code, runs concurrently)
# ---------------------------------------------------------------------------

async def research_order(ctx: Context, node_input: dict) -> str:
    plan = ResearchPlan(**node_input)
    if not plan.order_id:
        return NOT_NEEDED
    await asyncio.sleep(SIMULATED_API_LATENCY)
    details = get_order_details(plan.order_id, tool_context=ctx)
    findings = {"order": details}
    if details.get("status") == "success":
        order = details["order"]
        if order["tracking_number"]:
            findings["tracking"] = track_shipment(plan.order_id, tool_context=ctx)
        if plan.check_return_eligibility:
            findings["return_eligibility"] = {
                item["sku"]: check_return_eligibility(plan.order_id, item["sku"], tool_context=ctx)
                for item in order["items"]
            }
    return json.dumps(findings, default=str)


async def research_policy(node_input: dict) -> str:
    plan = ResearchPlan(**node_input)
    if not plan.policy_topics:
        return NOT_NEEDED
    await asyncio.sleep(SIMULATED_API_LATENCY)
    policies = {}
    for topic in dict.fromkeys(plan.policy_topics):  # dedupe, keep order
        policies.update(search_policies(topic).get("policies", {}))
    return json.dumps(policies)


async def research_product(node_input: dict) -> str:
    plan = ResearchPlan(**node_input)
    if not plan.product_query:
        return NOT_NEEDED
    await asyncio.sleep(SIMULATED_API_LATENCY)
    return json.dumps(search_products(plan.product_query))


research_done = JoinNode(name="research_done")


def compile_brief(ctx: Context, node_input: dict):
    """Turn the JoinNode's {node_name: output} dict into one brief for the responder."""
    plan = ctx.state.get("plan") or {}
    brief = "\n".join([
        f"Intent: {plan.get('intent', '')}",
        f"Order research: {node_input.get('research_order', NOT_NEEDED)}",
        f"Policy research: {node_input.get('research_policy', NOT_NEEDED)}",
        f"Product research: {node_input.get('research_product', NOT_NEEDED)}",
    ])
    return Event(output=brief, state={"research_brief": brief})


# ---------------------------------------------------------------------------
# Respond (LLM)
# ---------------------------------------------------------------------------

responder = LlmAgent(
    model=MODEL,
    name="responder",
    description="Combines the research into one reply and runs actions the customer confirmed.",
    instruction="""\
You are the customer-facing support agent for ShopSmart.

Verified customer: {customer_name?} (loyalty tier: {customer_tier?})
Identity note: {identity_note?}

Conversation so far (latest last):
{conversation}

Research gathered for this turn:
{research_brief}

Write ONE coherent reply that combines the research. Use only these facts. Don't mention
research or internal steps.
- If the request needs a specific order and the customer isn't verified, ask for their email.
- Before ANY write action (`initiate_return`, `cancel_order`, `issue_refund`), state exactly
  what you'll do (item, amount) and get an explicit "yes". When the customer's latest message
  confirms something you offered, do it.
- Returns: only if the research says the item is eligible; quote the estimated refund.
- Damaged-on-arrival items get a full refund with no return needed. Up to $100, use
  `issue_refund`. Over $100, or anything you can't resolve, use `create_support_ticket`
  (after confirming) and give the ticket id and response time.
- Confirm outcomes with ids (RMA, refund id, ticket id) and what happens next.
Be friendly, concise, and specific. Use the customer's first name.
""",
    tools=[get_order_details, initiate_return, cancel_order, issue_refund, create_support_ticket],
)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

RESEARCHERS = (research_order, research_policy, research_product)

root_agent = Workflow(
    name="support_parallel",
    description="Plan, research concurrently in code, then respond.",
    edges=[
        ("START", remember_customer, verify_identity, load_orders, planner),
        (planner, RESEARCHERS),
        (RESEARCHERS, research_done),
        (research_done, compile_brief, responder, record_reply),
    ],
)

app = App(name="support_parallel", root_agent=root_agent)
