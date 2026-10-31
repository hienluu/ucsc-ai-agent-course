"""Routing, as a graph: an LLM classifies, code routes.

    START -> remember_customer -> verify_identity -> classify -> dispatch
    dispatch --"order"-----------> order_agent ----------\\
             --"returns_refunds"-> returns_refunds_agent -+
             --"product_policy"--> product_policy_agent --+-> record_reply
             --"escalation"------> escalation_agent -----/
             --"verify"----------> ask_for_email
             --"unclear"---------> ask_to_clarify

Compared with the LLM-driven `transfer_to_agent` routing in ../multi-support-agent:
- The routing table is the graph's edges, so it's visible and testable.
- `classify` is an LLM with a structured `output_schema`. It only *labels* the message;
  `dispatch` (plain code) decides where it goes.
- Verification is enforced by the graph: order-related routes go to `ask_for_email`
  until `verify_identity` (code, not an LLM) has verified the customer.
- Follow-ups ("yes", "the second one") are sticky in code: `dispatch` sends them to
  the specialist that handled the previous turn (`last_route` in state).
- Specialists can't hand off mid-turn. Each turn is classified once.
"""

from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.workflow import Workflow
from pydantic import BaseModel, Field

from shared.graph import record_reply, remember_customer, reply, verify_identity
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

SPECIALIST_ROUTES = ("order", "returns_refunds", "product_policy", "escalation")
NEEDS_VERIFICATION = {"order", "returns_refunds", "escalation"}


# ---------------------------------------------------------------------------
# Classify (LLM, structured output) -> dispatch (code)
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    route: Literal["order", "returns_refunds", "product_policy", "escalation", "follow_up", "unclear"]
    reason: str = Field(description="One short sentence explaining the choice.")


classify = LlmAgent(
    model=MODEL,
    name="classify",
    description="Labels the customer's latest message with a support category.",
    instruction="""\
You label customer support messages for ShopSmart. You don't reply to the customer.

Conversation so far (latest last):
{conversation}

Label the customer's LATEST message:
- order: order status, tracking, delivery delays, cancelling an order
- returns_refunds: returns, refunds, damaged or defective items
- product_policy: product info, stock, prices, store policies (shipping, returns policy,
  price match, loyalty)
- escalation: wants a human, complaint, very upset, legal threat, safety issue, warranty claim
- follow_up: only answers or continues the agent's previous message ("yes", "no", "the
  second one", an email address the agent asked for)
- unclear: none of the above can be determined
""",
    output_schema=RouteDecision,
)


def dispatch(ctx: Context, node_input: dict):
    """Turn the classifier's label into a graph route. All routing rules live here."""
    route = node_input["route"]
    if route == "follow_up":
        route = ctx.state.get("last_route") or "unclear"
    if route in NEEDS_VERIFICATION and not ctx.state.get("customer_id"):
        # Remember where to go once the customer gives their email.
        return Event(output=ctx.state.get("customer_message", ""), route="verify", state={"last_route": route})
    return Event(output=ctx.state.get("customer_message", ""), route=route, state={"last_route": route})


def ask_for_email(ctx: Context, node_input: str):
    note = ctx.state.get("identity_note")
    prefix = f"{note} " if note else ""
    return reply(ctx, f"{prefix}To look into your order, could you share the email address on your account?")


def ask_to_clarify(ctx: Context, node_input: str):
    return reply(
        ctx,
        "Happy to help! Is this about an order (status, tracking, cancelling), a return or "
        "refund, or a question about a product or our policies?",
        last_route="",
    )


# ---------------------------------------------------------------------------
# Specialists (single_turn LLM nodes with narrow tool sets)
# ---------------------------------------------------------------------------

COMMON_RULES = """\
Verified customer: {customer_name?}

Conversation so far (latest last):
{conversation}

Shared rules:
- Use tools for facts. Never invent order details, tracking events, prices, or policies.
  If a tool returns an error, explain it plainly and offer the next best option.
- Before ANY write action, state exactly what you will do (item, amount) and get an
  explicit "yes". If the customer's latest message is that "yes" to your previous offer,
  do it now.
- If the customer asks about something outside your area, answer what you can and tell
  them you can help with the rest next.
- Be friendly, concise, and specific. Use the customer's first name. Confirm outcomes with
  ids (RMA, refund id, ticket id) and what happens next.
"""

order_agent = LlmAgent(
    model=MODEL,
    name="order_agent",
    description="Order status, tracking, delays, and cancellations.",
    instruction=COMMON_RULES + """
You are ShopSmart's order specialist.
- Use `list_customer_orders` when the customer doesn't give an order id.
- For delayed shipments, give the reason from the tracking events and the new estimate.
- For items that haven't shipped, check `get_order_details` for backorders and restock dates.
- Orders can only be cancelled while 'processing', and only as a whole order.
""",
    tools=[list_customer_orders, get_order_details, track_shipment, cancel_order],
)

returns_refunds_agent = LlmAgent(
    model=MODEL,
    name="returns_refunds_agent",
    description="Returns, eligibility, refunds, damaged items.",
    instruction=COMMON_RULES + """
You are ShopSmart's returns and refunds specialist.
- Change-of-mind returns: call `check_return_eligibility` before `initiate_return`, and quote
  the estimated refund (including any return shipping fee).
- Damaged-on-arrival items are NOT a normal return: full refund of the item price, no return
  needed, no fee. Up to $100, use `issue_refund`. Over $100, explain it needs supervisor
  approval and offer to escalate (the escalation specialist will open the ticket).
- Outside the return window: say so and mention the warranty if the product has one.
""",
    tools=[list_customer_orders, get_order_details, check_return_eligibility, initiate_return, issue_refund],
)

product_policy_agent = LlmAgent(
    model=MODEL,
    name="product_policy_agent",
    description="Product and store-policy questions.",
    instruction=COMMON_RULES + """
You are ShopSmart's product and policy specialist. You only answer questions.
- Use `search_products` for products (give the restock date if out of stock).
- Use `search_policies` for policies and answer only from what it returns.
""",
    tools=[search_products, search_policies],
)

escalation_agent = LlmAgent(
    model=MODEL,
    name="escalation_agent",
    description="Hands the case to a human with a support ticket.",
    instruction=COMMON_RULES + """
You are ShopSmart's escalation specialist. Hand the case to a human with everything they
need, so the customer doesn't have to repeat themselves.
- Write the ticket from the conversation: what the customer wants, order and item, amounts,
  and what was already tried. Use `get_order_details` if needed.
- Priority: "urgent" for legal, safety, or very upset customers; "high" for refunds over $100;
  otherwise "normal".
- Confirm the summary with the customer before creating the ticket, then give the ticket id
  and expected response time.
""",
    tools=[get_order_details, create_support_ticket],
)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

root_agent = Workflow(
    name="support_routing",
    description="Classify each customer message and route it to a specialist, with routing rules in code.",
    edges=[
        ("START", remember_customer, verify_identity, classify, dispatch),
        (
            dispatch,
            {
                "order": order_agent,
                "returns_refunds": returns_refunds_agent,
                "product_policy": product_policy_agent,
                "escalation": escalation_agent,
                "verify": ask_for_email,
                "unclear": ask_to_clarify,
            },
        ),
        ((order_agent, returns_refunds_agent, product_policy_agent, escalation_agent), record_reply),
    ],
)

app = App(name="support_routing", root_agent=root_agent)
