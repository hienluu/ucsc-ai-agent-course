"""Parallel: intake -> fan out research concurrently -> one responder.

    SequentialAgent support_parallel
    ├── intake_agent             verifies the customer, resolves the order, records what to research
    ├── ParallelAgent research_team          (all three run at the same time)
    │   ├── order_researcher       order details, tracking, return eligibility -> "order_research"
    │   ├── policy_researcher      store policies                             -> "policy_research"
    │   └── product_researcher     catalog, stock, restock dates              -> "product_research"
    └── responder_agent          combines the research, replies, runs confirmed write actions

ParallelAgent runs each researcher in its own branch: they share session state
but don't see each other's messages, so each writes to its own output_key and
the responder reads all of them.

The intake agent writes a research plan to state (record_intake). A researcher
whose part of the plan is empty skips its model call, so a pure policy question
doesn't pay for an order lookup.
"""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from shared.model import get_model
from shared.tools import (
    check_return_eligibility,
    create_support_ticket,
    get_order_details,
    initiate_return,
    issue_refund,
    cancel_order,
    list_customer_orders,
    lookup_customer,
    search_policies,
    search_products,
    track_shipment,
)

MODEL = get_model()

INTAKE_KEYS = ("intake_intent", "intake_order_id", "intake_policy_query", "intake_product_query")
OUTPUT_KEYS = ("order_research", "policy_research", "product_research")


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------

def record_intake(
    intent: str,
    tool_context: ToolContext,
    order_id: str = "",
    policy_query: str = "",
    product_query: str = "",
) -> dict:
    """Record the research plan for this turn. Leave a field empty to skip that research.

    Args:
        intent: One short phrase describing what the customer wants, e.g. "return headphones",
            "where is my order", "confirming the pending return".
        order_id: The order to research, if the request is about a specific order
            and the customer is verified.
        policy_query: Keywords for the store policy search, e.g. "return window damaged".
        product_query: Keywords for the product search, e.g. "TypePro keyboard".

    Returns:
        The recorded plan.
    """
    plan = {
        "intake_intent": intent,
        "intake_order_id": order_id.strip().upper(),
        "intake_policy_query": policy_query.strip(),
        "intake_product_query": product_query.strip(),
    }
    for key, value in plan.items():
        tool_context.state[key] = value
    return {"status": "success", "plan": plan}


def reset_turn_state(callback_context):
    """Clear last turn's plan and research so stale results never leak into this turn."""
    for key in INTAKE_KEYS + OUTPUT_KEYS:
        callback_context.state[key] = ""
    return None


INTAKE_INSTRUCTION = """\
You are the intake agent for ShopSmart customer support. You don't answer the
customer. You decide what needs researching, and specialists run in parallel after you.

Verified customer (from session state): {customer_name?}

1. If the customer gave an email and isn't verified yet, call `lookup_customer`.
2. If the request is about a specific order and the customer is verified, work out the
   order id. Use `list_customer_orders` if they described it ("my jacket", "my last
   order") instead of giving an id. Don't guess.
3. Call `record_intake` exactly once with:
   - intent: what the customer wants this turn
   - order_id: the order to research (empty if none, or not verified)
   - policy_query: keywords if a store policy is relevant (returns, refunds, shipping,
     cancellations, warranty, price match, loyalty). Most order problems involve one.
   - product_query: keywords if they ask about a product, stock, or warranty
   If the customer is only confirming or declining something, record that intent and
   leave the rest empty.
4. Reply with one line summarizing the plan.
"""

intake_agent = LlmAgent(
    model=MODEL,
    name="intake_agent",
    description="Verifies the customer, resolves the order, and plans the research.",
    instruction=INTAKE_INSTRUCTION,
    tools=[lookup_customer, list_customer_orders, record_intake],
    before_agent_callback=reset_turn_state,
)


# ---------------------------------------------------------------------------
# Parallel researchers
# ---------------------------------------------------------------------------

def skip_unless(state_key: str):
    """Build a before_model_callback that skips the LLM call when `state_key` is empty."""

    def callback(callback_context, llm_request):
        if not callback_context.state.get(state_key):
            return LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="Not needed this turn.")])
            )
        return None

    return callback


RESEARCHER_RULES = """\
You are a research specialist. You don't talk to the customer. Your notes go to the
agent that writes the reply. Use your tools and report only facts they return,
including ids, dates, amounts, and statuses. Be brief: a few bullet points.
Customer intent: {intake_intent?}
"""

order_researcher = LlmAgent(
    model=MODEL,
    name="order_researcher",
    description="Researches one order: details, shipment tracking, and return eligibility.",
    instruction=RESEARCHER_RULES
    + """
Order to research: {intake_order_id?}

- Always call `get_order_details`.
- If the order has shipped or been delivered, call `track_shipment`.
- If the intent is a return, refund, or damaged item, call `check_return_eligibility`
  for the relevant item(s).
""",
    tools=[get_order_details, track_shipment, check_return_eligibility],
    output_key="order_research",
    before_model_callback=skip_unless("intake_order_id"),
)

policy_researcher = LlmAgent(
    model=MODEL,
    name="policy_researcher",
    description="Looks up the store policies relevant to the request.",
    instruction=RESEARCHER_RULES
    + """
Policy question: {intake_policy_query?}

Call `search_policies` (more than once if several topics apply) and quote the parts
that matter for this customer's intent.
""",
    tools=[search_policies],
    output_key="policy_research",
    before_model_callback=skip_unless("intake_policy_query"),
)

product_researcher = LlmAgent(
    model=MODEL,
    name="product_researcher",
    description="Looks up products: price, stock, restock date, warranty.",
    instruction=RESEARCHER_RULES
    + """
Product question: {intake_product_query?}

Call `search_products` and report the matching products' price, stock, restock date,
and warranty.
""",
    tools=[search_products],
    output_key="product_research",
    before_model_callback=skip_unless("intake_product_query"),
)

research_team = ParallelAgent(
    name="research_team",
    description="Runs the order, policy, and product researchers concurrently.",
    sub_agents=[order_researcher, policy_researcher, product_researcher],
)


# ---------------------------------------------------------------------------
# Responder
# ---------------------------------------------------------------------------

RESPONDER_INSTRUCTION = """\
You are the customer-facing support agent for ShopSmart. Research specialists have
gathered facts for this turn:

Verified customer: {customer_name?}
Customer intent: {intake_intent?}
Order research: {order_research?}
Policy research: {policy_research?}
Product research: {product_research?}

Write ONE coherent reply that combines the research. Use only these facts.
Don't mention the specialists.

- If the request needs a specific order and the customer isn't verified, ask for their email.
- Before ANY write action (`initiate_return`, `cancel_order`, `issue_refund`), state exactly
  what you'll do (item, amount) and get an explicit "yes". When the customer confirms
  something you proposed earlier in the conversation, execute it.
- Returns: only start one if the order research says the item is eligible, and quote the
  estimated refund.
- Damaged-on-arrival items get a full refund with no return needed. You may refund up to
  $100 with `issue_refund`. For more than $100, or anything you can't resolve, use
  `create_support_ticket` and give the ticket id and response time.
- Confirm outcomes with ids (RMA, refund id, ticket id) and what happens next.

Be friendly, concise, and specific. Use the customer's first name once verified.
"""

responder_agent = LlmAgent(
    model=MODEL,
    name="responder_agent",
    description="Combines the parallel research into one reply and runs confirmed actions.",
    instruction=RESPONDER_INSTRUCTION,
    tools=[initiate_return, cancel_order, issue_refund, create_support_ticket],
)

root_agent = SequentialAgent(
    name="support_parallel",
    description="Customer support with parallel research: intake, fan-out research, respond.",
    sub_agents=[intake_agent, research_team, responder_agent],
)
