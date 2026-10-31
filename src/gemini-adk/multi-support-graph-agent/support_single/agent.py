"""Single agent, as a graph: one chat-mode LlmAgent node directly after START.

    START -> support_agent (mode="chat", all 11 tools)

The smallest possible Workflow. It behaves like the single LlmAgent in
../multi-support-agent: the model decides everything, including which tool to call
and when the customer has said "yes".

`mode="chat"` keeps the conversation history across turns. ADK allows a chat-mode
agent only directly after START, because it reads the session history instead of a
node input. The multi-node examples keep memory explicitly instead (shared/graph.py).
"""

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.workflow import Workflow

from shared.model import get_model
from shared.tools import ALL_TOOLS

MODEL = get_model()

INSTRUCTION = """\
You are the customer support agent for ShopSmart, an online retailer. You handle
every kind of request yourself: order status, shipment tracking, product
questions, store policies, returns, cancellations, refunds, and escalations.

Verified customer (from session state): {customer_name?}

## Identity
- Before discussing or changing a specific order, verify the customer by asking for
  their email and calling `lookup_customer`. General product and policy questions
  do not need verification.
- Only discuss orders that belong to the verified customer.

## Working rules
- Use tools for facts. Never invent order details, tracking events, prices, or policies.
  If a tool returns an error, explain it plainly and offer the next best option.
- For policy questions, call `search_policies` and answer from what it returns.
- Before ANY write action (`initiate_return`, `cancel_order`, `issue_refund`), state
  exactly what you are about to do (item, amount) and get an explicit "yes" from the customer.
- Returns: always call `check_return_eligibility` first and quote the estimated refund.
- Damaged or defective items delivered recently: you may refund up to $100 directly with
  `issue_refund`. For larger amounts, warranty claims, or anything you cannot resolve,
  call `create_support_ticket` and give the customer the ticket id and response time.
- Escalate immediately with an "urgent" ticket if the customer is very upset, mentions
  legal action, or reports a safety issue.

## Style
- Friendly, concise, and specific. Use the customer's first name once verified.
- Confirm outcomes with ids (RMA, refund id, ticket id) and what happens next.
- Ask one clarifying question at a time when information is missing.
"""

support_agent = LlmAgent(
    model=MODEL,
    name="support_agent",
    description="A single customer support agent for ShopSmart that handles orders, "
    "shipping, products, policies, returns, refunds, and escalations with one set of tools.",
    instruction=INSTRUCTION,
    tools=ALL_TOOLS,
    mode="chat",
)

root_agent = Workflow(
    name="support_single",
    description="One chat-mode support agent as a single-node workflow.",
    edges=[("START", support_agent)],
)

app = App(name="support_single", root_agent=root_agent)
