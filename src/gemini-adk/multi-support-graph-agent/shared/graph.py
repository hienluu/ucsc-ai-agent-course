"""Building blocks shared by the graph workflows.

Conversation memory
-------------------
An LlmAgent used as a workflow node runs in `single_turn` mode: it sees its node
input and nothing else, not even its own earlier turns. (A `chat`-mode agent keeps
history, but ADK only allows it directly after START.) So each multi-node graph
keeps the conversation explicitly in state:

    START -> remember_customer -> ... -> <reply node> -> record_reply

and LLM nodes read it through `{conversation}` in their instructions.

Identity
--------
`verify_identity` is a plain function node: if the message contains a known
email, the customer is verified. No LLM decides who the customer is.
"""

import re

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.genai import types

from shared import mock_data as db

MAX_CONVERSATION_CHARS = 6000
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def content_text(value) -> str:
    """Plain text from a node input: Content (START, LLM nodes), str, or anything else."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, types.Content):
        return "".join(p.text for p in value.parts or [] if p.text and not p.thought)
    return str(value)


def _append(conversation: str, speaker: str, text: str) -> str:
    updated = f"{conversation}\n{speaker}: {text}".strip()
    return updated[-MAX_CONVERSATION_CHARS:]


def remember_customer(ctx: Context, node_input: types.Content):
    """Record the customer's message in the conversation and pass it on as text."""
    text = content_text(node_input)
    return Event(
        output=text,
        state={
            "customer_message": text,
            "conversation": _append(ctx.state.get("conversation", ""), "Customer", text),
        },
    )


def record_reply(ctx: Context, node_input):
    """Record an LLM node's reply in the conversation. Terminal node: no output."""
    text = content_text(node_input)
    if not text:
        return None
    return Event(state={"conversation": _append(ctx.state.get("conversation", ""), "Agent", text)})


def reply(ctx: Context, text: str, **state) -> Event:
    """A customer-facing reply from a function node: shown in the UI and recorded."""
    return Event(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        output=text,
        state={"conversation": _append(ctx.state.get("conversation", ""), "Agent", text), **state},
    )


def verify_identity(ctx: Context, node_input: str):
    """If the message contains a known customer email, mark that customer as verified."""
    state = {"identity_note": ""}
    for email in EMAIL_RE.findall(node_input):
        email = email.lower().rstrip(".")
        customer = next((c for c in db.CUSTOMERS.values() if c["email"] == email), None)
        if customer:
            state.update(
                customer_id=customer["customer_id"],
                customer_name=customer["name"],
                customer_tier=customer["loyalty_tier"],
            )
            break
        state["identity_note"] = f"No account was found for {email}."
    return Event(output=node_input, state=state)
