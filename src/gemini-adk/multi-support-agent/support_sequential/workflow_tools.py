"""Propose -> review -> execute tools for the sequential pipeline.

The proposal lives in session state under "proposal":

    {
      "id": 1,
      "action": "issue_refund",
      "args": {"order_id": "ORD-5005", "amount": 50.0, "reason": "..."},
      "summary": "Refund $50.00 on ORD-5005 for ...",
      "review": None | {"approved": bool, "reason": str},
    }

- resolver_agent can only *propose* (propose_action).
- policy_checker_agent can only *review* (record_verdict).
- responder_agent can only *execute what was approved* (execute_approved_action),
  and cannot change the arguments.
"""

from google.adk.tools.tool_context import ToolContext

from shared import tools as support_tools

# The write tools a proposal may target, and the arguments each one takes.
ACTIONS = {
    "initiate_return": (support_tools.initiate_return, ("order_id", "sku", "reason")),
    "cancel_order": (support_tools.cancel_order, ("order_id",)),
    "issue_refund": (support_tools.issue_refund, ("order_id", "amount", "reason")),
    "create_support_ticket": (
        support_tools.create_support_ticket,
        ("subject", "description", "priority", "order_id"),
    ),
}


def _error(message: str) -> dict:
    return {"status": "error", "error_message": message}


def propose_action(
    action: str,
    summary: str,
    tool_context: ToolContext,
    order_id: str = "",
    sku: str = "",
    amount: float = 0.0,
    reason: str = "",
    subject: str = "",
    description: str = "",
    priority: str = "normal",
) -> dict:
    """Propose ONE write action for the policy checker to review. Does not execute it.

    Replaces any earlier proposal that hasn't been executed yet.

    Args:
        action: One of "initiate_return", "cancel_order", "issue_refund", "create_support_ticket".
        summary: One sentence describing the action for the customer, e.g.
            "Return AuraSound Wireless Headphones from ORD-5001 for an estimated $149.99 refund."
        order_id: The order id (required for every action except a ticket with no order).
        sku: The product SKU (initiate_return only).
        amount: Refund amount in USD (issue_refund only).
        reason: Why the customer wants this (initiate_return, issue_refund).
        subject: Ticket subject (create_support_ticket only).
        description: Ticket description with full context (create_support_ticket only).
        priority: Ticket priority: "low", "normal", "high", or "urgent" (create_support_ticket only).

    Returns:
        The stored proposal, pending review.
    """
    if action not in ACTIONS:
        return _error(f"Unknown action '{action}'. Must be one of: {', '.join(ACTIONS)}.")
    supplied = {
        "order_id": order_id, "sku": sku, "amount": amount, "reason": reason,
        "subject": subject, "description": description, "priority": priority,
    }
    _, arg_names = ACTIONS[action]
    args = {name: supplied[name] for name in arg_names}

    previous = tool_context.state.get("proposal") or {}
    proposal = {
        "id": previous.get("id", 0) + 1,
        "action": action,
        "args": args,
        "summary": summary,
        "review": None,
    }
    # Reassign the whole value so ADK records the state change.
    tool_context.state["proposal"] = proposal
    return {"status": "success", "proposal": proposal, "next": "pending policy review"}


def record_verdict(approved: bool, reason: str, tool_context: ToolContext) -> dict:
    """Record the policy review decision for the current proposal.

    Args:
        approved: True if the proposal complies with policy and the facts, else False.
        reason: Short justification citing the policy or fact you checked. If rejecting,
            say what the agent should do instead (e.g. "escalate with a high-priority ticket").

    Returns:
        The updated proposal.
    """
    proposal = tool_context.state.get("proposal")
    if not proposal:
        return _error("There is no proposal to review.")
    proposal = {**proposal, "review": {"approved": approved, "reason": reason}}
    tool_context.state["proposal"] = proposal
    return {"status": "success", "proposal": proposal}


def execute_approved_action(tool_context: ToolContext) -> dict:
    """Execute the current proposal exactly as approved by the policy checker.

    Only call this when the customer's latest message explicitly confirms the action.

    Returns:
        The result of the underlying action (RMA, refund id, ticket id, ...).
    """
    proposal = tool_context.state.get("proposal")
    if not proposal:
        return _error("There is no proposal to execute.")
    review = proposal.get("review")
    if not review:
        return _error("The proposal has not been reviewed yet.")
    if not review["approved"]:
        return _error(f"The proposal was rejected by policy review: {review['reason']}")

    func, _ = ACTIONS[proposal["action"]]
    result = func(**proposal["args"], tool_context=tool_context)
    if result.get("status") == "success":
        tool_context.state["proposal"] = None
        tool_context.state["last_executed"] = {"summary": proposal["summary"], "result": result}
    return result


def discard_proposal(tool_context: ToolContext) -> dict:
    """Discard the current proposal, e.g. when the customer declines it.

    Returns:
        Confirmation that the proposal was discarded.
    """
    tool_context.state["proposal"] = None
    return {"status": "success"}
