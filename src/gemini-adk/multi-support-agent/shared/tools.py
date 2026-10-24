"""Customer support tools backed by the mocked data in `mock_data.py`.

Read tools look things up; write tools (initiate_return, cancel_order,
issue_refund, create_support_ticket) mutate the in-memory data.

Anything tied to a specific order requires a verified customer:
`lookup_customer` stores the customer_id in session state, and order tools
refuse to read or change orders that don't belong to that customer. This
guardrail is enforced in code, not just in the prompt. Product and policy
search are open to anyone.

Every tool returns a dict with a "status" of "success" or "error" so the
LLM (and evals) can branch on it reliably.
"""

from datetime import date, datetime

from google.adk.tools.tool_context import ToolContext

from shared import mock_data as db

RETURN_WINDOW_DAYS = 30
RETURN_SHIPPING_FEE = 6.99


def _error(message: str) -> dict:
    return {"status": "error", "error_message": message}


def _verified_order(order_id: str, tool_context: ToolContext) -> tuple[dict | None, dict | None]:
    """Return (order, None) if the order belongs to the verified customer, else (None, error)."""
    customer_id = tool_context.state.get("customer_id")
    if not customer_id:
        return None, _error("Customer is not verified. Call lookup_customer with their email first.")
    order = db.ORDERS.get(order_id.strip().upper())
    if order is None:
        return None, _error(f"Order {order_id} not found.")
    if order["customer_id"] != customer_id:
        return None, _error(f"Order {order_id} does not belong to the verified customer.")
    return order, None


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

def lookup_customer(email: str, tool_context: ToolContext) -> dict:
    """Look up a customer by email and mark them as verified for this session.

    Args:
        email: The customer's email address.

    Returns:
        The customer's profile (id, name, loyalty tier, member since).
    """
    email = email.strip().lower()
    for customer in db.CUSTOMERS.values():
        if customer["email"] == email:
            tool_context.state["customer_id"] = customer["customer_id"]
            tool_context.state["customer_name"] = customer["name"]
            return {"status": "success", "customer": customer}
    return _error(f"No customer found with email {email}.")


def list_customer_orders(tool_context: ToolContext) -> dict:
    """List all orders for the verified customer, most recent first.

    Returns:
        A summary of each order: id, date, status, and total.
    """
    customer_id = tool_context.state.get("customer_id")
    if not customer_id:
        return _error("Customer is not verified. Call lookup_customer with their email first.")
    orders = [
        {k: o[k] for k in ("order_id", "order_date", "status", "total")}
        for o in db.ORDERS.values()
        if o["customer_id"] == customer_id
    ]
    if not orders:
        return _error("The verified customer has no orders.")
    orders.sort(key=lambda o: o["order_date"], reverse=True)
    return {"status": "success", "orders": orders}


def get_order_details(order_id: str, tool_context: ToolContext) -> dict:
    """Get full details for an order, including line items with product names.

    Args:
        order_id: The order id, e.g. "ORD-5001".

    Returns:
        The order with status, dates, total, tracking number, refunded amount, and items
        (including whether each item is in stock and its restock date if backordered).
    """
    order, err = _verified_order(order_id, tool_context)
    if err:
        return err
    items = []
    for item in order["items"]:
        product = db.PRODUCTS[item["sku"]]
        items.append({
            **item,
            "name": product["name"],
            "in_stock": product["in_stock"] > 0,
            "restock_date": product.get("restock_date"),
        })
    return {"status": "success", "order": {**order, "items": items}}


def track_shipment(order_id: str, tool_context: ToolContext) -> dict:
    """Get carrier tracking status and scan history for an order.

    Args:
        order_id: The order id, e.g. "ORD-5002".

    Returns:
        Carrier, current shipment status, estimated delivery date, and tracking events.
    """
    order, err = _verified_order(order_id, tool_context)
    if err:
        return err
    tracking_number = order["tracking_number"]
    if not tracking_number:
        return _error(f"Order {order_id} has not shipped yet (status: {order['status']}).")
    shipment = db.SHIPMENTS[tracking_number]
    return {
        "status": "success",
        "tracking_number": tracking_number,
        "carrier": shipment["carrier"],
        "shipment_status": shipment["status"],
        "estimated_delivery": shipment["estimated_delivery"],
        "events": shipment["events"],
    }


def search_products(query: str) -> dict:
    """Search the product catalog by name, category, or description keywords.

    Args:
        query: Keywords such as "headphones", "electronics", or "waterproof".

    Returns:
        Matching products with price, stock level, restock date, and warranty.
    """
    terms = query.lower().split()
    matches = [
        p for p in db.PRODUCTS.values()
        if any(t in f"{p['name']} {p['category']} {p['description']}".lower() for t in terms)
    ]
    if not matches:
        return _error(f"No products match '{query}'.")
    return {"status": "success", "products": matches}


def search_policies(query: str) -> dict:
    """Search store policies (returns, refunds, shipping, cancellations, warranty,
    price_match, loyalty).

    Args:
        query: A policy topic or keywords, e.g. "return window" or "express shipping".

    Returns:
        The matching policy texts keyed by topic.
    """
    # Naive keyword scoring: ignore short words, weight topic-name hits higher,
    # and return only the best-matching topics.
    terms = [t.strip("?.,!'\"") for t in query.lower().split()]
    terms = [t.rstrip("s") for t in terms if len(t) > 3]
    scores = {
        topic: sum(3 * (t in topic) + text.lower().count(t) for t in terms)
        for topic, text in db.KNOWLEDGE_BASE.items()
    }
    best = max(scores.values(), default=0)
    matches = {
        topic: db.KNOWLEDGE_BASE[topic]
        for topic, score in scores.items()
        if best and score >= best / 2
    }
    if not matches:
        return _error(f"No policy found for '{query}'. Available topics: {', '.join(db.KNOWLEDGE_BASE)}.")
    return {"status": "success", "policies": matches}


def check_return_eligibility(order_id: str, sku: str, tool_context: ToolContext) -> dict:
    """Check whether an item from an order can be returned and the estimated refund.

    Args:
        order_id: The order id, e.g. "ORD-5001".
        sku: The product SKU in that order, e.g. "SKU-HP-200".

    Returns:
        Whether the item is eligible, the reason, days since delivery, and estimated refund.
    """
    order, err = _verified_order(order_id, tool_context)
    if err:
        return err
    return _return_eligibility(order, sku)


def _return_eligibility(order: dict, sku: str) -> dict:
    sku = sku.strip().upper()
    item = next((i for i in order["items"] if i["sku"] == sku), None)
    if item is None:
        return _error(f"{sku} is not part of order {order['order_id']}.")
    product = db.PRODUCTS[sku]

    def ineligible(reason: str) -> dict:
        return {"status": "success", "eligible": False, "reason": reason}

    if order["status"] != "delivered":
        return ineligible(f"Order is '{order['status']}', not delivered. Only delivered items can be returned.")
    if not product["returnable"]:
        return ineligible(f"{product['name']} is non-returnable.")
    if any(r["order_id"] == order["order_id"] and r["sku"] == sku for r in db.RETURNS.values()):
        return ineligible("A return has already been started for this item.")

    days = (date.today() - date.fromisoformat(order["delivered_date"])).days
    if days > RETURN_WINDOW_DAYS:
        return ineligible(f"Delivered {days} days ago; the return window is {RETURN_WINDOW_DAYS} days.")

    gold = db.CUSTOMERS[order["customer_id"]]["loyalty_tier"] == "gold"
    fee = 0.0 if gold else RETURN_SHIPPING_FEE
    return {
        "status": "success",
        "eligible": True,
        "days_since_delivery": days,
        "item_value": round(item["unit_price"] * item["qty"], 2),
        "return_shipping_fee": fee,
        "estimated_refund": round(item["unit_price"] * item["qty"] - fee, 2),
    }


# ---------------------------------------------------------------------------
# Write tools (require a verified customer)
# ---------------------------------------------------------------------------

def initiate_return(order_id: str, sku: str, reason: str, tool_context: ToolContext) -> dict:
    """Start a return for an item and generate a return authorization (RMA).

    Only call this after check_return_eligibility says the item is eligible
    and the customer has confirmed they want to return it.

    Args:
        order_id: The order id, e.g. "ORD-5001".
        sku: The product SKU to return.
        reason: The customer's reason for the return.

    Returns:
        The RMA number, return shipping label URL, and estimated refund.
    """
    order, err = _verified_order(order_id, tool_context)
    if err:
        return err
    eligibility = _return_eligibility(order, sku)
    if eligibility["status"] == "error" or not eligibility["eligible"]:
        return _error(f"Item is not eligible for return: {eligibility.get('reason') or eligibility.get('error_message')}")

    rma = f"RMA-{7000 + len(db.RETURNS) + 1}"
    db.RETURNS[rma] = {
        "rma": rma,
        "order_id": order["order_id"],
        "sku": sku.strip().upper(),
        "reason": reason,
        "estimated_refund": eligibility["estimated_refund"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return {
        "status": "success",
        "rma": rma,
        "label_url": f"https://shopsmart.example.com/labels/{rma}.pdf",
        "estimated_refund": eligibility["estimated_refund"],
        "next_steps": "Ship the item within 14 days. Refund is issued 5-7 business days after we receive it.",
    }


def cancel_order(order_id: str, tool_context: ToolContext) -> dict:
    """Cancel an order that is still in 'processing' status and refund it in full.

    Only call this after the customer has confirmed they want to cancel.

    Args:
        order_id: The order id, e.g. "ORD-5004".

    Returns:
        Confirmation of the cancellation and the refunded amount.
    """
    order, err = _verified_order(order_id, tool_context)
    if err:
        return err
    if order["status"] != "processing":
        return _error(f"Order {order['order_id']} is '{order['status']}' and can no longer be cancelled.")
    order["status"] = "cancelled"
    order["refunded_amount"] = order["total"]
    return {"status": "success", "order_id": order["order_id"], "refunded_amount": order["total"]}


def issue_refund(order_id: str, amount: float, reason: str, tool_context: ToolContext) -> dict:
    """Issue a refund to the original payment method, e.g. for a damaged or missing item.

    Refunds above the auto-approval limit ($100) are rejected and must be
    escalated with create_support_ticket. Only call this after the customer
    has confirmed the amount.

    Args:
        order_id: The order id, e.g. "ORD-5005".
        amount: The refund amount in USD.
        reason: Why the refund is being issued.

    Returns:
        The refund id and amount, or an error if the amount exceeds the limit or order total.
    """
    order, err = _verified_order(order_id, tool_context)
    if err:
        return err
    if amount <= 0:
        return _error("Refund amount must be positive.")
    remaining = round(order["total"] - order["refunded_amount"], 2)
    if amount > remaining:
        return _error(f"Amount exceeds the refundable balance of ${remaining:.2f}.")
    if amount > db.AUTO_REFUND_LIMIT:
        return _error(
            f"Refunds over ${db.AUTO_REFUND_LIMIT:.2f} need supervisor approval. "
            "Create a high-priority support ticket instead."
        )

    refund_id = f"RF-{9000 + len(db.REFUNDS) + 1}"
    order["refunded_amount"] = round(order["refunded_amount"] + amount, 2)
    db.REFUNDS[refund_id] = {
        "refund_id": refund_id,
        "order_id": order["order_id"],
        "amount": amount,
        "reason": reason,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return {"status": "success", "refund_id": refund_id, "amount": amount, "arrives_in": "5-7 business days"}


def create_support_ticket(
    subject: str,
    description: str,
    priority: str,
    tool_context: ToolContext,
    order_id: str = "",
) -> dict:
    """Escalate an issue to the human support team by creating a ticket.

    Use for refunds over the auto-approval limit, warranty claims, angry or
    repeat-contact customers, or anything the tools can't resolve.

    Args:
        subject: A short summary of the issue.
        description: Full context: what the customer wants and what was already tried.
        priority: One of "low", "normal", "high", "urgent".
        order_id: The related order id, if any.

    Returns:
        The ticket id and the expected response time.
    """
    customer_id = tool_context.state.get("customer_id")
    if not customer_id:
        return _error("Customer is not verified. Call lookup_customer with their email first.")
    priority = priority.strip().lower()
    if priority not in {"low", "normal", "high", "urgent"}:
        return _error("priority must be one of: low, normal, high, urgent.")

    response_times = {"low": "3 business days", "normal": "1 business day", "high": "4 hours", "urgent": "1 hour"}
    if db.CUSTOMERS[customer_id]["loyalty_tier"] == "gold" and priority in {"low", "normal"}:
        priority = "high"  # gold members get priority support

    ticket_id = f"TKT-{3000 + len(db.TICKETS) + 1}"
    db.TICKETS[ticket_id] = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "order_id": order_id or None,
        "subject": subject,
        "description": description,
        "priority": priority,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return {
        "status": "success",
        "ticket_id": ticket_id,
        "priority": priority,
        "expected_response": response_times[priority],
    }


READ_TOOLS = [
    lookup_customer,
    list_customer_orders,
    get_order_details,
    track_shipment,
    search_products,
    search_policies,
    check_return_eligibility,
]

WRITE_TOOLS = [
    initiate_return,
    cancel_order,
    issue_refund,
    create_support_ticket,
]

ALL_TOOLS = READ_TOOLS + WRITE_TOOLS
