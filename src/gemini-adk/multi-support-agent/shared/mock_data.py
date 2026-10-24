"""Mocked data for the "ShopSmart" e-commerce customer support use case.

Everything lives in module-level dicts so the tools can read *and* mutate it
(returns, refunds, tickets). State resets every time the process restarts.

Dates are computed relative to today so return-window checks stay meaningful
no matter when the demo is run.
"""

from datetime import date, timedelta

TODAY = date.today()


def _days_ago(n: int) -> str:
    return (TODAY - timedelta(days=n)).isoformat()


def _days_from_now(n: int) -> str:
    return (TODAY + timedelta(days=n)).isoformat()


COMPANY_NAME = "ShopSmart"

# Refunds at or below this amount can be issued by the agent directly.
# Anything larger must be escalated to a human supervisor.
AUTO_REFUND_LIMIT = 100.00

CUSTOMERS = {
    "C1001": {
        "customer_id": "C1001",
        "name": "Alice Chen",
        "email": "alice.chen@example.com",
        "phone": "555-0101",
        "loyalty_tier": "gold",
        "member_since": "2021-03-14",
    },
    "C1002": {
        "customer_id": "C1002",
        "name": "Bob Martinez",
        "email": "bob.martinez@example.com",
        "phone": "555-0102",
        "loyalty_tier": "standard",
        "member_since": "2024-11-02",
    },
    "C1003": {
        "customer_id": "C1003",
        "name": "Priya Patel",
        "email": "priya.patel@example.com",
        "phone": "555-0103",
        "loyalty_tier": "silver",
        "member_since": "2023-06-21",
    },
}

PRODUCTS = {
    "SKU-HP-200": {
        "sku": "SKU-HP-200",
        "name": "AuraSound Wireless Headphones",
        "category": "electronics",
        "price": 149.99,
        "in_stock": 42,
        "returnable": True,
        "warranty_months": 12,
        "description": "Over-ear Bluetooth headphones with active noise cancellation and 30h battery.",
    },
    "SKU-KB-110": {
        "sku": "SKU-KB-110",
        "name": "TypePro Mechanical Keyboard",
        "category": "electronics",
        "price": 89.00,
        "in_stock": 0,
        "restock_date": _days_from_now(12),
        "returnable": True,
        "warranty_months": 24,
        "description": "Hot-swappable mechanical keyboard with brown switches and RGB backlight.",
    },
    "SKU-MG-050": {
        "sku": "SKU-MG-050",
        "name": "Ember Ceramic Mug (12oz)",
        "category": "home",
        "price": 18.50,
        "in_stock": 230,
        "returnable": True,
        "warranty_months": 0,
        "description": "Dishwasher-safe stoneware mug.",
    },
    "SKU-GC-025": {
        "sku": "SKU-GC-025",
        "name": "ShopSmart Gift Card ($25)",
        "category": "gift_card",
        "price": 25.00,
        "in_stock": 9999,
        "returnable": False,
        "warranty_months": 0,
        "description": "Digital gift card delivered by email. Non-refundable.",
    },
    "SKU-JK-300": {
        "sku": "SKU-JK-300",
        "name": "TrailMax Rain Jacket",
        "category": "apparel",
        "price": 129.00,
        "in_stock": 15,
        "returnable": True,
        "warranty_months": 0,
        "description": "Waterproof, breathable shell jacket. Sizes XS-XXL.",
    },
    "SKU-SW-410": {
        "sku": "SKU-SW-410",
        "name": "PulseFit Smartwatch",
        "category": "electronics",
        "price": 249.00,
        "in_stock": 7,
        "returnable": True,
        "warranty_months": 12,
        "description": "Fitness smartwatch with GPS, heart-rate and sleep tracking.",
    },
}

ORDERS = {
    # Delivered recently -> eligible for return.
    "ORD-5001": {
        "order_id": "ORD-5001",
        "customer_id": "C1001",
        "status": "delivered",
        "order_date": _days_ago(14),
        "delivered_date": _days_ago(10),
        "items": [
            {"sku": "SKU-HP-200", "qty": 1, "unit_price": 149.99},
            {"sku": "SKU-MG-050", "qty": 2, "unit_price": 18.50},
        ],
        "total": 186.99,
        "tracking_number": "1Z999AA10123456784",
        "refunded_amount": 0.0,
    },
    # In transit and running late.
    "ORD-5002": {
        "order_id": "ORD-5002",
        "customer_id": "C1001",
        "status": "shipped",
        "order_date": _days_ago(6),
        "delivered_date": None,
        "items": [{"sku": "SKU-JK-300", "qty": 1, "unit_price": 129.00}],
        "total": 129.00,
        "tracking_number": "1Z999AA10987654321",
        "refunded_amount": 0.0,
    },
    # Delivered long ago -> outside the 30-day return window.
    "ORD-5003": {
        "order_id": "ORD-5003",
        "customer_id": "C1002",
        "status": "delivered",
        "order_date": _days_ago(52),
        "delivered_date": _days_ago(47),
        "items": [{"sku": "SKU-SW-410", "qty": 1, "unit_price": 249.00}],
        "total": 249.00,
        "tracking_number": "9400100000000000000001",
        "refunded_amount": 0.0,
    },
    # Still processing -> can be cancelled; contains a backordered item.
    "ORD-5004": {
        "order_id": "ORD-5004",
        "customer_id": "C1002",
        "status": "processing",
        "order_date": _days_ago(1),
        "delivered_date": None,
        "items": [
            {"sku": "SKU-KB-110", "qty": 1, "unit_price": 89.00},
            {"sku": "SKU-GC-025", "qty": 1, "unit_price": 25.00},
        ],
        "total": 114.00,
        "tracking_number": None,
        "refunded_amount": 0.0,
    },
    # Delivered but customer says it arrived damaged (no tracking issue).
    "ORD-5005": {
        "order_id": "ORD-5005",
        "customer_id": "C1003",
        "status": "delivered",
        "order_date": _days_ago(5),
        "delivered_date": _days_ago(2),
        "items": [{"sku": "SKU-SW-410", "qty": 1, "unit_price": 249.00}],
        "total": 249.00,
        "tracking_number": "1Z999AA10555555555",
        "refunded_amount": 0.0,
    },
}

SHIPMENTS = {
    "1Z999AA10123456784": {
        "carrier": "UPS",
        "status": "delivered",
        "estimated_delivery": _days_ago(10),
        "events": [
            {"date": _days_ago(13), "location": "Oakland, CA", "event": "Picked up"},
            {"date": _days_ago(11), "location": "Sacramento, CA", "event": "In transit"},
            {"date": _days_ago(10), "location": "Davis, CA", "event": "Delivered - front porch"},
        ],
    },
    "1Z999AA10987654321": {
        "carrier": "UPS",
        "status": "delayed",
        "estimated_delivery": _days_from_now(2),
        "events": [
            {"date": _days_ago(5), "location": "Reno, NV", "event": "Picked up"},
            {"date": _days_ago(4), "location": "Salt Lake City, UT", "event": "In transit"},
            {"date": _days_ago(2), "location": "Salt Lake City, UT", "event": "Delayed - weather"},
        ],
    },
    "9400100000000000000001": {
        "carrier": "USPS",
        "status": "delivered",
        "estimated_delivery": _days_ago(47),
        "events": [
            {"date": _days_ago(50), "location": "Austin, TX", "event": "Accepted"},
            {"date": _days_ago(47), "location": "Dallas, TX", "event": "Delivered - mailbox"},
        ],
    },
    "1Z999AA10555555555": {
        "carrier": "UPS",
        "status": "delivered",
        "estimated_delivery": _days_ago(2),
        "events": [
            {"date": _days_ago(4), "location": "Seattle, WA", "event": "Picked up"},
            {"date": _days_ago(2), "location": "Portland, OR", "event": "Delivered - left with front desk"},
        ],
    },
}

# Small knowledge base of store policies, keyed by topic.
KNOWLEDGE_BASE = {
    "returns": (
        "Most items can be returned within 30 days of delivery for a full refund to the "
        "original payment method. Items must be unused and in original packaging. Gift cards "
        "are non-returnable. Gold-tier members get free return shipping; other customers pay "
        "a $6.99 return shipping fee, deducted from the refund."
    ),
    "refunds": (
        "Refunds are issued to the original payment method and appear within 5-7 business "
        "days after the return is received. Damaged-on-arrival items are refunded in full, "
        "including shipping, without needing to be returned first."
    ),
    "shipping": (
        "Standard shipping takes 3-5 business days and is free on orders over $50. Express "
        "shipping (1-2 business days) costs $14.99. We ship to the US and Canada only."
    ),
    "cancellations": (
        "Orders can be cancelled free of charge while they are still in 'processing' status. "
        "Cancellation applies to the whole order: individual items can't be removed from an "
        "order, but the customer can cancel and re-order the items they still want. "
        "Once an order has shipped it cannot be cancelled, but it can be returned after delivery."
    ),
    "warranty": (
        "Electronics include a manufacturer warranty (see product page for duration). Warranty "
        "claims cover defects, not accidental damage. Open a support ticket with the order ID and "
        "a description of the defect to start a claim."
    ),
    "price_match": (
        "We match the price of identical items sold by major US retailers within 14 days of "
        "purchase. The difference is refunded as store credit."
    ),
    "loyalty": (
        "Loyalty tiers: standard, silver (spend $500/yr), gold (spend $1500/yr). Silver gets "
        "free express shipping twice per year; gold gets free returns and priority support."
    ),
}

# Records created by write tools during a session.
RETURNS: dict[str, dict] = {}
REFUNDS: dict[str, dict] = {}
TICKETS: dict[str, dict] = {}
