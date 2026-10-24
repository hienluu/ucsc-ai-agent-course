"""Deterministic grounding check: is every concrete detail in a draft backed by a fact?

The case agent logs every tool result to state["tool_facts"]. Before the reviewer
runs, `check_draft` extracts the concrete details from the draft (dollar amounts,
order/RMA/ticket/refund ids, SKUs, calendar dates) and reports any that don't
appear in those tool results or in the store policies.

This catches invented details reliably, run after run, which an LLM reviewer can't
guarantee. Judging tone and completeness is left to the LLM reviewer.
"""

import re
from datetime import date, datetime

from shared import mock_data as db

ID_RE = re.compile(r"\b(?:ORD|RMA|TKT|RF)-\d+\b|\bSKU-[A-Z]+-\d+\b")
MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
WRITTEN_DATE_RE = re.compile(rf"\b({MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b")


def _month_day(month_name: str, day: str) -> tuple[int, int] | None:
    name = month_name[:3]
    try:
        return datetime.strptime(f"{name} {day}", "%b %d").month, int(day)
    except ValueError:
        return None


def check_draft(draft: str, tool_facts: list[str]) -> list[str]:
    """Return a list of details in `draft` that no tool result or policy supports."""
    facts = "\n".join(tool_facts)
    reference = facts + "\n" + "\n".join(db.KNOWLEDGE_BASE.values())
    issues = []

    # Ids must come from tool results (an RMA or ticket id proves an action happened).
    for found in sorted(set(ID_RE.findall(draft))):
        if found not in facts:
            issues.append(f"{found} does not appear in any tool result. Don't invent ids or claim actions that weren't taken.")

    # Dollar amounts must match a number in the facts or policies.
    known_numbers = {float(n) for n in NUMBER_RE.findall(reference)} | {db.AUTO_REFUND_LIMIT}
    for raw in sorted(set(MONEY_RE.findall(draft))):
        if float(raw.replace(",", "")) not in known_numbers:
            issues.append(f"${raw} is not in any tool result or policy. Use only amounts the tools returned.")

    # Calendar dates must match a date in the tool results (compare month and day).
    known_dates = {(int(m), int(d)) for _, m, d in ISO_DATE_RE.findall(facts)}
    known_dates.add((date.today().month, date.today().day))
    draft_dates = {(int(m), int(d)) for _, m, d in ISO_DATE_RE.findall(draft)}
    for month_name, day in WRITTEN_DATE_RE.findall(draft):
        md = _month_day(month_name, day)
        if md:
            draft_dates.add(md)
    for month, day in sorted(draft_dates):
        if (month, day) not in known_dates:
            label = date(2000, month, day).strftime("%B %-d")
            issues.append(f"The date {label} is not in any tool result. Don't invent dates.")

    return issues
