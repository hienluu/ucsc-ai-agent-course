"""Time the research phase: fan-out + JoinNode vs the same researchers one after another.

Usage (from the multi-support-graph-agent directory):
    python -m support_parallel.benchmark
"""

import asyncio
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import JoinNode, Workflow
from google.genai.types import Content, Part

from shared.graph import record_reply, remember_customer, verify_identity
from support_parallel import agent as pa

PROMPT = (
    "I'm alice.chen@example.com. My rain jacket order is late. Can I still return it after it "
    "arrives and do I pay return shipping? Also is the TypePro keyboard in stock?"
)


async def research_one_by_one(ctx: Context, node_input: dict) -> dict:
    """The same three researchers, awaited in sequence instead of fanned out."""
    return {
        "research_order": await pa.research_order(ctx, node_input),
        "research_policy": await pa.research_policy(node_input),
        "research_product": await pa.research_product(node_input),
    }


def build(mode: str) -> App:
    # Fresh copies: an agent instance can belong to only one workflow.
    planner, responder = pa.planner.model_copy(), pa.responder.model_copy()
    head = ("START", remember_customer, verify_identity, pa.load_orders, planner)
    if mode == "parallel":
        join = JoinNode(name="research_done")
        edges = [head, (planner, pa.RESEARCHERS), (pa.RESEARCHERS, join),
                 (join, pa.compile_brief, responder, record_reply)]
    else:
        edges = [head, (planner, research_one_by_one, pa.compile_brief, responder, record_reply)]
    return App(name=f"bench_{mode}", root_agent=Workflow(name=f"bench_{mode}", edges=edges))


async def run(mode: str, i: int):
    app = build(mode)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name=app.name, user_id="u")
    marks = {}
    t0 = time.perf_counter()
    async for ev in runner.run_async(
        user_id="u", session_id=session.id, new_message=Content(role="user", parts=[Part(text=PROMPT)])
    ):
        node = (ev.node_info.path or "").rsplit("/", 1)[-1].split("@", 1)[0] if ev.node_info else ""
        if ev.output is not None:
            marks[node] = time.perf_counter() - t0
    total = time.perf_counter() - t0
    research = marks["compile_brief"] - marks["planner"]
    print(f"{mode:10s} run {i}: total {total:5.1f}s  research phase {research:4.1f}s")


async def main():
    print(f"Each researcher simulates {pa.SIMULATED_API_LATENCY}s of backend latency.\n")
    for i in (1, 2):
        for mode in ("sequential", "parallel"):
            await run(mode, i)


if __name__ == "__main__":
    asyncio.run(main())
