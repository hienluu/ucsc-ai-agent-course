"""Time the research phase with ParallelAgent vs SequentialAgent on the same prompt.

Usage (from the multi-support-agent directory):
    python -m support_parallel.benchmark
"""

import asyncio
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from support_parallel import agent as pa

PROMPT = ("I'm alice.chen@example.com. My rain jacket order is late. Can I still return it after it "
          "arrives and do I pay return shipping? Also is the TypePro keyboard in stock?")

def build(mode):
    """Rebuild the pipeline with fresh agent copies (an agent can only have one parent)."""
    researchers = [a.clone() for a in pa.research_team.sub_agents]
    team = (pa.ParallelAgent if mode == "parallel" else SequentialAgent)(name="research_team", sub_agents=researchers)
    return SequentialAgent(name="bench", sub_agents=[pa.intake_agent.clone(), team, pa.responder_agent.clone()])

async def run(mode, i):
    svc = InMemorySessionService()
    s = await svc.create_session(app_name="b", user_id="u")
    runner = Runner(agent=build(mode), app_name="b", session_service=svc)
    marks = {}
    t0 = time.perf_counter()
    async for ev in runner.run_async(user_id="u", session_id=s.id, new_message=Content(role="user", parts=[Part(text=PROMPT)])):
        marks.setdefault(ev.author + "_first", time.perf_counter() - t0)
        marks[ev.author + "_last"] = time.perf_counter() - t0
    total = time.perf_counter() - t0
    research = max(marks.get(f"{r}_last", 0) for r in ("order_researcher","policy_researcher","product_researcher")) - marks["intake_agent_last"]
    print(f"{mode:10s} run {i}: total {total:5.1f}s  research phase {research:5.1f}s")

async def main():
    # LLM latency is noisy, so run each mode twice.
    for i in (1, 2):
        for mode in ("sequential", "parallel"):
            await run(mode, i)


if __name__ == "__main__":
    asyncio.run(main())
