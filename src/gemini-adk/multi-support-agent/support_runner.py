"""Interactive CLI runner for the customer support agents.

Usage (from the multi-support-agent directory):
    python support_runner.py                    # runs support_single/agent.py
    python support_runner.py support_single     # same thing, explicit
    python support_runner.py support_single -v  # also dump every event as JSON
    python support_runner.py support_routing    # triage agent + specialists
    python support_runner.py support_sequential # resolver -> policy checker -> responder
    python support_runner.py support_parallel   # intake -> parallel research -> responder
    python support_runner.py support_loop       # draft -> review -> revise until approved
"""

import argparse
import asyncio
import importlib
import json
import time

from dotenv import load_dotenv
from rich import print as rprint
from rich.markup import escape

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()

APP_NAME = "customer_support_app"
USER_ID = "user_123"
SESSION_ID = "session_123"

# Agents whose output is behind-the-scenes, not a reply to the customer.
INTERNAL_AGENTS = {
    "resolver_agent", "policy_checker_agent",                                     # sequential
    "intake_agent", "order_researcher", "policy_researcher", "product_researcher",  # parallel
    "case_agent", "drafter_agent", "reviewer_agent",                              # loop
}


async def chat_loop(agent_package: str, verbose: bool):
    root_agent = importlib.import_module(f"{agent_package}.agent").root_agent

    print("Welcome to ShopSmart Customer Support.")
    print(f"Active agent: {root_agent.name}")
    print("Try: 'Where is my order ORD-5002? my email is alice.chen@example.com'. Type 'exit' to quit.\n")

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    while True:
        user_input = await asyncio.to_thread(input, "You: ")
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        new_message = Content(role="user", parts=[Part(text=user_input)])
        events = runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=new_message,
        )

        event_no = 0
        start = time.perf_counter()
        try:
            async for event in events:
                event_no += 1
                if verbose:
                    print_json_response(event, f"Event {event_no}")
                print_tool_activity(event, time.perf_counter() - start)
                if event.is_final_response() and event.content and event.content.parts:
                    # Skip "thought" parts so model reasoning doesn't leak into the reply.
                    final_response = "".join(
                        p.text for p in event.content.parts if p.text and not p.thought
                    )
                    # Don't stop at the first final response: in a pipeline, several
                    # agents each finish in turn. Internal agents are printed dimmed.
                    if not final_response.strip():
                        continue
                    if event.author in INTERNAL_AGENTS:
                        rprint(f"\n[dim]  {escape(f'[{event.author}]')} {escape(final_response)}[/dim]\n")
                    else:
                        rprint(f"\n[bold green]{event.author}:[/bold green] {final_response}\n")
        finally:
            await events.aclose()
        rprint(f"[dim](turn took {time.perf_counter() - start:.1f}s)[/dim]\n")


def print_tool_activity(event, elapsed: float) -> None:
    """Print a one-line summary of each tool call and tool result, with time since turn start."""
    t = f"[dim]{elapsed:5.1f}s[/dim]"
    for call in event.get_function_calls():
        if call.name == "transfer_to_agent":
            rprint(f"  {t} [magenta]⇢ {event.author} transfers to {call.args.get('agent_name')}[/magenta]")
            continue
        rprint(f"  {t} [cyan]→ {event.author} calls {call.name}({escape(json.dumps(call.args))})[/cyan]")
    for resp in event.get_function_responses():
        if resp.name == "transfer_to_agent":
            continue
        status = (resp.response or {}).get("status", "?")
        color = "green" if status == "success" else "red"
        rprint(f"  {t} [{color}]← {event.author} {resp.name}: {status}[/{color}]")


def print_json_response(response, title: str) -> None:
    print(f"\n--- {title} ---")
    try:
        data = response.model_dump(mode="json", exclude_none=True)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        rprint(f"[red bold]Error printing response to JSON:[/red bold] {e}")
        rprint(repr(response))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent", nargs="?", default="support_single", help="agent package to run (default: support_single)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every event as JSON")
    args = parser.parse_args()
    asyncio.run(chat_loop(args.agent, args.verbose))
