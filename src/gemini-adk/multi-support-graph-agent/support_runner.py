"""Interactive CLI runner for the graph-based customer support workflows.

Usage (from the multi-support-graph-agent directory):
    python support_runner.py                     # runs support_single
    python support_runner.py support_routing     # classify + route in code
    python support_runner.py support_sequential  # propose -> check -> human confirm -> execute
    python support_runner.py support_parallel    # plan -> parallel lookups -> join -> respond
    python support_runner.py support_loop        # draft -> fact check -> review -> revise cycle
    python support_runner.py support_loop -v     # also dump every event as JSON

Each example module exports `app` (an ADK App). When a workflow pauses for human
input (RequestInput), the runner shows the question and sends your answer back
as the function response that resumes the workflow.
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
from google.genai.types import Content, FunctionResponse, Part

load_dotenv()

USER_ID = "user_123"
REQUEST_INPUT = "adk_request_input"

# Nodes whose text is behind-the-scenes work, not a reply to the customer.
INTERNAL_NODES = {
    "classify",                                     # routing
    "resolver", "policy_checker",                   # sequential
    "planner",                                      # parallel
    "case_agent", "drafter", "reviewer",            # loop
}


def node_name(event) -> str:
    """'support_loop@1/drafter@2' -> 'drafter'. Falls back to the event author."""
    path = getattr(getattr(event, "node_info", None), "path", None) or ""
    return path.rsplit("/", 1)[-1].split("@", 1)[0] or event.author


async def chat_loop(package: str, verbose: bool):
    app = importlib.import_module(f"{package}.agent").app

    print("Welcome to ShopSmart Customer Support (graph workflows).")
    print(f"Active app: {app.name}")
    print("Try: 'Where is my order ORD-5002? my email is alice.chen@example.com'. Type 'exit' to quit.\n")

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=app.name, user_id=USER_ID)
    runner = Runner(app=app, session_service=session_service)

    pending = None  # (interrupt id, message) when the workflow is waiting for input
    while True:
        prompt = "You (answer): " if pending else "You: "
        user_input = await asyncio.to_thread(input, prompt)
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        if pending:
            part = Part(function_response=FunctionResponse(
                id=pending[0], name=REQUEST_INPUT, response={"result": user_input},
            ))
            pending = None
        else:
            part = Part(text=user_input)

        start = time.perf_counter()
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session.id, new_message=Content(role="user", parts=[part])
        ):
            if verbose:
                print(json.dumps(event.model_dump(mode="json", exclude_none=True), indent=2))
            interrupt = print_event(event, time.perf_counter() - start)
            if interrupt:
                pending = interrupt
        rprint(f"[dim](turn took {time.perf_counter() - start:.1f}s)[/dim]\n")


def print_event(event, elapsed: float):
    """Print tool activity and text. Returns (id, message) if the workflow asks for input."""
    t = f"[dim]{elapsed:5.1f}s[/dim]"
    name = node_name(event)
    interrupt = None

    for call in event.get_function_calls():
        if call.name == REQUEST_INPUT:
            message = (call.args or {}).get("message") or "Input needed:"
            rprint(f"\n[bold yellow]{escape(name)} asks:[/bold yellow] {escape(message)}")
            interrupt = (call.id, message)
            continue
        rprint(f"  {t} [cyan]→ {escape(name)} calls {call.name}({escape(json.dumps(call.args))})[/cyan]")
    for resp in event.get_function_responses():
        if resp.name == REQUEST_INPUT:
            continue
        status = (resp.response or {}).get("status", "?")
        color = "green" if status == "success" else "red"
        rprint(f"  {t} [{color}]← {escape(name)} {resp.name}: {status}[/{color}]")

    # Show the path through the graph: which route each routing node picked.
    route = getattr(event.actions, "route", None)
    if route is not None:
        detail = ""
        if isinstance(event.output, dict) and event.output.get("issues"):
            detail = " — " + "; ".join(event.output["issues"])
        rprint(f"  {t} [magenta]⤷ {escape(name)} → {escape(str(route))}{escape(detail)}[/magenta]")

    # Code-only research nodes have no tool calls or text; show when they finish.
    if name.startswith("research_") and event.output is not None:
        rprint(f"  {t} [blue]◆ {escape(name)} finished[/blue]")

    text = ""
    if event.content and event.content.parts and not event.get_function_calls():
        text = "".join(p.text for p in event.content.parts if p.text and not p.thought).strip()
    if text and not event.partial:
        if name in INTERNAL_NODES:
            rprint(f"\n[dim]  {escape(f'[{name}]')} {escape(text)}[/dim]\n")
        else:
            rprint(f"\n[bold green]{escape(name)}:[/bold green] {escape(text)}\n")
    return interrupt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", nargs="?", default="support_single", help="example to run (default: support_single)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every event as JSON")
    args = parser.parse_args()
    asyncio.run(chat_loop(args.package, args.verbose))
