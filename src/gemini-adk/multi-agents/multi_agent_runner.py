import asyncio
import json

from dotenv import load_dotenv
from rich import print as rprint

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()

from single.agent import root_agent

APP_NAME = "multi_agent_blog_app"
USER_ID = "user_123"
SESSION_ID = "session_123"


async def chat_loop():
    print("Welcome to the Multi-Agent Blog Runner.")
    print(f"Active agent: {root_agent.name}")
    print("Describe a blog post (topic, audience, optional post_type). Type 'exit' to quit.\n")

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print(f"Session created: {session.id}\n")

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
        try:
            async for event in events:
                event_no += 1
                print_json_response(event, f"Event {event_no}")
                if event.is_final_response():
                    final_response = event.content.parts[0].text
                    print(f"\n=== Agent response ===\n{final_response}\n")
                    break
        finally:
            await events.aclose()


def print_json_response(response, title: str) -> None:
    print(f"\n--- {title} ---")
    try:
        if hasattr(response, "root"):
            data = response.root.model_dump(mode="json", exclude_none=True)
        else:
            data = response.model_dump(mode="json", exclude_none=True)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        rprint(f"[red bold]Error printing response to JSON:[/red bold] {e}")
        rprint(repr(response))


if __name__ == "__main__":
    asyncio.run(chat_loop())
