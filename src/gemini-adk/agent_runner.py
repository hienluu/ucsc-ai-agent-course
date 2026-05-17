import asyncio
from dotenv import load_dotenv
import json

from rich import print as rprint
from rich.syntax import Syntax

from google.genai.types import Content, Part
from google.adk.runners import Runner
from google.adk.sessions  import InMemorySessionService

load_dotenv()

from agents.basic_agent.agent import root_agent

# resource: 
# - https://adk.dev/tutorials/agent-team/#step-1-your-first-agent-basic-weather-lookup
# - https://docs.litellm.ai/docs/tutorials/google_adk

async def chat_loop():
    print("Welcome to the Gemini ADK Chat Loop! Type 'exit' to quit.")
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="basic_agent_app",
                                                   user_id="user_123",
                                                   session_id="session_123")
    
    print(f"Session created with ID: {session.id}")

    runner = Runner(agent=root_agent, 
                    app_name="basic_agent_app",
                    session_service=session_service)
    
    while True:
        # Use asyncio.to_thread to prevent blocking the event loop
        user_input = await asyncio.to_thread(input, "You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Exiting chat loop. Goodbye!")
            break
        
        new_message = Content(role="user", parts=[Part(text=user_input)])
        events = runner.run_async(
            user_id="user_123",
            session_id=session.id,
            new_message=new_message)
       
        final_response = ""
        event_no = 0
        try:
            async for event in events:
                event_no += 1
                print_json_response(event, f"==== Event {event_no} ====")
                
                if event.is_final_response():
                    final_response = event.content.parts[0].text
                    print(f"\n agent response: {final_response}\n")
                    break
        finally:
            # Explicitly close the async generator to ensure proper cleanup
            await events.aclose()

def print_json_response(response: any, title:str) -> None:
    print(f"\n====== {title} =====")
    try:
        if hasattr(response, 'root'):
            data = response.root.model_dump(mode="json", exclude_none=True)
        else:
            data = response.model_dump(mode="json", exclude_none=True)

        json_str = json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        rprint(f"[red bold]Error printing response to JSON:[/red bold] {e}")
        rprint(repr(response))
                    

if __name__ == "__main__":
    asyncio.run(chat_loop())