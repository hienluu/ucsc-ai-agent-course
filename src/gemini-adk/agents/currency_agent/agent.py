import os
import types
from google.adk.agents.llm_agent import Agent, LlmAgent
from google.genai import types
from google.adk.planners import BuiltInPlanner
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset, McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StreamableHTTPConnectionParams
from mcp import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

from dotenv import load_dotenv
load_dotenv()


DOUBLEWORD_API_KEY = os.getenv("DOUBLEWORD_API_KEY")
DOUBLEWORD_MODEL = os.getenv("DOUBLEWORD_MODEL")
DOUBLEWORD_API_URL = os.getenv("DOUBLEWORD_API_URL")

if DOUBLEWORD_API_KEY is None:
    raise ValueError("DOUBLEWORD_API_KEY is not set in the environment variables.")

print(f"DOUBLEWORD_MODEL: {DOUBLEWORD_MODEL}")
print(f"DOUBLEWORD_API_URL: {DOUBLEWORD_API_URL}")

SYSTEM_INSTRUCTION = (
    "You are a specialized assistant for currency conversions. "
    "Your sole purpose is to use the 'get_exchange_rate' tool to answer questions about currency exchange rates. "
    "If the user asks about anything other than currency conversion or exchange rates, "
    "politely state that you cannot help with that topic and can only assist with currency-related queries. "
    "Do not attempt to answer unrelated questions or use tools for other purposes."
)

current_exchange_rate_mcp_tool = MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
            )
        ) 

litellm = LiteLlm(
        model=f"{DOUBLEWORD_MODEL}",
        api_base=f"{DOUBLEWORD_API_URL}",
        api_key=f"{DOUBLEWORD_API_KEY}",
        include_reasoning=False, # Set to False to exclude reasoning steps from the response
)

root_agent = LlmAgent(
    model=litellm,
    planner=BuiltInPlanner(        
        thinking_config=types.ThinkingConfig(    
            include_thoughts=False,            
            thinking_budget=0,        
        ),
    ),
    name="open_source_currency_agent",
    description="An agent that can help with currency conversions",
    instruction=SYSTEM_INSTRUCTION,
    tools=[current_exchange_rate_mcp_tool]
)
