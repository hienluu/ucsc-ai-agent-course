import os
import types
from google.adk.agents.llm_agent import Agent
from google.genai import types
from google.adk.planners import BuiltInPlanner
from google.adk.models.lite_llm import LiteLlm

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

from dotenv import load_dotenv
load_dotenv()


LLM_PROVIDER_API_KEY = os.getenv("LLM_PROVIDER_API_KEY")
LLM_PROVIDER_MODEL = os.getenv("LLM_PROVIDER_MODEL")
LLM_PROVIDER_URL = os.getenv("LLM_PROVIDER_URL")

if LLM_PROVIDER_API_KEY is None:
    raise ValueError("LLM_PROVIDER_API_KEY is not set in the environment variables.")

print(f"LLM_PROVIDER_MODEL: {LLM_PROVIDER_MODEL}")
print(f"LLM_PROVIDER_URL: {LLM_PROVIDER_URL}")

litellm = LiteLlm(
        model=f"{LLM_PROVIDER_MODEL}",
        api_base=f"{LLM_PROVIDER_URL}",
        api_key=f"{LLM_PROVIDER_API_KEY}",
        include_reasoning=False, # Set to False to exclude reasoning steps from the response
)

root_agent = Agent(
    model=litellm,
    planner=BuiltInPlanner(        
        thinking_config=types.ThinkingConfig(    
            include_thoughts=False,            
            thinking_budget=0,        
        ),
    ),
    name="open_source_model_agent",
    instruction="You are a helpful assistant with a sense of humor and openness.",
)
