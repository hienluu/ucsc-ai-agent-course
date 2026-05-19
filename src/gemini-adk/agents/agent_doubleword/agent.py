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


DOUBLEWORD_API_KEY = os.getenv("DOUBLEWORD_API_KEY")
DOUBLEWORD_MODEL = os.getenv("DOUBLEWORD_MODEL")
DOUBLEWORD_API_URL = os.getenv("DOUBLEWORD_API_URL")

if DOUBLEWORD_API_KEY is None:
    raise ValueError("DOUBLEWORD_API_KEY is not set in the environment variables.")

print(f"DOUBLEWORD_MODEL: {DOUBLEWORD_MODEL}")
print(f"DOUBLEWORD_API_URL: {DOUBLEWORD_API_URL}")

litellm = LiteLlm(
        model=f"{DOUBLEWORD_MODEL}",
        api_base=f"{DOUBLEWORD_API_URL}",
        api_key=f"{DOUBLEWORD_API_KEY}",
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
