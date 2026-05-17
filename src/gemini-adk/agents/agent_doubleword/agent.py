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


litellm = LiteLlm(
        model="openai/Qwen/Qwen3.6-35B-A3B-FP8",
        api_base="https://api.doubleword.ai/v1",
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
    name="my_agent",
    instruction="You are a helpful assistant.",
)
