from google.adk.agents.llm_agent import Agent, LlmAgent
from google.adk.tools import google_search

root_agent = LlmAgent(
    model='gemini-3.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge. You have access to only one tool: google_search',
    tools=[google_search]
)
