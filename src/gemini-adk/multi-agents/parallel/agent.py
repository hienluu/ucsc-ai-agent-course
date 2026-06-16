from google.adk.agents import ParallelAgent, SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import google_search

from shared.model import get_model, get_gemini_model
from shared.types import Outline, Section

MODEL = get_model()

# researcher focuses on AI and ML trends.
tech_researcher = LlmAgent(
    name="TechResearcher",
    model=get_gemini_model(),
    instruction="""Research the latest AI/ML trends. Include 3 key developments,
                   the main companies involved, and the potential impact. Keep the report very concise (100 words).""",
    tools=[google_search],
    output_key="tech_research",  
)

health_researcher = LlmAgent(
    name="HealthResearcher",
    model=get_gemini_model(),
    instruction="""Research recent medical breakthroughs. Include 3 significant advances,
                   their practical applications, and estimated timelines. Keep the report concise (100 words).""",
    tools=[google_search],
    output_key="health_research",  
)

finance_researcher = LlmAgent(
    name="FinanceResearcher",
    model=get_gemini_model(),
    instruction="""Research current fintech trends. Include 3 key trends,
                   their market implications, and the future outlook. Keep the report concise (100 words).""",
    tools=[google_search],
    output_key="finance_research",  # The result will be stored with this key.
)

# aggregator agent that takes the outputs from the three parallel researchers and synthesizes them into a single executive summary.
aggregator_agent = LlmAgent(
    name="AggregatorAgent",
    model=MODEL,
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these three research findings into a single executive summary:

    **Technology Trends:**
    {tech_research}
    
    **Health Breakthroughs:**
    {health_research}
    
    **Finance Innovations:**
    {finance_research}
    
    Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words.""",
    output_key="executive_summary",  # This will be the final output of the entire system.
)

# runs all its sub-agents simultaneously.
parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[tech_researcher, health_researcher, finance_researcher],
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_research_team, aggregator_agent],
)