from google.adk.agents import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent

from agent_quality.weather_tool import get_weather_forecast
from shared.model import get_model

from langsmith.integrations.google_adk import configure_google_adk

# Configure LangSmith tracing
configure_google_adk()

MODEL = get_model()

weather_agent = LlmAgent(
    model=MODEL,
    name="weather_agent",
    description="Looks up today's weather forecast for a user-provided location.",
    instruction="""\
You are the weather assistant. The user mentions a location (city/region) and asks
about today's weather or what to wear.

Steps:
1. Extract the location from the user's message.
2. Call the `get_weather_forecast` tool with that location.
3. Return a concise JSON object summarizing the forecast for the next agent:

{
  "location": <resolved place>,
  "current_temp_f": <number>,
  "temp_max_f": <number>,
  "temp_min_f": <number>,
  "precipitation_mm": <number>,
  "precipitation_probability_pct": <number>,
  "wind_speed_kmh": <number>,
  "humidity_pct": <number>,
  "conditions": <short text>
}

If the tool returns an error, return exactly: {"error": "<message>"} and do NOT
invent forecast values.
""",
    tools=[get_weather_forecast],
    output_key="forecast",
)

outfit_agent = LlmAgent(
    model=MODEL,
    name="outfit_agent",
    description="Recommends an outfit based on the forecast in session state.",
    instruction="""\
Forecast is in session state under `forecast`:
{forecast}

If the forecast contains an "error" field, apologize briefly and report the error.
Do NOT make a clothing recommendation in that case.

Otherwise, recommend a complete outfit for the day. Cover:
- Layers (base / mid / outer) tuned to temp_min_f and temp_max_f.
- Rain gear when precipitation_mm > 1 OR precipitation_probability_pct > 40.
- Wind protection when wind_speed_kmh > 25.
- Footwear suited to conditions.
- One small accessory (hat, sunglasses, gloves) when relevant.

Format the response as Markdown:
- A one-line header naming the resolved city, e.g. "**Today in Seattle**: rainy and cool".
- A 1-2 sentence "Reasoning" paragraph tying picks to specific forecast values.
- An "Outfit" section with bullet points.
""",
    output_key="recommendation",
)

root_agent = SequentialAgent(
    name="weather_outfit_pipeline",
    description=(
        "Two-agent sequential pipeline: weather lookup -> outfit recommendation. "
        "Designed to illustrate observability and evaluation of a tool-using agent."
    ),
    sub_agents=[weather_agent, outfit_agent],
)
