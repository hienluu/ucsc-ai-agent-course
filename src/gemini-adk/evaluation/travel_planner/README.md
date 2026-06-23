# Travel Planner Agent (Google ADK)

**Your AI co-pilot for end-to-end trip planning - from weather and budget to packing lists and day-by-day itineraries, built on the Google Agent Development Kit.**

## Overview

This is the [Google ADK](https://google.github.io/adk-docs/) port of the original LangChain-based [`travel_planner_agent`](../travel_planner_agent). It keeps the same conversational behavior and tool set, but swaps the agent runtime for ADK's `LlmAgent` + `Runner` pattern and the LLM for Gemini.

Describe where you want to go, when, your budget, and what you enjoy - the agent gathers missing details, researches your destination with live tools, and returns a structured travel plan.

## Features

- **Conversational trip planning** - describe trips in plain English; the agent asks follow-up questions when details are missing
- **Live weather data** - current conditions and a 5-day forecast for any destination via OpenWeatherMap
- **Budget and currency tools** - convert amounts between currencies and estimate trip costs via ExchangeRate-API
- **Destination research** - attractions, food, culture, and travel tips sourced from DuckDuckGo web search
- **Smart packing lists** - recommendations tailored to forecast, climate, and trip length
- **Structured travel plans** - overview, weather, budget, highlights, packing, and a suggested day-by-day itinerary
- **Session-based chat history** - multi-turn conversations preserved via ADK's `InMemorySessionService`

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent Orchestration | [Google ADK](https://google.github.io/adk-docs/) (`LlmAgent` + `Runner`) |
| LLM | [Gemini](https://ai.google.dev/) via [`google-genai`](https://pypi.org/project/google-genai/) (default `gemini-2.5-flash`) |
| Weather | [OpenWeatherMap API](https://openweathermap.org/api) |
| Currency | [ExchangeRate-API](https://www.exchangerate-api.com/) |
| Web Search | [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) |
| UI | [Streamlit](https://streamlit.io/) chat interface |

## What changed vs. the LangChain version

| Concern | LangChain version | ADK version |
|---------|-------------------|-------------|
| Agent loop | `langchain.agents.create_agent` | `google.adk.agents.LlmAgent` |
| Tool definition | `@tool` decorator on functions | Plain Python functions with type hints + docstrings |
| Invocation | `agent.invoke({"messages": [...]})` | `Runner.run_async(...)` streaming events |
| Conversation state | List of `HumanMessage`/`AIMessage` rebuilt per turn | ADK session, persisted in `InMemorySessionService` |
| LLM provider | Orq.ai router (`deepseek-v4-flash`) | Google AI Studio (`gemini-2.5-flash`) |
| Required env vars | `ORQ_API_KEY` | `GOOGLE_API_KEY` |

## Prerequisites

- **Python 3.10 or higher**
- [**uv**](https://docs.astral.sh/uv/) for dependency and environment management
- API keys for:
  - [Google AI Studio](https://aistudio.google.com/app/apikey) - Gemini model access
  - [OpenWeatherMap](https://openweathermap.org/api) - weather and forecast data
  - [ExchangeRate-API](https://www.exchangerate-api.com/) - currency conversion


## Additional Environment Variables

| Variable | Description | Where to Get It |
|----------|-------------|-----------------|
| `OPENWEATHER_API_KEY` | Fetches current weather and 5-day forecasts for destinations | [OpenWeatherMap](https://home.openweathermap.org/api_keys) → API keys |
| `EXCHANGERATE_API_KEY` | Converts currencies and returns live exchange rates | [ExchangeRate-API](https://www.exchangerate-api.com/) → Get Free Key |


## Usage

### Run the app

cd into travel_planner directory 

```bash
uv run streamlit run app.py
```

Streamlit opens the app in your browser (typically at `http://localhost:8501`). Use the chat input at the bottom of the page to describe your trip.

### Langsmith

* To enable tracing, make sure to set LANGSMITH_TRACING to "true" in .env file
* URL: https://smith.langchain.com

### Example user inputs

| Scenario | Example prompt |
|----------|----------------|
| Full trip plan | *Plan a 7-day trip to Tokyo in April. My budget is $3,000 USD. I love food, temples, and street photography.* |
| Weather check | *What's the weather like in Barcelona? I'm visiting for 5 days in June.* |
| Currency conversion | *Convert 2,500 USD to EUR and THB for a two-week trip across Europe and Thailand.* |
| Packing help | *I'm going to Iceland for 8 days in February. What should I pack based on the weather?* |

### What the agent returns

Once it has enough context, the agent calls its tools and delivers a structured plan with:

1. **Trip Overview** - destination, dates, duration, and traveler preferences
2. **Weather** - current conditions and a forecast summary from OpenWeatherMap
3. **Budget & Currency** - cost estimates and currency conversions from ExchangeRate-API
4. **Destination Highlights** - attractions, dining, culture, and travel tips from web research
5. **Packing List** - clothing and essentials based on weather and trip length
6. **Suggested Itinerary** - a day-by-day outline aligned with your interests and schedule

The agent uses tools for all factual data (weather, rates, research) rather than guessing. If details are missing, it will ask clarifying questions before building the full plan.

Use **Clear conversation** in the sidebar to start a new trip from scratch (this also rotates the ADK session id).

## Project Structure

```
adk_travel_planner_agent/
├── app.py              # Streamlit UI + ADK Runner wiring
├── agent.py            # ADK LlmAgent definition and tool functions
└── README.md
```
