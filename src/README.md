# UCSC Extension - [Developing AI Agent Applications Course](https://www.ucsc-extension.edu/courses/developing-ai-agent-applications)

## Setup Steps

1. **Install dependencies** (this project uses [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```

2. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

3. **Get a Google API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/api-keys?project=gen-lang-client-0014961850)
   - Create a new API key
   - Add it to your `.env` file

4. **Run a script**:
   ```bash
   uv run python tools/gemini/gemini.py
   ```



## Project Structure

```
src/
├── tools/
│   ├── gemini/                 # Gemini tool example and CLI
│   │   └── gemini.py
│   ├── claude/                 # Claude tool example
│   │   └── claude.py
│   ├── openai/                 # OpenAI tool example
│   │   └── openai.py
│   └── tool_functions.py       # Example tool functions
├── gemini-adk/                 # Google ADK agent examples
│   ├── agent_runner.py
│   └── agents/
├── pyproject.toml              # Project configuration
├── env.example                 # Environment variables template
└── README.md                   # This file
```

## Requirements

- Python 3.13+
- Google GenAI API key
- Dependencies listed in `pyproject.toml`

## Basic ADK command example
- `adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY`
