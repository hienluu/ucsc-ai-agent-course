# Claude Function Calling Examples

This module demonstrates how to use Claude with function calling (tool use) capabilities.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r ../../requirements.txt
   # or
   pip install anthropic
   ```

2. **Set up environment variables**:
   ```bash
   cp ../../env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

3. **Get an Anthropic API Key**:
   - Go to [Anthropic Console](https://console.anthropic.com)
   - Create an API key
   - Add it to your `.env` file as `ANTHROPIC_API_KEY`

## Requirements

- Python 3.14+
- Anthropic Claude API key
- `anthropic` package (tested with 0.97.0+)
- Dependencies listed in `../../pyproject.toml`

## Usage

### Manual Function Calling
Claude determines when to call tools and you handle the results:

```bash
uv run -m tools.claude.claude --mode "manual" --prompt "What is the temperature in London?"
```

### Automatic Function Calling
Claude automatically selects and calls tools using the beta tool runner:

```bash
uv run -m tools.claude.claude --mode "auto" --prompt "What is the weather in London and Paris?"
```

This mode uses the `@anthropic.beta_tool` decorator to create tools that Claude can automatically execute.

### Custom Prompt
```bash
uv run -m tools.claude.claude --mode "manual" --prompt "What is 15 + 27?" --model "claude-3-5-sonnet-20241022"
```

## Available Tools

- **get_weather(city)** - Get weather information for a city
- **calculate_sum(a, b)** - Calculate the sum of two numbers
- **get_current_time()** - Get the current time with AM/PM format
- **get_location(...)** - Get latitude and longitude for a location

## Examples

The module provides two example functions:

1. **local_tool_example()** - Uses manual function calling where the client handles all tool execution
2. **local_tool_auto_func_calling_example()** - Uses automatic function calling with Claude

Both examples work with the tool functions defined in `../tool_functions.py`.

## Resources

- [Anthropic Function Calling Guide](https://docs.anthropic.com/en/docs/build-a-system-prompt)
- [Tool Use Documentation](https://docs.anthropic.com/en/docs/concepts/tool-use)
- [Claude Python SDK](https://github.com/anthropics/anthropic-sdk-python)
