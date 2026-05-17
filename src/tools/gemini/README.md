# Tools Examples

This folder demonstrates how to use function calling tools with Google's Generative AI (GenAI) library

## Features

- **Tool Creation**: Automatically create GenAI tool schemas from Python functions
- **Function Calling**: Enable AI models to invoke external tools
- **CLI Mode Support**: Select `manual`, `auto`, or `builtin` invocation modes
- **Built-in Tool Diagnostics**: Print tool calls, responses, and grounding metadata



## Usage

Run the main CLI entrypoint:

```bash
uv run -m tools.gemini.gemini --prompt "What's the weather in San Francisco?" --mode auto
```

Available modes:
- `manual`: uses custom tool schemas generated from Python functions
- `auto`: uses Python callables directly for automatic function calling
- `builtin`: uses built-in Gemini tools such as Google Search

Example with built-in tools and grounding metadata output:

```bash
uv run -m tools.gemini.gemini --prompt "What's the latitude and longitude of San Jose University?" --mode builtin
```


## Key Components

### Gemini module (`tools/gemini/gemini.py`)

- `get_client()`: Reads `GOOGLE_API_KEY` and returns a configured GenAI client
- `create_tool_from_function()`: Builds `FunctionDeclaration` objects for custom Python tools
- `execute_tool_call()`: Runs model-suggested tool calls locally
- `generate_with_tools()`: Sends prompts and handles tool-enabled responses
- `builtin_tool_example()`: Demonstrates built-in tool usage and prints grounding metadata

### Example Tools (`tools/tool_functions.py`)

The example includes mock tools:
- `get_weather(city)`: Returns a weather string
- `calculate_sum(a, b)`: Adds two numbers
- `get_current_time()`: Returns the current timestamp
- `get_location(...)`: Returns a location lookup payload

## How It Works

1. **Tool Definition**: Python functions are converted to tool schemas or passed as callables.
2. **AI Generation**: The model receives the prompt and the available tools.
3. **Function Calling**: The model may call a tool directly.
4. **Result Handling**: Tool outputs are integrated into the final response.
