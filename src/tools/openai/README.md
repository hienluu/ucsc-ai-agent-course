# OpenAI Function Calling Examples

This directory contains examples demonstrating how to use the OpenAI API with function calling (tool use).

## Setup

1. Ensure you have the OpenAI API key set:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. The examples use the following environment variables (optional):
- `OPENAI_MODEL`: The model to use (default: `gpt-4o-mini`)

## Features

### Local Tools
The examples demonstrate how to use local functions as tools:
- `get_weather`: Get weather information for a location
- `get_location`: Get location information
- `get_current_time`: Get the current time
- `calculate_sum`: Calculate the sum of numbers

### Modes

#### Manual Mode (Default)
Uses explicit function calling with manual tool execution:
```bash
python -m tools.openai.openai --mode manual --prompt "What's the temperature in London?"
```

#### Auto Mode
Uses automatic function calling with OpenAI's auto mode:
```bash
python -m tools.openai.openai --mode auto --prompt "What's the temperature in London?"
```

## Usage

### Basic Example
```bash
python -m tools.openai.openai
```

### Custom Prompt
```bash
python -m tools.openai.openai --prompt "Calculate 5 + 3"
```

### Specify Model
```bash
python -m tools.openai.openai --model gpt-4 --prompt "What time is it?"
```

## API Reference

### `get_client() -> OpenAI`
Returns a configured OpenAI client using the `OPENAI_API_KEY` environment variable.

### `create_tool_from_function(func: Callable) -> Dict`
Converts a Python function into an OpenAI tool schema.

### `execute_tool_call(tool_name: str, tool_args: Dict, tools: Dict[str, Callable]) -> Any`
Executes a tool call returned by the OpenAI API.

### `generate_with_tools(client: OpenAI, prompt: str, tools: List[Dict], tool_functions: Dict[str, Callable], model: str) -> str`
Generates a response using manual function calling mode.

## Comparison with Gemini

| Feature | Gemini | OpenAI |
|---------|--------|--------|
| Client | `genai.Client` | `openai.OpenAI` |
| Tool Creation | `types.FunctionDeclaration` | Dictionary schema |
| API Call | `client.models.generate_content()` | `client.chat.completions.create()` |
| Response Format | `response.candidates[0].content.parts` | `response.choices[0].message` |
| Built-in Tools | Google Search | Not implemented |

## Resources

- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [OpenAI Models](https://platform.openai.com/docs/models)
