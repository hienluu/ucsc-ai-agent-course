import os
import json
from typing import Any, Dict, List, Callable
import argparse

from ..tool_functions import get_location, get_current_time, get_weather, calculate_sum

from dotenv import load_dotenv
from anthropic import Anthropic, beta_tool


# Function calling examples with Claude (Anthropic)
# Resources:
# * https://docs.anthropic.com/en/docs/build-a-system-prompt
# * https://github.com/anthropics/anthropic-sdk-python
# * https://github.com/anthropics/claude-cookbooks/tree/main/tool_use
# * https://docs.anthropic.com/en/docs/concepts/tool-use


def get_client() -> Anthropic:
    """Get a configured Anthropic Claude client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    return Anthropic(api_key=api_key)


def create_tool_from_function(func: Callable) -> Dict:
    """
    Create a Claude tool schema from a Python function.

    Args:
        func: A Python function with type hints and docstring

    Returns:
        A Claude tool schema dictionary
    """
    import inspect

    # Get function signature
    sig = inspect.signature(func)
    params = {}
    required_params = []

    for name, param in sig.parameters.items():
        param_info = {
            "type": "string",  # Default type
            "description": f"Parameter {name}"
        }

        # Try to infer type from annotation
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == int:
                param_info["type"] = "integer"
            elif param.annotation == float:
                param_info["type"] = "number"
            elif param.annotation == bool:
                param_info["type"] = "boolean"
            elif param.annotation == list:
                param_info["type"] = "array"
            # Add more type mappings as needed

        # Track required parameters (those without defaults)
        if param.default == inspect.Parameter.empty:
            required_params.append(name)

        params[name] = param_info

    # Create the tool schema for Claude
    tool_schema = {
        "name": func.__name__,
        "description": func.__doc__ or f"Function {func.__name__}",
        "input_schema": {
            "type": "object",
            "properties": params,
            "required": required_params
        }
    }

    #print(f"----- Created tool schema for function '{func.__name__} -------")
    #print(f"{json.dumps(tool_schema, indent=2)}\n---")  # Separator for readability

    return tool_schema


def execute_tool_call(tool_name: str, tool_args: Dict[str, Any], tools: Dict[str, Callable]) -> Any:
    """
    Execute a tool call from the AI model.

    Args:
        tool_name: The name of the tool function to execute
        tool_args: Dictionary of arguments to pass to the function
        tools: Dictionary mapping function names to callable functions

    Returns:
        The result of executing the function
    """
    print(f">>> Executing tool: {tool_name} with args: {tool_args}")
    if tool_name not in tools:
        raise ValueError(f"Unknown tool function: {tool_name}")

    func = tools[tool_name]

    try:
        return func(**tool_args)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def generate_with_tools(
    client: Anthropic,
    prompt: str,
    tools: List[Dict],
    tool_functions: Dict[str, Callable],
    model: str
) -> str:
    """
    Generate a response using tools with manual function calling.

    Args:
        client: Anthropic client
        prompt: The user prompt
        tools: List of tool schemas
        tool_functions: Dictionary of function name to callable
        model: Model name to use

    Returns:
        The final response after tool execution
    """
    messages = [{"role": "user", "content": prompt}]

    # Initial request with tools
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # Process tool calls in a loop until there are no more
    while response.stop_reason == "tool_use":
        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response.content})
        
        # Process each tool use block
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_args = block.input
                
                # Execute the tool
                tool_result = execute_tool_call(tool_name, tool_args, tool_functions)
                print(f"Tool result: {tool_result}")
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(tool_result)
                })
        
        # Add tool results to messages
        messages.append({"role": "user", "content": tool_results})
        
        # Generate follow-up response
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
    
    # Extract final text response
    final_response = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_response += block.text
    
    return final_response


def local_tool_example(model: str, prompt: str = "What's the temperature in London?") -> None:
    """Example using local tools with manual function calling."""
    print(">>> Running local tool example")
    try:
        # Initialize client
        client = get_client()
        print("✓ Connected to Claude")

        # Create tools
        tool_functions = {
            "get_weather": get_weather,
            "calculate_sum": calculate_sum,
            "get_current_time": get_current_time,
            "get_location": get_location
        }

        tools = [create_tool_from_function(func) for func in tool_functions.values()]
        print(f"✓ Created {len(tools)} tools")

        print(f"\n--- Testing: {prompt} ---")
        try:
            response = generate_with_tools(
                client=client,
                prompt=prompt,
                tools=tools,
                tool_functions=tool_functions,
                model=model
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in environment variables")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def local_tool_auto_func_calling_example(model: str, prompt: str = "What's the temperature in London?") -> None:
    """Example using local tools with automatic tool calling."""
    print(f"Testing automatic function calling with model: {model} and prompt: {prompt}")
    try:
        # Initialize client
        client = get_client()
        print("✓ Connected to Claude")

        # Create tools using beta_tool decorator
        @beta_tool
        def get_weather_tool(city: str) -> str:
            """Get weather for a city."""
            return get_weather(city)
        
        @beta_tool  
        def calculate_sum_tool(a: int, b: int) -> int:
            """Calculate the sum of two numbers."""
            return calculate_sum(a, b)
            
        @beta_tool
        def get_current_time_tool() -> str:
            """Get the current time."""
            return get_current_time()
            
        @beta_tool
        def get_location_tool(**kwargs) -> list[dict]:
            """Get latitude and longitude for a given location."""
            return get_location(**kwargs)

        tools = [get_weather_tool, calculate_sum_tool, get_current_time_tool, get_location_tool]

        messages = [{"role": "user", "content": prompt}]

        # Make the request
        response = client.beta.messages.tool_runner(
            model=model,
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
       
        for message in response:
            #print(f"message content: {message.content}")
            for content in message.content:
                if content.type == "text":
                    print(f"Response: {content.text}")
                elif content.type == "tool_use":                    
                    print(f"Tool called: {content.name} with input: {content.input}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in environment variables")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to demonstrate Claude tools functionality."""
    parser = argparse.ArgumentParser(description='Test Claude function calling')
    parser.add_argument('--prompt', '-p',
                       default="What's the temperature in London?",
                       help='Prompt to send to the AI model')
    parser.add_argument('--mode', '-m',
                       choices=['auto', 'manual'],
                       default='manual',
                       help='Function calling mode (auto or manual, default: manual)')
    parser.add_argument('--model',
                       default=None,
                       help='Claude model to use (overrides CLAUDE_MODEL env var)')

    args = parser.parse_args()

    load_dotenv()
    claude_model = args.model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
    print(f"Using Claude model: {claude_model}")
    print(f"Prompt: {args.prompt}")
    print(f"Mode: {args.mode}")

    if args.mode == "auto":
        local_tool_auto_func_calling_example(claude_model, args.prompt)
    elif args.mode == "manual":
        local_tool_example(claude_model, args.prompt)
    else:
        print(f"Unknown mode: {args.mode}. Please choose 'auto' or 'manual'.")


if __name__ == "__main__":
    main()
