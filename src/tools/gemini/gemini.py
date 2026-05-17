import os
import json
from urllib import response
import argparse

from ..tool_functions import get_location, get_current_time, get_weather, calculate_sum
    
from dotenv import load_dotenv
from typing import Any, Dict, List, Callable
from google import genai
from google.genai import types




# function calling examples
# Resources:
# * https://ai.google.dev/gemini-api/docs/function-calling
# * https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_function_calling.ipynb
#
#

def get_client() -> genai.Client:
    """Get a configured Google GenAI client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


def create_tool_from_function(func: Callable) -> types.Tool:
    """
    Create a Google GenAI tool from a Python function.

    Args:
        func: A Python function with type hints and docstring

    Returns:
        A Google GenAI Tool object
    """
    import inspect

    # Get function signature
    sig = inspect.signature(func)
    params = {}

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

        # Use parameter docstring if available
        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default

        params[name] = param_info

    # Create the function declaration
    func_name = func.__name__
    print(f"DEBUG: Original function name: {repr(func_name)}")
    
    # Ensure the function name is valid
    import re
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.:-]*$', func_name):
        print(f"WARNING: Function name {repr(func_name)} is invalid, using 'custom_function'")
        func_name = 'custom_function'
    
    function_declaration = types.FunctionDeclaration(
        name=func_name,
        description=func.__doc__ or f"Function {func_name}",
        parameters={
            "type": "object",
            "properties": params,
            "required": [
                name for name, param in sig.parameters.items()
                if param.default == inspect.Parameter.empty
            ]
        }
    )
    
    print(f"DEBUG: Created function declaration with name: {repr(function_declaration.name)}")
    print(f"DEBUG: Function name length: {len(function_declaration.name)}")

    return types.Tool(function_declarations=[function_declaration])


def execute_tool_call(tool_call: types.FunctionCall, tools: Dict[str, Callable]) -> Any:
    """
    Execute a tool call from the AI model.

    Args:
        tool_call: The function call from the AI
        tools: Dictionary mapping function names to callable functions

    Returns:
        The result of executing the function
    """
    func_name = tool_call.name
    if func_name not in tools:
        raise ValueError(f"Unknown tool function: {func_name}")

    func = tools[func_name]
    args = tool_call.args if hasattr(tool_call, 'args') else {}

    print(f">>> Executing tool manually: {func_name} with args: {args}")
    try:
        return func(**args)
    except Exception as e:
        return f"Error executing {func_name}: {str(e)}"


def generate_with_tools(
    client: genai.Client,
    prompt: str,
    tools: List[types.Tool],
    tool_functions: Dict[str, Callable],
    model: str
) -> str:
    """
    Generate a response using tools.

    Args:
        client: Google GenAI client
        prompt: The user prompt
        tools: List of tool schemas
        tool_functions: Dictionary of function name to callable
        model: Model name to use

    Returns:
        The final response after tool execution
    """
    # Generate initial response
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="AUTO"
                )
            )
        )
    )

    # Check if there are tool calls
    if not response.candidates or not response.candidates[0].content.parts:
        return "No response generated"

    # Collect all function calls and text from the initial response
    function_calls = []
    initial_text = ""

    for part in response.candidates[0].content.parts:
        if hasattr(part, 'text') and part.text:
            initial_text += part.text
        elif hasattr(part, 'function_call'):
            function_calls.append(part.function_call)

    # If there are function calls, execute them all and generate follow-up
    if function_calls:
        # Execute all tool calls
        tool_results = []
        for tool_call in function_calls:
            tool_result = execute_tool_call(tool_call, tool_functions)
            tool_results.append(f"Tool result for {tool_call.name}: {tool_result}")

        # Combine all tool results
        combined_results = "\n".join(tool_results)

        # Generate follow-up response with all tool results
        follow_up_response = client.models.generate_content(
            model=model,
            contents=[
                prompt,
                types.Content(
                    role="model",
                    parts=[types.Part(text=initial_text)] if initial_text else []
                ),
                types.Content(
                    role="user",
                    parts=[types.Part(text=f"Tool results:\n{combined_results}")]
                )
            ],
            config=types.GenerateContentConfig(tools=tools)
        )

        # Extract final response
        final_response = initial_text  # Start with any initial text
        if follow_up_response.candidates and follow_up_response.candidates[0].content.parts:
            for part in follow_up_response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    final_response += part.text

        return final_response
    else:
        # No function calls, return the initial response
        return initial_text


def local_tool_example(model: str, prompt: str = "What's the temperature in London?") -> None:
    print(">>> Running local tool example")
    try:
        # Initialize client
        client = get_client()
        print("✓ Connected to Google GenAI")

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

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Make sure GOOGLE_API_KEY is set in environment variables")
    except Exception as e:
        print(f"Unexpected error: {e}")

def local_tool_auto_func_calling_example(model: str, prompt: str = "What's the temperature in London?") -> None:
    print(f"Testing auto-function calling with model: {model} and prompt: {prompt}")
    try:
        # Initialize client
        client = get_client()
        print("✓ Connected to Google GenAI")

        # tool list
        # pass a Python function directly and it will be automatically called and responded by default.
        tool_functions = [get_weather, calculate_sum, get_current_time, get_location]
        
        config = types.GenerateContentConfig(
            tools=tool_functions,  # Pass the tool functions directly
            temperature=0,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="AUTO"
                )
            )
        ) 

        # Make the request
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        print(response.text)  

        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Make sure GOOGLE_API_KEY is set in environment variables")
    except Exception as e:
        print(f"Unexpected error: {e}")

def builtin_tool_example(model: str, prompt: str = "What's the latest Google stock price?") -> None:
    print(">>> Running built-in tool example")
    try:
        # Initialize client
        client = get_client()
        print("✓ Connected to Google GenAI")

        try:
            # Make the request
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(google_search=types.GoogleSearch())  # Only built-in tool for now
                    ]
                ),
            )

            print(f"***** Response with tool calls start *****")
            for part in response.candidates[0].content.parts:
                if part.tool_call:
                    print(f"Tool call: {part.tool_call.tool_type} (ID: {part.tool_call.id})")
                if part.tool_response:
                    print(f"Tool response: {part.tool_response.tool_type} (ID: {part.tool_response.id})")
                if part.function_call:
                    print(f"Function call: {part.function_call.name} (ID: {part.function_call.id})")
                # Print any grounding metadata attached to the part
                if hasattr(part, "grounding_metadata") and part.grounding_metadata is not None:
                    print("Grounding metadata:", part.grounding_metadata)
                if getattr(part, "tool_response", None) is not None and hasattr(part.tool_response, "grounding_metadata"):
                    print("Tool response grounding metadata:", part.tool_response.grounding_metadata)
                if getattr(part, "tool_call", None) is not None and hasattr(part.tool_call, "grounding_metadata"):
                    print("Tool call grounding metadata:", part.tool_call.grounding_metadata)


            print(f"***** Response grounding metadata *****")
            for candidate in response.candidates:
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata.web_search_queries is not None:
                    print("Candidate grounding metadata web_search_queries:", candidate.grounding_metadata.web_search_queries)

            print(f"***** Response with tool calls end *****")

            print(f"========== Response ========== ")
            for part in response.candidates[0].content.parts:
                if part.text:
                    print(part.text)

        except Exception as e:
            print(f"Error: {e}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Make sure GOOGLE_API_KEY is set in environment variables")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    # main function to demonstrate the gemini tools functionality
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Google GenAI function calling')
    parser.add_argument('--prompt', '-p', 
                       default="What's the temperature in London?",
                       help='Prompt to send to the AI model')
    parser.add_argument('--mode', '-m', 
                       choices=['auto', 'manual', 'builtin'], 
                       default='manual',
                       help='Function calling mode (auto or manual, - default: manual)')
    parser.add_argument('--model', 
                       default=None,
                       help='Gemini model to use (overrides GEMINI_MODEL env var)')
    
    args = parser.parse_args()
    
    load_dotenv()
    gemini_model = args.model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    print(f"Using Gemini model: {gemini_model}")
    print(f"Prompt: {args.prompt}")
    print(f"Mode: {args.mode}")

    if args.mode == "auto":
        local_tool_auto_func_calling_example(gemini_model, args.prompt)
    elif args.mode == "manual":        
        local_tool_example(gemini_model, args.prompt)
    elif args.mode == "builtin":
        builtin_tool_example(gemini_model, args.prompt)
    else:
        print(f"Unknown mode: {args.mode}. Please choose 'auto', 'manual', or 'builtin'.")
    #local_tool_example(gemini_model)
    #local_tool_auto_func_calling_example(gemini_model, args.prompt, args.mode)
    #builtin_tool_example(gemini_model)
    


if __name__ == "__main__":
    main()