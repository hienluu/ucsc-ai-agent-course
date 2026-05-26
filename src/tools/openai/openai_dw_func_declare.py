import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI, base_url

from ..tool_functions import get_location, get_weather
from .openai import create_tool_from_function

"""
This example use the model from DoubleWord, which is a wrapper around OpenAI's function calling. 
The code structure is similar to the Gemini example, but adapted for OpenAI's API and tools.

Make sure to set the DOUBLEWORD_API_KEY, DOUBLEWORD_MODEL in your .env file before running this example.
"""

def get_client() -> OpenAI:
    """Get a configured OpenAI client."""
    dw_api_key = os.getenv("DOUBLEWORD_API_KEY")
    if not dw_api_key:
        raise ValueError("DOUBLEWORD_API_KEY environment variable is required")
    
    dw_base_url = os.getenv("DOUBLEWORD_API_URL")
    if not dw_base_url:
        raise ValueError("DOUBLEWORD_API_URL environment variable is required")
    
    print(f"Using DoubleWord API with base URL: {dw_base_url}")
    return OpenAI(api_key=dw_api_key, base_url=dw_base_url)


# 1. Define the Python function the model can call
def get_stock_price(ticker: str, currency: str = "USD") -> str:
    """Retrieves the current stock price for a given ticker symbol.

    Args:
        ticker: The stock symbol (e.g., 'GOOG', 'AAPL').
        currency: The currency to return the price in. Defaults to 'USD'.
    """
    if ticker.upper() == "GOOG":
        return f"180 {currency}"
    return f"Data not found for {ticker}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the OpenAI function declaration example with an optional question."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="The question to send to OpenAI. Defaults to a stock price prompt.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model to use (overrides OPENAI_MODEL env var).",
    )
    return parser.parse_args()


SYSTEM_INSTRUCTION = """
    You are a helpful assistant and curteous.
    If you don't know the answer, say you don't know. You can call functions to get information you need to answer the question."
"""


def main(question: str | None = None, model: str | None = None) -> None:
    """Run the OpenAI example with the provided question."""
    client = get_client()

    # Auto-derive tool schemas from Python functions — analogous to
    # google.genai's types.FunctionDeclaration.from_callable.
    # Doubleword's Responses endpoint requires the nested Chat-Completions
    # tool shape, so wrap each flat schema with a "function" key.
    flat_tools = [
        create_tool_from_function(get_weather),
        create_tool_from_function(get_stock_price),
        create_tool_from_function(get_location),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in flat_tools
    ]

    prompt = question or "Can you check the stock price for GOOG in EUR?"
    print(f"Prompt sent to model: {prompt}")

    dw_model = model or os.getenv("DOUBLEWORD_MODEL")
    if not dw_model:
        raise ValueError("DOUBLEWORD_MODEL environment variable is required")
    else:
        print(f"Using model: {dw_model}")

    response = client.responses.create(
        model=dw_model,
        instructions=SYSTEM_INSTRUCTION,
        input=prompt,
        tools=tools,
    )

    function_calls = [item for item in response.output if item.type == "function_call"]

    if function_calls:
        for call in function_calls:
            args = json.loads(call.arguments)
            print(f"Model wants to call function: '{call.name}'")
            print(f"Arguments provided by model: {args}")
            if call.name == "get_stock_price":
                result = get_stock_price(**args)
                print(f"Function execution result: {result}")
            elif call.name == "get_location":
                result = get_location(**args)
                print(f"Function execution result: {result}")
            elif call.name == "get_weather":
                result = get_weather(**args)
                print(f"Function execution result: {result}")
    else:
        print("Model did not call any functions:\n" + response.output_text)


BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"


if __name__ == "__main__":
    bar = "=" * 70
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{BOLD}{CYAN} Running OpenAI (DoubleWord) function declaration example...{RESET}")
    print(f"{CYAN}{bar}{RESET}")
    load_dotenv()
    args = parse_args()
    main(args.question, args.model)
