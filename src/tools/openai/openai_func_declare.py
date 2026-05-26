import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from ..tool_functions import get_location, get_weather
from .openai import create_tool_from_function


def get_client() -> OpenAI:
    """Get a configured OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return OpenAI(api_key=api_key)


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
    tools = [
        create_tool_from_function(get_weather),
        create_tool_from_function(get_stock_price),
        create_tool_from_function(get_location),
    ]

    prompt = question or "Can you check the stock price for GOOG in EUR?"
    print(f"Prompt sent to model: {prompt}")

    openai_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    response = client.responses.create(
        model=openai_model,
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
    print(f"{BOLD}{CYAN} Running OpenAI function declaration example...{RESET}")
    print(f"{CYAN}{bar}{RESET}")
    load_dotenv()
    args = parse_args()
    main(args.question, args.model)
