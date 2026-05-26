import argparse
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from ..tool_functions import get_location, get_current_time, get_weather, calculate_sum


def get_client() -> genai.Client:
    """Get a configured Google GenAI client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


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
        description="Run the Gemini function declaration example with an optional question."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="The question to send to Gemini. Defaults to a stock price prompt.",
    )
    return parser.parse_args()

SYSTEM_INSTRUCTION = """
    You are a helpful assistant and curteous. 
    If you don't know the answer, say you don't know. You can call functions to get information you need to answer the question."
"""

def main(question: str | None = None) -> None:
    """Run the Gemini example with the provided question."""
    client = get_client()

    weather_func = types.FunctionDeclaration.from_callable(
        client=client,
        callable=get_weather,
    )

    stock_price_func = types.FunctionDeclaration.from_callable(
        client=client,
        callable=get_stock_price,
    )

    get_location_func = types.FunctionDeclaration.from_callable(
        client=client,
        callable=get_location,
    )

    tools = [types.Tool(function_declarations=[stock_price_func, weather_func, get_location_func])]

    prompt = question or "Can you check the stock price for GOOG in EUR?"
    print(f"Prompt sent to model: {prompt}")

    client_config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION,
                                                tools=tools)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=client_config,
    )

    if response.function_calls:
        for call in response.function_calls:
            print(f"Model wants to call function: '{call.name}'")
            print(f"Arguments provided by model: {call.args}")
            if call.name == "get_stock_price":
                result = get_stock_price(**call.args)
                print(f"Function execution result: {result}")
            elif call.name == "get_location":
                result = get_location(**call.args)
                print(f"Function execution result: {result}")
    else:
        print("Model did not call any functions." + response.text)
            


BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"


if __name__ == "__main__":
    bar = "=" * 70
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{BOLD}{CYAN} Running Gemini function declaration example...{RESET}")
    print(f"{CYAN}{bar}{RESET}")
    load_dotenv()
    args = parse_args()
    main(args.question)
