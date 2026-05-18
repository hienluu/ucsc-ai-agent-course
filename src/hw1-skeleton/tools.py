"""
Tool functions for the HW1 agent.

Each tool is a regular Python function with type hints and a docstring.
The google-genai SDK will inspect these to generate the tool schema that
the model sees when deciding which tool to call.

TODO (students): implement the two tool functions below.
"""


def get_weather(city: str) -> str:
    """Get the current weather report for a given city.

    Args:
        city (str): The name of the city to look up weather for.

    Returns:
        str: A short, human-readable description of the weather.
    """
    # TODO (students):
    # For HW1 you can return a hard-coded weather string per city
    # (e.g. use a small dict like {"New York": "Sunny, 72°F", ...}).
    # Stretch goal: call a real weather API instead of returning a stub.
    raise NotImplementedError("Implement get_weather")


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the sum.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of a and b.
    """
    # TODO (students): return the sum of a and b.
    raise NotImplementedError("Implement add_numbers")


# Registry of tools the agent is allowed to call.
# The google-genai SDK can take Python callables directly in the
# `tools=[...]` field of GenerateContentConfig — keep this list in sync
# with whatever the agent should expose.
AVAILABLE_TOOLS = [
    get_weather,
    add_numbers,
]
