import os
import argparse
import base64
from dotenv import load_dotenv

from google import genai
from google.genai import types

"""
### This is a simple example of using the Gemini API with function calls.
### It defines a function declaration for getting the current temperature for a given location,
### and then sends a request to the Gemini API with that function declaration included in the tools configuration. 
### The response is checked for a function call, and if found, the function name and arguments are printed out.
### 
### - To run this example, make sure you have the GEMINI_API_KEY and GEMINI_MODEL environment variables set in your .env file.
### - You can customize the prompt by passing a different string to the --prompt argument when running the script.
### - go into week1 directory first
### Example usage: uv run src/simple_example.py --prompt "What's the temperature in New York?"
###
"""


# Dictionary of top 10 US cities with weather data
WEATHER_DATA = {
    "New York": {"temperature": 72, "condition": "cloudy"},
    "Los Angeles": {"temperature": 85, "condition": "sunny"},
    "Chicago": {"temperature": 68, "condition": "rainy"},
    "Houston": {"temperature": 88, "condition": "sunny"},
    "Phoenix": {"temperature": 95, "condition": "sunny"},
    "Philadelphia": {"temperature": 70, "condition": "cloudy"},
    "San Antonio": {"temperature": 86, "condition": "sunny"},
    "San Diego": {"temperature": 78, "condition": "sunny"},
    "Dallas": {"temperature": 82, "condition": "windy"},
    "San Jose": {"temperature": 75, "condition": "cloudy"},
    "Seattle": {"temperature": 54, "condition": "raining"},
}

def get_current_weather(location: str) -> str:
    """Returns the current weather for a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA or san francisco, ca
    
    Returns:
        A string with temperature (in Fahrenheit) and weather condition
    """
    # Normalize location to title case for case-insensitive lookup
    normalized_location = location.title()
    
    if normalized_location in WEATHER_DATA:
        weather = WEATHER_DATA[normalized_location]
        return f"{normalized_location}: {weather['temperature']}°F, {weather['condition']}"
    else:
        return f"Weather data not available for {location}. Available cities: {', '.join(WEATHER_DATA.keys())}"

def main(prompt: str):
    print("====== Main function started ======")
    # load the environment variables from the .env file
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL")
    print(f"Using Gemini model: {gemini_model}")

    if not api_key:
        raise Exception("GEMINI_API_KEY not found in environment variables.")

    if not gemini_model:
        raise Exception("GEMINI_MODEL not found in environment variables.")

    # Configure the client and tools
    client = genai.Client(api_key=api_key)    
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True
        ),
        tools=[get_current_weather]
    )

    # Send request with function declarations
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=config,
    )

    for part in response.candidates[0].content.parts:
        # show thought process        
        if part.thought:
            print(" *** thought summary ***")        
            print(part.text)
            print(" *** end thought summary ***\n")        
        # Check for a function call
        elif part.function_call:
            print("\n *** function call ***")  
            function_call = response.candidates[0].content.parts[0].function_call
            print(f"function_call object: {function_call}")
            print(f"Function to call: {function_call.name}")
            print(f"ID: {function_call.id}")
            print(f"Arguments: {function_call.args}")
            print("")
            #  In a real app, you would call your function here:
            #  result = get_current_temperature(**function_call.args)
        else:
            print("No function call found in the response.")
            print("\nAnswer:")
            print(response.text)

if __name__ == "__main__":
    print("\n===============================================================")
    print(f" Running: {os.path.basename(__file__)}")
    print("=================================================================\n")

    parser = argparse.ArgumentParser(description="Simple example of using Gemini API with function calls.")
    parser.add_argument("--prompt", type=str, help="prompt to test", default="What's the temperature in London?")
    
    
    args = parser.parse_args()
    user_prompt = args.prompt
    print(f"Using user_prompt: {user_prompt}")
    main(user_prompt)