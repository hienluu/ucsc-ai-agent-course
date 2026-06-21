import os

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from google.adk.models.lite_llm import LiteLlm

from langsmith.integrations.google_adk import configure_google_adk

from dotenv import load_dotenv
load_dotenv()

langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower().strip() == "true"
print(f">>> Configuring LangSmith tracing: {langsmith_tracing}")
if langsmith_tracing:
    print("LangSmith tracing enabled, configuring tracing...")
    configure_google_adk(project_name="ProductResearchAgentEvaluation")

DOUBLEWORD_API_KEY = os.getenv("DOUBLEWORD_API_KEY")
DOUBLEWORD_MODEL = os.getenv("DOUBLEWORD_MODEL")
DOUBLEWORD_API_URL = os.getenv("DOUBLEWORD_API_URL")

model_provider = os.getenv("MODEL_PROVIDER")
print(f"MODEL_PROVIDER: {model_provider}")


if os.getenv("MODEL_PROVIDER").lower() == "doubleword" and not DOUBLEWORD_API_KEY:
    raise ValueError("DOUBLEWORD_API_KEY is not set in the environment variables.")

# Tool Definitions
def get_product_details(product_name: str) -> str:
    """Gathers details about a product in the catalog."""
    details = {
        "smartphone": "A cutting-edge smartphone with advanced camera features and lightning-fast processing.",
        "usb charger": "A super fast and light usb charger",
        "shoes": "High-performance running shoes designed for comfort, support, and speed.",
        "headphones": "Wireless headphones with advanced noise cancellation technology for immersive audio.",
        "speaker": "A voice-controlled smart speaker that plays music, sets alarms, and controls smart home devices.",
    }
    return details.get(product_name.lower(), "Product details not found.")


def get_product_price(product_name: str) -> str:
    """Gathers price about a product. Returns price as a string for consistency with other tool."""
    prices = {
        "smartphone": "500",
        "usb charger": "10",
        "shoes": "100",
        "headphones": "50",
        "speaker": "80",
    }
    return prices.get(product_name.lower(), None)


def lookup_product_information(product_name: str) -> str:
    """Looks up specific information for a product in the catalog."""
    backend_info = {
        "smartphone": "SKU: G-SMRT-001, Inventory: 550 units",
        "usb charger": "SKU: G-CHRG-003, Inventory: 1200 units",
        "shoes": "SKU: G-SHOE-007, Inventory: 800 units",
        "headphones": "SKU: G-HDPN-002, Inventory: 950 units",
        "speaker": "SKU: G-SPKR-001, Inventory: 400 units",
    }
    return backend_info.get(product_name.lower(), "Backend information not found.")


# Wrap functions as ADK FunctionTools
get_product_details_tool = FunctionTool(func=get_product_details)
get_product_price_tool = FunctionTool(func=get_product_price)
get_product_information_tool = FunctionTool(func=lookup_product_information)

# Original instruction
ORIGINAL_INSTRUCTION = """You are a helpful customer support agent specializing in product information.
Your goal is to answer user queries about product details or prices.

1.  Analyze the user's query to identify the product name.
2.  If the user is asking for the price, use the `get_product_price` tool.
3.  If the user is asking for information or details, use the available tools to find the answer depending on the specific aspect they are inquiring about.
"""

IMPROVED_INSTRUCTION = """You are a helpful customer support agent specializing in product information.
Your goal is to answer user queries about product details or prices.


1.  For general, customer-facing descriptions (like 'tell me about...'), ALWAYS use the `get_product_details` tool.
2.  For internal data like SKU or inventory, use the `lookup_product_information` tool.
3.  If the user is asking for the price, use the `get_product_price` tool.
"""

doubleword_llm = LiteLlm(
        model=f"{DOUBLEWORD_MODEL}",
        api_base=f"{DOUBLEWORD_API_URL}",
        api_key=f"{DOUBLEWORD_API_KEY}",
        include_reasoning=False, # Set to False to exclude reasoning steps from the response
)

# Define the Root Agent
root_agent = LlmAgent(
    name="ProductResearchAgent",
    model=doubleword_llm,
    #model="gemini-2.5-flash",    
    description="An agent that provides details and prices for various products.",
    instruction=ORIGINAL_INSTRUCTION,
    tools=[
        get_product_details_tool,
        get_product_price_tool,
        get_product_information_tool,
    ],
)