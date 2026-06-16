import os

from dotenv import load_dotenv


load_dotenv()

def get_gemini_model():
    """Return the Gemini model name from the GEMINI_MODEL env var, or a default."""
    return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

def get_model():
    """Return a model spec suitable for LlmAgent(model=...).

    Switch via the MODEL_PROVIDER env var:
      - "gemini" (default): returns the Gemini model name from GEMINI_MODEL
                            (default: "gemini-3.1-flash-lite")
      - "doubleword":       returns a LiteLlm instance backed by Doubleword,
                            using DOUBLEWORD_API_KEY / DOUBLEWORD_MODEL /
                            DOUBLEWORD_API_URL.
    """
    provider = os.getenv("MODEL_PROVIDER", "gemini").lower()

    if provider == "doubleword":
        from google.adk.models.lite_llm import LiteLlm

        api_key = os.getenv("DOUBLEWORD_API_KEY")
        model = os.getenv("DOUBLEWORD_MODEL")
        api_url = os.getenv("DOUBLEWORD_API_URL")
        if api_key is None:
            raise ValueError("DOUBLEWORD_API_KEY is not set in the environment.")
        return LiteLlm(
            model=model,
            api_base=api_url,
            api_key=api_key,
            include_reasoning=False,
        )

    return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
