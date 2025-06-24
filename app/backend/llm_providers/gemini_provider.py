import os
import litellm
from typing import Any, Dict
from .base import LLMProvider, GeminiConfig

# litellm.set_verbose = True # For debugging

class GeminiLLMProvider(LLMProvider):
    """
    LLMProvider implementation for Google Gemini models using LiteLLM.
    """
    def __init__(self, config: GeminiConfig):
        self.config = config
        if config.api_key:
            # LiteLLM expects GEMINI_API_KEY to be set in the environment
            os.environ["GEMINI_API_KEY"] = config.api_key
        # No specific API base for Gemini via LiteLLM in the same way as OpenAI

    def _prepare_litellm_params(
        self,
        model_name: str, # e.g., "gemini/gemini-pro", "gemini-1.5-pro-latest"
        temperature: float,
        max_tokens: int,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepares parameters for litellm.completion call."""
        params: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        # LiteLLM expects model names for Gemini like "gemini/gemini-pro" or "gemini-pro"
        # If a prefix like "gemini/" isn't already there, LiteLLM might add it or expect it.
        # For safety, we can ensure it's prefixed if it's a known Gemini model without it.
        # However, LiteLLM is generally good at figuring this out.
        # We will pass the model_name as is, assuming user provides a LiteLLM-compatible Gemini model name.
        # e.g., "gemini/gemini-pro", "gemini-1.5-flash"
        if not model_name.startswith("gemini/"):
            params["model"] = f"gemini/{model_name}"
        else:
            params["model"] = model_name

        return params

    def generate_text(
        self,
        prompt: str,
        model_name: str, # e.g., "gemini-pro", "gemini/gemini-1.5-pro-latest"
        temperature: float = 0.1,
        max_tokens: int = 150,
        **kwargs: Any
    ) -> str:
        litellm_params = self._prepare_litellm_params(model_name, temperature, max_tokens, **kwargs)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = litellm.completion(messages=messages, **litellm_params)
            content = response.choices[0].message.content
            if content is None:
                raise Exception("LLM response content is None.")
            return content.strip()
        except Exception as e:
            self._handle_llm_error(e, context=f"Gemini text generation with model {litellm_params.get('model')}")
            return "" # Unreachable

    def generate_code(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 300,
        **kwargs: Any
    ) -> str:
        return self.generate_text(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def generate_summary(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.3,
        max_tokens: int = 200,
        **kwargs: Any
    ) -> str:
        return self.generate_text(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

if __name__ == '__main__':
    print("Testing GeminiLLMProvider...")
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print(f"Loaded .env file from {dotenv_path}")
    else:
        print(f".env file not found at {dotenv_path}, relying on environment variables.")

    if os.getenv("GEMINI_API_KEY"):
        print("\n--- Testing Gemini ---")
        gem_config = GeminiConfig(api_key=os.environ["GEMINI_API_KEY"])
        gem_provider = GeminiLLMProvider(config=gem_config)
        try:
            # Ensure the model name is one LiteLLM supports for Gemini, e.g., "gemini/gemini-pro" or "gemini-1.5-flash-latest"
            # For this test, we'll use a common one.
            # Note: "gemini-pro" is often a good default for text generation.
            # "gemini-1.5-flash-latest" or "gemini/gemini-1.5-flash" might be faster/cheaper
            # model_to_test = "gemini/gemini-pro" # or "gemini-pro-vision" for multimodal
            model_to_test = "gemini-1.5-flash" # Using a generally available and fast model

            print(f"Attempting to use model: {model_to_test} (will be prefixed to gemini/{model_to_test} if not already)")
            response_text = gem_provider.generate_text(f"What is the main export of Brazil? Respond concisely.", model_name=model_to_test)
            print(f"Gemini Text Response: {response_text}")

            # Test code generation (though Gemini's strength varies by model version)
            response_code = gem_provider.generate_code(f"Write a simple Python function to greet someone.", model_name=model_to_test)
            print(f"Gemini Code Response:\n{response_code}")

        except Exception as e:
            print(f"Error during Gemini test: {e}")
            print("Please ensure your GEMINI_API_KEY is correct and the model name is valid for your API access level.")
            print("Common models: 'gemini/gemini-pro', 'gemini-1.5-flash', 'gemini-1.5-pro'.")
    else:
        print("\nSkipping Gemini test: GEMINI_API_KEY not set.")

    print("\nGeminiLLMProvider tests finished.")
