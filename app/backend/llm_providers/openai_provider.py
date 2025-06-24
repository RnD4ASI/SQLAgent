import os
import litellm
from typing import Any, Dict
from .base import LLMProvider, OpenAIConfig

# Set LiteLLM verbosity for debugging if needed
# litellm.set_verbose = True

class OpenAILLMProvider(LLMProvider):
    """
    LLMProvider implementation for OpenAI models (including Azure OpenAI) using LiteLLM.
    """
    def __init__(self, config: OpenAIConfig):
        self.config = config
        # LiteLLM uses environment variables for API keys, base, version, etc.
        # We ensure they are set here based on the config object if provided,
        # otherwise LiteLLM will try to pick them up from the environment directly.

        if config.api_key:
            os.environ["OPENAI_API_KEY"] = config.api_key # Standard OpenAI
            os.environ["AZURE_API_KEY"] = config.api_key    # For Azure

        if config.api_base:
            os.environ["OPENAI_API_BASE"] = config.api_base # Standard OpenAI custom base
            os.environ["AZURE_API_BASE"] = config.api_base  # For Azure endpoint

        if config.deployment_name: # Azure specific
            # The model string for LiteLLM's completion call for Azure should be "azure/{deployment_name}"
            # We don't set an env var for deployment_name itself, it's part of the model identifier.
            pass

        if config.api_version: # Azure specific
            os.environ["AZURE_API_VERSION"] = config.api_version


    def _prepare_litellm_params(
        self,
        model_name: str,
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

        # Handle Azure model prefix
        # model_name would be like "gpt-3.5-turbo" or "text-davinci-003" for standard OpenAI
        # or "your-deployment-name" for Azure.
        if self.config.deployment_name:
            # LiteLLM expects "azure/" prefix for Azure models if using deployment name
            # and relying on AZURE_API_BASE, AZURE_API_KEY, AZURE_API_VERSION env vars.
            params["model"] = f"azure/{self.config.deployment_name}"
        elif model_name.startswith("azure/"): # If user explicitly passes "azure/deployment-name"
            params["model"] = model_name
        else:
            # For standard OpenAI, model_name is directly used.
            # LiteLLM also allows "openai/model_name"
            params["model"] = model_name # e.g., "gpt-3.5-turbo"

        return params

    def generate_text(
        self,
        prompt: str,
        model_name: str, # e.g., "gpt-3.5-turbo" or "your-azure-deployment"
        temperature: float = 0.1,
        max_tokens: int = 150,
        **kwargs: Any
    ) -> str:
        litellm_params = self._prepare_litellm_params(model_name, temperature, max_tokens, **kwargs)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = litellm.completion(messages=messages, **litellm_params)
            # LiteLLM's response format is OpenAI-like:
            # response['choices'][0]['message']['content']
            content = response.choices[0].message.content
            if content is None:
                raise Exception("LLM response content is None.")
            return content.strip()
        except Exception as e:
            self._handle_llm_error(e, context=f"OpenAI text generation with model {litellm_params.get('model')}")
            return "" # Should be unreachable due to _handle_llm_error raising

    def generate_code(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 300, # Adjusted default for code
        **kwargs: Any
    ) -> str:
        # Could have specific logic or prompt wrapping for code generation if needed
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

# Example usage (for testing or direct use if needed):
if __name__ == '__main__':
    # This is for illustrative purposes.
    # In the actual app, the factory would create and configure this.
    print("Testing OpenAILLMProvider...")

    # Load .env file for API keys if not already set in environment
    from dotenv import load_dotenv
    # Assuming .env is in the project root, two levels up from llm_providers/
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print(f"Loaded .env file from {dotenv_path}")
    else:
        print(f".env file not found at {dotenv_path}, relying on environment variables.")


    # Standard OpenAI Example
    if os.getenv("OPENAI_API_KEY"):
        print("\n--- Testing Standard OpenAI ---")
        std_config = OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])
        std_provider = OpenAILLMProvider(config=std_config)
        try:
            # model_name should be a valid OpenAI model like "gpt-3.5-turbo-instruct" for completions or "gpt-3.5-turbo" for chat completions
            # LiteLLM handles both completion and chat models through litellm.completion
            response_text = std_provider.generate_text("What is the capital of France?", model_name="gpt-3.5-turbo")
            print(f"Text Response: {response_text}")
            response_code = std_provider.generate_code("Python function to add two numbers:", model_name="gpt-3.5-turbo")
            print(f"Code Response:\n{response_code}")
        except Exception as e:
            print(f"Error during Standard OpenAI test: {e}")
    else:
        print("\nSkipping Standard OpenAI test: OPENAI_API_KEY not set.")

    # Azure OpenAI Example
    if os.getenv("AZURE_API_KEY") and os.getenv("AZURE_API_BASE") and os.getenv("AZURE_DEPLOYMENT_NAME_GPT35") and os.getenv("AZURE_API_VERSION"):
        print("\n--- Testing Azure OpenAI ---")
        azure_config = OpenAIConfig(
            api_key=os.environ["AZURE_API_KEY"],
            api_base=os.environ["AZURE_API_BASE"],
            deployment_name=os.environ["AZURE_DEPLOYMENT_NAME_GPT35"], # Your actual deployment name for a chat model
            api_version=os.environ["AZURE_API_VERSION"]
        )
        azure_provider = OpenAILLMProvider(config=azure_config)
        try:
            # For Azure, model_name to generate_text is the deployment name.
            # The provider's _prepare_litellm_params will prefix it with "azure/" for LiteLLM.
            response_azure = azure_provider.generate_text("Translate 'hello' to Spanish.", model_name=azure_config.deployment_name) # Pass deployment name
            print(f"Azure Text Response: {response_azure}")
        except Exception as e:
            print(f"Error during Azure OpenAI test: {e}")
    else:
        print("\nSkipping Azure OpenAI test: One or more Azure environment variables missing (AZURE_API_KEY, AZURE_API_BASE, AZURE_DEPLOYMENT_NAME_GPT35, AZURE_API_VERSION).")
        print("Ensure AZURE_DEPLOYMENT_NAME_GPT35 points to a chat-completion compatible model deployment (e.g., gpt-3.5-turbo).")

    # Example for custom OpenAI-compatible endpoint (e.g. self-hosted)
    # Requires OPENAI_API_KEY (can be dummy if server doesn't need it), OPENAI_API_BASE, and a model name the server expects.
    # For this example, let's assume a local server like Ollama serving an OpenAI-compatible endpoint.
    # Note: This is slightly different from the LocalHFLMMProvider which might use LiteLLM's direct Ollama integration.
    # This test is for an OpenAI *API compatible* endpoint.
    if os.getenv("CUSTOM_OPENAI_API_BASE") and os.getenv("CUSTOM_OPENAI_MODEL_NAME"):
        print("\n--- Testing Custom OpenAI-compatible Endpoint ---")
        # CUSTOM_OPENAI_API_KEY might be optional or a dummy value depending on the server
        custom_api_key = os.getenv("CUSTOM_OPENAI_API_KEY", "dummy-key")
        custom_config = OpenAIConfig(
            api_key=custom_api_key,
            api_base=os.environ["CUSTOM_OPENAI_API_BASE"]
        )
        custom_provider = OpenAILLMProvider(config=custom_config)
        try:
            # The model_name here must be what the custom server expects.
            response_custom = custom_provider.generate_text(
                "Tell me a short joke.",
                model_name=os.environ["CUSTOM_OPENAI_MODEL_NAME"]
            )
            print(f"Custom Endpoint Text Response: {response_custom}")
        except Exception as e:
            print(f"Error during Custom OpenAI-compatible endpoint test: {e}")
    else:
        print("\nSkipping Custom OpenAI-compatible endpoint test: CUSTOM_OPENAI_API_BASE or CUSTOM_OPENAI_MODEL_NAME not set.")

    print("\nOpenAILLMProvider tests finished.")
