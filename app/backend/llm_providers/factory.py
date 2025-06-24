import os
from .base import LLMProvider, OpenAIConfig, GeminiConfig, LocalHFConfig
from .openai_provider import OpenAILLMProvider
from .gemini_provider import GeminiLLMProvider
from .local_hf_provider import LocalHFLMMProvider

# Helper to load from environment, could be expanded
def _get_env_var(var_name: str, default: str | None = None) -> str | None:
    return os.environ.get(var_name, default)

class LLMFactory:
    """
    Factory class to create instances of LLM Providers.
    It centralizes the logic for instantiating providers based on configuration.
    """

    @staticmethod
    def get_llm_provider(provider_name: str) -> LLMProvider:
        """
        Gets an LLM provider instance based on its name and loads configuration
        from environment variables.

        Args:
            provider_name (str): The name of the provider (e.g., "openai", "gemini", "local_hf_ollama").

        Returns:
            LLMProvider: An instance of the requested LLM provider.

        Raises:
            ValueError: If the provider name is unknown or required configuration is missing.
        """
        if provider_name.lower() == "openai":
            api_key = _get_env_var("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI provider.")

            # Check for Azure specific vars to determine if it's an Azure OpenAI setup
            azure_api_base = _get_env_var("AZURE_OPENAI_ENDPOINT") # Matches .env.example
            azure_deployment_name = _get_env_var("AZURE_DEPLOYMENT_NAME")
            azure_api_version = _get_env_var("AZURE_OPENAI_API_VERSION")

            if azure_api_base and azure_deployment_name:
                # If Azure specific vars are present, assume Azure OpenAI
                # Note: LiteLLM also uses AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
                # We use OPENAI_API_KEY as the source for AZURE_API_KEY for simplicity here.
                config = OpenAIConfig(
                    api_key=api_key, # This will be used as AZURE_API_KEY by the provider
                    api_base=azure_api_base,
                    deployment_name=azure_deployment_name,
                    api_version=azure_api_version or "2023-07-01-preview" # Default if not set
                )
                return OpenAILLMProvider(config)
            else:
                # Standard OpenAI or other OpenAI-compatible (non-Azure)
                api_base = _get_env_var("OPENAI_API_BASE") # For non-Azure custom OpenAI endpoints
                config = OpenAIConfig(api_key=api_key, api_base=api_base)
                return OpenAILLMProvider(config)

        elif provider_name.lower() == "gemini":
            api_key = _get_env_var("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set for Gemini provider.")
            config = GeminiConfig(api_key=api_key)
            return GeminiLLMProvider(config)

        elif provider_name.lower() == "local_hf_ollama":
            # Example for a specific type of local_hf provider: Ollama
            # These env vars are suggestions from .env.example
            model_prefix = _get_env_var("LOCAL_HF_OLLAMA_MODEL_PREFIX", "ollama/") # Default for prefix is fine
            api_base = _get_env_var("LOCAL_HF_OLLAMA_API_BASE") # Made strictly required by removing default here

            if not api_base: # This check will now work as intended if env var is not set
                raise ValueError("LOCAL_HF_OLLAMA_API_BASE environment variable not set for local_hf_ollama provider.")

            # api_key for Ollama via LiteLLM is usually not needed or is arbitrary.
            api_key = _get_env_var("LOCAL_HF_OLLAMA_API_KEY") # Optional

            config = LocalHFConfig(model_name_prefix=model_prefix, api_base=api_base, api_key=api_key)
            return LocalHFLMMProvider(config)

        elif provider_name.lower() == "local_hf_generic":
            # Example for a generic OpenAI-compatible local server
            model_prefix = _get_env_var("LOCAL_HF_GENERIC_MODEL_PREFIX", "") # Can be empty
            api_base = _get_env_var("LOCAL_HF_GENERIC_API_BASE")
            api_key = _get_env_var("LOCAL_HF_GENERIC_API_KEY") # Could be "EMPTY" or an actual key

            if api_base is None: # model_prefix can be empty, but base is crucial
                raise ValueError("LOCAL_HF_GENERIC_API_BASE not set for local_hf_generic provider.")

            config = LocalHFConfig(model_name_prefix=model_prefix, api_base=api_base, api_key=api_key)
            return LocalHFLMMProvider(config)

        # Add more providers here as needed, e.g., "local_hf_vllm", "local_hf_tgi"
        # each potentially reading slightly different env vars for their configs.

        else:
            raise ValueError(f"Unknown LLM provider name: {provider_name}")

# Example of how to get a provider:
# try:
#     openai_provider = LLMFactory.get_llm_provider("openai")
#     # Use openai_provider
# except ValueError as e:
#     print(e)
#
# try:
#     gemini_provider = LLMFactory.get_llm_provider("gemini")
#     # Use gemini_provider
# except ValueError as e:
#     print(e)
#
# try:
#     ollama_provider = LLMFactory.get_llm_provider("local_hf_ollama")
#     # Use ollama_provider, then call generate_text with a short model name like "mistral"
# except ValueError as e:
#     print(e)
