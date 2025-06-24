from abc import ABC, abstractmethod
from typing import Any, Dict

class LLMProvider(ABC):
    """
    Abstract base class for Large Language Model providers.
    """

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        model_name: str, # Specific model identifier for the provider
        temperature: float = 0.1,
        max_tokens: int = 150,
        **kwargs: Any
    ) -> str:
        """
        Generates text (e.g., code, a query) based on a given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            model_name (str): The specific model to use (e.g., "gpt-3.5-turbo", "gemini/gemini-pro").
            temperature (float): Controls randomness. Lower is more deterministic.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional provider-specific arguments.

        Returns:
            str: The LLM-generated text.

        Raises:
            Exception: If the LLM call fails.
        """
        pass

    @abstractmethod
    def generate_code(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 200, # Code might need more tokens
        **kwargs: Any
    ) -> str:
        """
        Specialized method for generating code.
        This might have different default parameters or internal handling.
        """
        pass

    @abstractmethod
    def generate_summary(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.3, # Summaries can be a bit more creative
        max_tokens: int = 200,
        **kwargs: Any
    ) -> str:
        """
        Specialized method for generating a natural language summary.
        """
        pass

    def _handle_llm_error(self, error: Exception, context: str = "") -> None:
        """
        Protected helper method to handle common LLM errors.
        Can be overridden by subclasses for specific error handling.
        """
        error_message = f"LLM API error"
        if context:
            error_message += f" during {context}"
        error_message += f": {str(error)}"
        # Log the error (e.g., print or use a proper logger)
        print(error_message) # Replace with actual logging in a production app
        raise Exception(error_message) from error

class LLMProviderConfig:
    """
    A simple class to hold configuration for an LLM provider.
    This helps in passing around common configurations like API keys, base URLs, etc.
    Subclasses of LLMProvider can expect specific attributes to be present in this config.
    """
    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs):
        self.api_key = api_key
        self.api_base = api_base
        self.custom_config: Dict[str, Any] = kwargs

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_config.get(key, default)

    def __repr__(self):
        return f"LLMProviderConfig(api_key={'*****' if self.api_key else None}, api_base={self.api_base}, custom_config={self.custom_config})"

# Example of how a specific provider might define its config needs
class OpenAIConfig(LLMProviderConfig):
    def __init__(self, api_key: str, api_base: str | None = None, deployment_name: str | None = None, api_version: str | None = None):
        super().__init__(api_key=api_key, api_base=api_base)
        self.deployment_name = deployment_name # For Azure
        self.api_version = api_version # For Azure

class GeminiConfig(LLMProviderConfig):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key)

class LocalHFConfig(LLMProviderConfig):
    def __init__(self, model_name_prefix: str, api_base: str, api_key: str | None = None):
        super().__init__(api_key=api_key, api_base=api_base)
        # model_name_prefix might be like "ollama/" or "localai/" if LiteLLM uses it
        self.model_name_prefix = model_name_prefix
        # The actual model name like "mistral" or "llama2" will be passed to generate_text/code

    def get_full_model_name(self, short_model_name: str) -> str:
        """
        Constructs the full model name that LiteLLM expects, e.g., "ollama/mistral".
        """
        if self.model_name_prefix and not short_model_name.startswith(self.model_name_prefix):
            return f"{self.model_name_prefix}{short_model_name}"
        return short_model_name
