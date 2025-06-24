import os
import litellm
from typing import Any, Dict
from .base import LLMProvider, LocalHFConfig

# litellm.set_verbose = True # For debugging

class LocalHFLMMProvider(LLMProvider):
    """
    LLMProvider implementation for Local HuggingFace models accessed via an
    OpenAI-compatible API endpoint (e.g., Ollama, TGI, vLLM) using LiteLLM.
    """
    def __init__(self, config: LocalHFConfig):
        self.config = config
        # LiteLLM can take api_base and api_key directly in the completion call
        # for custom models, or it can use environment variables.
        # For local models, api_key is often not required or can be a dummy string.

        # If an API key is provided in config, set it for LiteLLM if it expects it
        # for this custom provider. Often for local models, this is not needed.
        if config.api_key:
            os.environ["CUSTOM_API_KEY"] = config.api_key # Generic for LiteLLM's custom provider

        # The api_base is crucial and will be passed directly to LiteLLM.
        # The model_name will be constructed using config.get_full_model_name().

    def _prepare_litellm_params(
        self,
        short_model_name: str, # e.g., "mistral", "llama2" (without prefix)
        temperature: float,
        max_tokens: int,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepares parameters for litellm.completion call."""

        full_model_name = self.config.get_full_model_name(short_model_name)

        params: Dict[str, Any] = {
            "model": full_model_name, # e.g., "ollama/mistral" or "custom/my-local-model"
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Crucially, pass the api_base for the custom local model
            "api_base": self.config.api_base,
            **kwargs
        }
        # If an API key was specifically set for this provider and is needed
        if self.config.api_key:
            params["api_key"] = self.config.api_key

        return params

    def generate_text(
        self,
        prompt: str,
        model_name: str, # This is the short model name, e.g., "mistral"
        temperature: float = 0.1,
        max_tokens: int = 150,
        **kwargs: Any
    ) -> str:
        litellm_params = self._prepare_litellm_params(model_name, temperature, max_tokens, **kwargs)
        messages = [{"role": "user", "content": prompt}]

        try:
            # For some local models/servers, LiteLLM might need a specific prefix like "ollama/"
            # The full_model_name from config.get_full_model_name() should handle this.
            response = litellm.completion(messages=messages, **litellm_params)
            content = response.choices[0].message.content
            if content is None:
                raise Exception("LLM response content is None.")
            return content.strip()
        except Exception as e:
            self._handle_llm_error(e, context=f"LocalHF text generation with model {litellm_params.get('model')} at {self.config.api_base}")
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
    print("Testing LocalHFLMMProvider...")
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print(f"Loaded .env file from {dotenv_path}")
    else:
        print(f".env file not found at {dotenv_path}, relying on environment variables.")

    # Example for Ollama (ensure Ollama server is running and 'mistral' model is pulled)
    # Define these in your .env or set them directly for testing
    # LOCAL_HF_MODEL_NAME_PREFIX="ollama/"
    # LOCAL_HF_API_BASE="http://localhost:11434"
    # LOCAL_HF_SHORT_MODEL_NAME="mistral" # or whatever model you have on Ollama

    ollama_prefix = os.getenv("LOCAL_HF_OLLAMA_MODEL_PREFIX", "ollama/") # e.g. "ollama/"
    ollama_base_url = os.getenv("LOCAL_HF_OLLAMA_API_BASE", "http://localhost:11434")
    ollama_model = os.getenv("LOCAL_HF_OLLAMA_SHORT_MODEL_NAME", "mistral") # The model name Ollama uses

    if ollama_base_url and ollama_model:
        print(f"\n--- Testing Local HuggingFace (Ollama: {ollama_prefix}{ollama_model} via {ollama_base_url}) ---")
        # API key for local Ollama is typically not needed or can be arbitrary if LiteLLM requires one.
        # LiteLLM's ollama integration usually doesn't need an explicit key.
        local_hf_config = LocalHFConfig(
            model_name_prefix=ollama_prefix, # LiteLLM uses this to route to "ollama/mistral"
            api_base=ollama_base_url,
            api_key="dummy_key_if_needed" # Or None
        )
        local_hf_provider = LocalHFLMMProvider(config=local_hf_config)
        try:
            # model_name is the short name, e.g., "mistral"
            response_text = local_hf_provider.generate_text(
                f"Why is the sky blue? Respond concisely.",
                model_name=ollama_model # Pass the short name like "mistral"
            )
            print(f"LocalHF (Ollama) Text Response: {response_text}")

            response_code = local_hf_provider.generate_code(
                f"Write a shell script to list files in a directory.",
                model_name=ollama_model
            )
            print(f"LocalHF (Ollama) Code Response:\n{response_code}")

        except Exception as e:
            print(f"Error during LocalHF (Ollama) test: {e}")
            print(f"Ensure Ollama is running at {ollama_base_url} and model '{ollama_model}' is pulled (e.g., 'ollama pull {ollama_model}').")
            print(f"LiteLLM will try to call: {local_hf_config.get_full_model_name(ollama_model)}")
    else:
        print("\nSkipping Local HuggingFace (Ollama) test: LOCAL_HF_OLLAMA_API_BASE or LOCAL_HF_OLLAMA_SHORT_MODEL_NAME not set in environment.")

    # Example for a generic OpenAI-compatible local server
    # LOCAL_HF_GENERIC_MODEL_PREFIX = "custom/" (or "" if the server doesn't need it in model name string)
    # LOCAL_HF_GENERIC_API_BASE = "http://localhost:8000/v1"
    # LOCAL_HF_GENERIC_SHORT_MODEL_NAME = "my-local-quantized-model"
    # LOCAL_HF_GENERIC_API_KEY = "EMPTY" (or actual key if server needs one)

    generic_prefix = os.getenv("LOCAL_HF_GENERIC_MODEL_PREFIX")
    generic_base_url = os.getenv("LOCAL_HF_GENERIC_API_BASE")
    generic_model = os.getenv("LOCAL_HF_GENERIC_SHORT_MODEL_NAME")
    generic_api_key = os.getenv("LOCAL_HF_GENERIC_API_KEY")

    if generic_base_url and generic_model and generic_prefix is not None: # Prefix can be empty string
        print(f"\n--- Testing Local HuggingFace (Generic Server: {generic_prefix or ''}{generic_model} via {generic_base_url}) ---")
        generic_config = LocalHFConfig(
            model_name_prefix=generic_prefix,
            api_base=generic_base_url,
            api_key=generic_api_key
        )
        generic_provider = LocalHFLMMProvider(config=generic_config)
        try:
            response_text_generic = generic_provider.generate_text(
                "What are the primary colors?",
                model_name=generic_model
            )
            print(f"LocalHF (Generic) Text Response: {response_text_generic}")
        except Exception as e:
            print(f"Error during LocalHF (Generic) test: {e}")
            print(f"Ensure your generic model server is running at {generic_base_url} and configured for model '{generic_model}'.")
            print(f"LiteLLM will try to call: {generic_config.get_full_model_name(generic_model)}")
    else:
        print("\nSkipping Local HuggingFace (Generic Server) test: One or more of LOCAL_HF_GENERIC_MODEL_PREFIX, LOCAL_HF_GENERIC_API_BASE, LOCAL_HF_GENERIC_SHORT_MODEL_NAME not set.")

    print("\nLocalHFLMMProvider tests finished.")
