import pytest
from abc import ABC, abstractmethod
from app.backend.llm_providers.base import LLMProvider, LLMProviderConfig, OpenAIConfig, GeminiConfig, LocalHFConfig

# Test that LLMProvider is an Abstract Base Class
def test_llm_provider_is_abc():
    assert issubclass(LLMProvider, ABC)

# Test that all core methods are abstract
def test_llm_provider_abstract_methods():
    expected_abstract_methods = {"generate_text", "generate_code", "generate_summary"}
    actual_abstract_methods = LLMProvider.__abstractmethods__
    assert actual_abstract_methods == expected_abstract_methods

# Test concrete subclass for completeness (optional, but good practice)
class DummyLLMProvider(LLMProvider):
    def generate_text(self, prompt: str, model_name: str, temperature: float = 0.1, max_tokens: int = 150, **kwargs) -> str:
        return "dummy text"

    def generate_code(self, prompt: str, model_name: str, temperature: float = 0.1, max_tokens: int = 200, **kwargs) -> str:
        return "dummy code"

    def generate_summary(self, prompt: str, model_name: str, temperature: float = 0.3, max_tokens: int = 200, **kwargs) -> str:
        return "dummy summary"

def test_dummy_llm_provider_instantiation():
    provider = DummyLLMProvider()
    assert provider.generate_text("prompt", "model") == "dummy text"
    assert provider.generate_code("prompt", "model") == "dummy code"
    assert provider.generate_summary("prompt", "model") == "dummy summary"

# --- Test Config Classes ---
def test_llm_provider_config_basic():
    config = LLMProviderConfig(api_key="test_key", api_base="http://localhost", custom_arg="value")
    assert config.api_key == "test_key"
    assert config.api_base == "http://localhost"
    assert config.custom_config["custom_arg"] == "value"
    assert config.get("api_key") == "test_key"
    assert config.get("custom_arg") == "value"
    assert config.get("non_existent_arg", "default") == "default"
    assert "test_key" not in repr(config) # Check API key is masked
    assert "*****" in repr(config)

def test_openai_config():
    config = OpenAIConfig(api_key="oa_key", api_base="oa_base", deployment_name="oa_deploy", api_version="oa_version")
    assert config.api_key == "oa_key"
    assert config.api_base == "oa_base"
    assert config.deployment_name == "oa_deploy"
    assert config.api_version == "oa_version"
    assert config.get("deployment_name") == "oa_deploy"

def test_gemini_config():
    config = GeminiConfig(api_key="gem_key")
    assert config.api_key == "gem_key"
    assert config.get("api_key") == "gem_key"

def test_local_hf_config():
    config = LocalHFConfig(model_name_prefix="ollama/", api_base="http://localhf:11434", api_key="hf_key_optional")
    assert config.model_name_prefix == "ollama/"
    assert config.api_base == "http://localhf:11434"
    assert config.api_key == "hf_key_optional"
    assert config.get_full_model_name("mistral") == "ollama/mistral"
    assert config.get_full_model_name("ollama/mistral") == "ollama/mistral" # Should not double prefix

    config_no_prefix = LocalHFConfig(model_name_prefix="", api_base="http://localhf:11434")
    assert config_no_prefix.get_full_model_name("mymodel") == "mymodel"

from unittest.mock import patch, MagicMock
import os
from app.backend.llm_providers.openai_provider import OpenAILLMProvider


# --- Tests for OpenAILLMProvider ---

@pytest.fixture
def openai_std_config():
    return OpenAIConfig(api_key="test_openai_key")

@pytest.fixture
def openai_azure_config():
    return OpenAIConfig(
        api_key="test_azure_key",
        api_base="https://test-azure.openai.com",
        deployment_name="test-deployment",
        api_version="2023-07-01-preview"
    )

@patch('litellm.completion')
def test_openai_provider_generate_text_standard(mock_litellm_completion, openai_std_config):
    provider = OpenAILLMProvider(config=openai_std_config)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Standard OpenAI response"
    mock_litellm_completion.return_value = mock_response

    prompt = "Hello"
    model_name = "gpt-3.5-turbo"
    response = provider.generate_text(prompt, model_name)

    mock_litellm_completion.assert_called_once_with(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )
    assert response == "Standard OpenAI response"
    # Check if OS environ was set (optional, as LiteLLM can also take direct params)
    assert os.environ["OPENAI_API_KEY"] == "test_openai_key"


@patch('litellm.completion')
def test_openai_provider_generate_text_azure(mock_litellm_completion, openai_azure_config):
    provider = OpenAILLMProvider(config=openai_azure_config)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Azure OpenAI response"
    mock_litellm_completion.return_value = mock_response

    prompt = "Hello Azure"
    # For Azure, the model_name passed to generate_text is the deployment name
    # The provider should prepend "azure/" for LiteLLM
    response = provider.generate_text(prompt, model_name=openai_azure_config.deployment_name)

    mock_litellm_completion.assert_called_once_with(
        model=f"azure/{openai_azure_config.deployment_name}",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )
    assert response == "Azure OpenAI response"
    assert os.environ["AZURE_API_KEY"] == "test_azure_key"
    assert os.environ["AZURE_API_BASE"] == "https://test-azure.openai.com"
    assert os.environ["AZURE_API_VERSION"] == "2023-07-01-preview"


@patch('litellm.completion')
def test_openai_provider_generate_code(mock_litellm_completion, openai_std_config):
    provider = OpenAILLMProvider(config=openai_std_config)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "def hello(): pass"
    mock_litellm_completion.return_value = mock_response

    prompt = "Create a Python function"
    model_name = "gpt-3.5-turbo"
    response = provider.generate_code(prompt, model_name, temperature=0.05, max_tokens=250)

    mock_litellm_completion.assert_called_once_with(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.05,
        max_tokens=250
    )
    assert response == "def hello(): pass"

@patch('litellm.completion')
def test_openai_provider_generate_summary(mock_litellm_completion, openai_std_config):
    provider = OpenAILLMProvider(config=openai_std_config)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a summary."
    mock_litellm_completion.return_value = mock_response

    prompt = "Summarize this text"
    model_name = "gpt-3.5-turbo"
    response = provider.generate_summary(prompt, model_name, temperature=0.2, max_tokens=180)

    mock_litellm_completion.assert_called_once_with(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=180
    )
    assert response == "This is a summary."


@patch('litellm.completion')
def test_openai_provider_llm_error_handling(mock_litellm_completion, openai_std_config):
    provider = OpenAILLMProvider(config=openai_std_config)
    mock_litellm_completion.side_effect = Exception("LiteLLM API Error")

    with pytest.raises(Exception) as excinfo:
        provider.generate_text("test prompt", "gpt-3.5-turbo")

    assert "LLM API error during OpenAI text generation" in str(excinfo.value)
    assert "LiteLLM API Error" in str(excinfo.value)

@patch('litellm.completion')
def test_openai_provider_none_content_handling(mock_litellm_completion, openai_std_config):
    provider = OpenAILLMProvider(config=openai_std_config)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None # Simulate LLM returning None
    mock_litellm_completion.return_value = mock_response

    with pytest.raises(Exception) as excinfo:
        provider.generate_text("test prompt", "gpt-3.5-turbo")

    assert "LLM response content is None" in str(excinfo.value)


# --- Tests for GeminiLLMProvider ---
from app.backend.llm_providers.gemini_provider import GeminiLLMProvider

@pytest.fixture
def gemini_config_fixture(): # Renamed to avoid conflict with class name
    return GeminiConfig(api_key="test_gemini_key")

@patch('litellm.completion')
def test_gemini_provider_generate_text(mock_litellm_completion, gemini_config_fixture):
    provider = GeminiLLMProvider(config=gemini_config_fixture)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Gemini response"
    mock_litellm_completion.return_value = mock_response

    prompt = "Hello Gemini"
    model_name = "gemini-pro" # User might pass this
    expected_model_in_litellm = "gemini/gemini-pro" # Provider should ensure this format
    response = provider.generate_text(prompt, model_name)

    mock_litellm_completion.assert_called_once_with(
        model=expected_model_in_litellm,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )
    assert response == "Gemini response"
    assert os.environ["GEMINI_API_KEY"] == "test_gemini_key"

@patch('litellm.completion')
def test_gemini_provider_generate_text_with_prefix(mock_litellm_completion, gemini_config_fixture):
    provider = GeminiLLMProvider(config=gemini_config_fixture)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Gemini response"
    mock_litellm_completion.return_value = mock_response

    prompt = "Hello Gemini"
    model_name = "gemini/gemini-1.5-pro-latest" # User passes with prefix
    response = provider.generate_text(prompt, model_name)

    mock_litellm_completion.assert_called_once_with(
        model=model_name, # Should remain as is
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )
    assert response == "Gemini response"


@patch('litellm.completion')
def test_gemini_provider_llm_error_handling(mock_litellm_completion, gemini_config_fixture):
    provider = GeminiLLMProvider(config=gemini_config_fixture)
    mock_litellm_completion.side_effect = Exception("LiteLLM Gemini API Error")

    with pytest.raises(Exception) as excinfo:
        provider.generate_text("test prompt", "gemini-pro")

    assert "LLM API error during Gemini text generation" in str(excinfo.value)
    assert "LiteLLM Gemini API Error" in str(excinfo.value)


# --- Tests for LocalHFLMMProvider ---
from app.backend.llm_providers.local_hf_provider import LocalHFLMMProvider

@pytest.fixture
def local_hf_ollama_config():
    return LocalHFConfig(
        model_name_prefix="ollama/",
        api_base="http://localhost:11434",
        api_key=None # Ollama typically doesn't need an API key via LiteLLM
    )

@pytest.fixture
def local_hf_custom_config():
    return LocalHFConfig(
        model_name_prefix="custom/", # Or "" if not using a prefix for LiteLLM model string
        api_base="http://localhost:8000/v1",
        api_key="custom_api_key_if_needed"
    )

@patch('litellm.completion')
def test_local_hf_provider_ollama_style(mock_litellm_completion, local_hf_ollama_config):
    provider = LocalHFLMMProvider(config=local_hf_ollama_config)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Ollama response"
    mock_litellm_completion.return_value = mock_response

    prompt = "Hello Ollama"
    short_model_name = "mistral"
    expected_full_model_name = "ollama/mistral"

    response = provider.generate_text(prompt, short_model_name)

    mock_litellm_completion.assert_called_once_with(
        model=expected_full_model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150,
        api_base=local_hf_ollama_config.api_base
        # api_key should not be passed if None in config
    )
    assert response == "Ollama response"
    # No specific env var is set by this provider typically, relies on direct param passing to LiteLLM

@patch('litellm.completion')
def test_local_hf_provider_custom_server_style(mock_litellm_completion, local_hf_custom_config):
    provider = LocalHFLMMProvider(config=local_hf_custom_config)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Custom server response"
    mock_litellm_completion.return_value = mock_response

    prompt = "Hello Custom Server"
    short_model_name = "my-model"
    expected_full_model_name = "custom/my-model"

    response = provider.generate_text(prompt, short_model_name)

    mock_litellm_completion.assert_called_once_with(
        model=expected_full_model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150,
        api_base=local_hf_custom_config.api_base,
        api_key=local_hf_custom_config.api_key
    )
    assert response == "Custom server response"


@patch('litellm.completion')
def test_local_hf_provider_llm_error_handling(mock_litellm_completion, local_hf_ollama_config):
    provider = LocalHFLMMProvider(config=local_hf_ollama_config)
    mock_litellm_completion.side_effect = Exception("LiteLLM Local API Error")

    with pytest.raises(Exception) as excinfo:
        provider.generate_text("test prompt", "mistral")

    assert "LLM API error during LocalHF text generation" in str(excinfo.value)
    assert "LiteLLM Local API Error" in str(excinfo.value)
    assert "ollama/mistral" in str(excinfo.value) # Check model name in error
    assert local_hf_ollama_config.api_base in str(excinfo.value) # Check api_base in error


# --- Tests for LLMFactory ---
from app.backend.llm_providers.factory import LLMFactory

@patch.dict(os.environ, {
    "OPENAI_API_KEY": "factory_openai_key",
    "AZURE_OPENAI_ENDPOINT": "", # Ensure Azure var is not set or empty
    "AZURE_DEPLOYMENT_NAME": "", # Ensure Azure var is not set or empty
    "AZURE_OPENAI_API_VERSION": "" # Ensure Azure var is not set or empty
}, clear=True) # clear=True ensures only these are set for the test
def test_llm_factory_get_openai_provider_standard():
    # Also explicitly remove from os.environ if @patch.dict doesn't fully isolate during import/first call
    # This is a bit belt-and-suspenders but can help with tricky env var test pollution.
    if "AZURE_OPENAI_ENDPOINT" in os.environ: del os.environ["AZURE_OPENAI_ENDPOINT"]
    if "AZURE_DEPLOYMENT_NAME" in os.environ: del os.environ["AZURE_DEPLOYMENT_NAME"]
    if "AZURE_OPENAI_API_VERSION" in os.environ: del os.environ["AZURE_OPENAI_API_VERSION"]
    # Crucially, also ensure OPENAI_API_BASE is not set from a previous test if it could interfere
    if "OPENAI_API_BASE" in os.environ: del os.environ["OPENAI_API_BASE"]
    os.environ["OPENAI_API_KEY"] = "factory_openai_key" # Set the one we need

    provider = LLMFactory.get_llm_provider("openai")
    assert isinstance(provider, OpenAILLMProvider)
    assert provider.config.api_key == "factory_openai_key"
    assert provider.config.api_base is None, f"Expected api_base to be None, but got {provider.config.api_base}"
    assert provider.config.deployment_name is None, f"Expected deployment_name to be None, but got {provider.config.deployment_name}"

@patch.dict(os.environ, {
    "OPENAI_API_KEY": "factory_azure_key",
    "AZURE_OPENAI_ENDPOINT": "https://factory.azure.com",
    "AZURE_DEPLOYMENT_NAME": "factory_deployment",
    "AZURE_OPENAI_API_VERSION": "2024-02-15-preview"
})
def test_llm_factory_get_openai_provider_azure():
    provider = LLMFactory.get_llm_provider("openai") # Factory should detect Azure from env vars
    assert isinstance(provider, OpenAILLMProvider)
    assert provider.config.api_key == "factory_azure_key"
    assert provider.config.api_base == "https://factory.azure.com"
    assert provider.config.deployment_name == "factory_deployment"
    assert provider.config.api_version == "2024-02-15-preview"

@patch.dict(os.environ, {"GEMINI_API_KEY": "factory_gemini_key"})
def test_llm_factory_get_gemini_provider():
    provider = LLMFactory.get_llm_provider("gemini")
    assert isinstance(provider, GeminiLLMProvider)
    assert provider.config.api_key == "factory_gemini_key"

@patch.dict(os.environ, {
    "LOCAL_HF_OLLAMA_MODEL_PREFIX": "ollama/",
    "LOCAL_HF_OLLAMA_API_BASE": "http://factory-ollama:11434",
    "LOCAL_HF_OLLAMA_API_KEY": "ollama_key_optional" # Test with optional key
})
def test_llm_factory_get_local_hf_ollama_provider():
    provider = LLMFactory.get_llm_provider("local_hf_ollama")
    assert isinstance(provider, LocalHFLMMProvider)
    assert provider.config.model_name_prefix == "ollama/"
    assert provider.config.api_base == "http://factory-ollama:11434"
    assert provider.config.api_key == "ollama_key_optional"

@patch.dict(os.environ, {
    "LOCAL_HF_GENERIC_MODEL_PREFIX": "custom/",
    "LOCAL_HF_GENERIC_API_BASE": "http://factory-generic:8000/v1",
    "LOCAL_HF_GENERIC_API_KEY": "generic_key"
})
def test_llm_factory_get_local_hf_generic_provider():
    provider = LLMFactory.get_llm_provider("local_hf_generic")
    assert isinstance(provider, LocalHFLMMProvider)
    assert provider.config.model_name_prefix == "custom/"
    assert provider.config.api_base == "http://factory-generic:8000/v1"
    assert provider.config.api_key == "generic_key"


def test_llm_factory_unknown_provider():
    with pytest.raises(ValueError) as excinfo:
        LLMFactory.get_llm_provider("unknown_provider")
    assert "Unknown LLM provider name: unknown_provider" in str(excinfo.value)

@patch.dict(os.environ, {}, clear=True) # Ensure no relevant env vars are set
def test_llm_factory_missing_env_vars_openai():
    with pytest.raises(ValueError) as excinfo:
        LLMFactory.get_llm_provider("openai")
    assert "OPENAI_API_KEY environment variable not set" in str(excinfo.value)

@patch.dict(os.environ, {}, clear=True)
def test_llm_factory_missing_env_vars_gemini():
    with pytest.raises(ValueError) as excinfo:
        LLMFactory.get_llm_provider("gemini")
    assert "GEMINI_API_KEY environment variable not set" in str(excinfo.value)

@patch.dict(os.environ, {"LOCAL_HF_OLLAMA_MODEL_PREFIX": "ollama/"}, clear=True) # Missing API base
def test_llm_factory_missing_env_vars_local_hf_ollama():
    with pytest.raises(ValueError) as excinfo:
        LLMFactory.get_llm_provider("local_hf_ollama")
    assert "LOCAL_HF_OLLAMA_API_BASE environment variable not set" in str(excinfo.value)

@patch.dict(os.environ, {"LOCAL_HF_GENERIC_MODEL_PREFIX": ""}, clear=True) # Missing API base
def test_llm_factory_missing_env_vars_local_hf_generic():
    with pytest.raises(ValueError) as excinfo:
        LLMFactory.get_llm_provider("local_hf_generic")
    assert "LOCAL_HF_GENERIC_API_BASE not set" in str(excinfo.value)


# It's good practice to also test the _handle_llm_error method if it had more complex logic,
# but for now, its direct test might be coupled with testing a concrete provider's call.
# We can ensure its basic functionality through a concrete provider test later.
class ErrorTestProvider(LLMProvider):
    def generate_text(self, prompt: str, model_name: str, **kwargs) -> str:
        try:
            raise ValueError("Test LLM Error")
        except ValueError as e:
            self._handle_llm_error(e, context="text_generation")
        return "" # Should not be reached if exception is raised correctly

    def generate_code(self, prompt: str, model_name: str, **kwargs) -> str:
        # Not implementing for this test
        pass

    def generate_summary(self, prompt: str, model_name: str, **kwargs) -> str:
        # Not implementing for this test
        pass

def test_handle_llm_error_raising():
    provider = ErrorTestProvider()
    with pytest.raises(Exception) as excinfo:
        provider.generate_text("test", "test_model")
    assert "LLM API error during text_generation: Test LLM Error" in str(excinfo.value)

# Test that instantiating LLMProvider directly raises TypeError
def test_cannot_instantiate_llmprovider_directly():
    with pytest.raises(TypeError) as excinfo:
        LLMProvider()
    # Check for a core part of the message and the method names
    assert "Can't instantiate abstract class" in str(excinfo.value)
    assert "LLMProvider" in str(excinfo.value)
    assert "generate_code" in str(excinfo.value)
    assert "generate_summary" in str(excinfo.value)
    assert "generate_text" in str(excinfo.value)

# Ensure __init__.py exists for app.backend.llm_providers
# This will be done by a separate tool call if needed.

# Run pytest:
# Ensure you have an __init__.py in app/backend and app/backend/llm_providers
# PYTHONPATH=. pytest tests/test_llm_providers.py
# (or configure pytest to find your app root)
