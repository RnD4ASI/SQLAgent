from google.adk.agents import LlmAgent
import os

# This module will define and configure ADK agents.

def create_basic_gemini_agent() -> LlmAgent:
    """
    Creates a very basic ADK LlmAgent configured to use a Gemini model.
    This agent has no tools and a simple instruction.
    """
    # ADK typically relies on environment variables for API keys.
    # Ensure GEMINI_API_KEY is set in the environment.
    # The specific model name might need adjustment based on availability
    # and what ADK expects (e.g., "gemini-pro", "gemini-1.5-flash-latest").
    # Using a common one for now.
    gemini_model_name = os.environ.get("ADK_GEMINI_MODEL", "gemini-1.5-flash-latest")

    if not os.environ.get("GEMINI_API_KEY"):
        # This is a fallback/warning. In a real scenario, the calling code
        # (e.g., a factory or app.py) should ensure keys are present
        # before attempting to create an agent that needs them.
        # ADK itself might also raise errors if keys are missing for selected models.
        print("Warning: GEMINI_API_KEY not found in environment. Basic Gemini agent might fail.")
        # Alternatively, could raise an error here:
        # raise ValueError("GEMINI_API_KEY not set, cannot create Gemini agent.")

    agent = LlmAgent(
        name="BasicGeminiAssistant",
        model=gemini_model_name, # ADK will try to use this with Google's backend
        instruction="You are a helpful assistant. Please respond clearly and concisely to the user's query.",
        description="A basic assistant powered by a Gemini model.",
        tools=[] # No tools for this basic agent yet
    )
    return agent

if __name__ == '__main__':
    # Example of how to use this function (requires GEMINI_API_KEY to be set)
    # Load .env for local testing if needed
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    load_dotenv(dotenv_path)

    print("Attempting to create basic Gemini agent...")
    try:
        basic_agent = create_basic_gemini_agent()
        print(f"Agent '{basic_agent.name}' created successfully with model '{basic_agent.model}'.")

        # To actually run it (example, might need more setup like a session):
        # from google.adk.sessions import Session as AdkSession # Actual class is Session
        # with AdkSession() as session:
        #     response = basic_agent.send_sync(session=session, message="Hello, world!")
        #     print(f"Agent response: {response.message}")

    except Exception as e:
        print(f"Error creating or testing basic agent: {e}")
        print("Ensure your GEMINI_API_KEY is set in your .env file or environment.")
        print("You might also need to run 'gcloud auth application-default login' if ADK uses ADC.")

    # TODO: Add similar functions or a factory for creating agents with other models (OpenAI, Local HF)
    # once ADK's mechanism for specifying non-Gemini models is clear.
    # For example:
    # def create_basic_openai_agent(api_key: str, model_name: str = "gpt-3.5-turbo"):
    #     # How does ADK take the API key and specify OpenAI?
    #     # Does it use LiteLLM, or do we need to configure a custom model service?
    #     pass
