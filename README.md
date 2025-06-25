# Data Analytics Agent with Multi-LLM and Topology Support

This project is a tool that allows users to query structured data files (CSV, Parquet, RData, SQLite) using natural language. It translates natural language questions into SQL, R (data.table), or Python (Pandas) code, executes it, and returns the results.

**Note:** This project is currently undergoing a refactoring to integrate the **Google Agent Development Kit (ADK)**. This will enhance its capabilities for defining agent behavior, integrating various Large Language Models (LLMs), and managing complex task execution flows.

## User Guide

For detailed instructions on how to set up and use this application, especially if you have a non-coding background, please see our comprehensive [User Guide](./USER_GUIDE.md).

## Directory Structure

- `app/`: Contains the main application code.
  - `ui/`: Frontend components.
  - `backend/`: Server-side logic and API endpoints.
  - `data/`: Storage for uploaded user data and metadata.
  - `models/`: Storage or configuration for ML models.
  - `static/`: CSS, JavaScript, and images for the UI.
  - `templates/`: HTML templates (if using a framework like Flask or Django).
- `tests/`: Unit and integration tests.
- `README.md`: This file, providing a basic project description.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `requirements.txt`: Lists Python dependencies for the project.

## Prerequisites

This application requires Python 3.x. Python dependencies are listed in `requirements.txt`.

The application supports querying via different "agents":
- **SQL Agent**: Uses DuckDB to query CSV, Parquet, and SQLite files.
- **R Agent**: Uses R's `data.table` package to query RData (`.Rdata`, `.rda`) files.
- **Python Pandas Agent**: Uses Python's `pandas` library to query CSV and Parquet files.

### R Agent Requirements

To use the R data.table agent for querying `.Rdata` or `.rda` files, you also need:

1.  **R Installation**: R must be installed on the system and the `Rscript` command should be accessible from your system's PATH. You can download R from [CRAN (The Comprehensive R Archive Network)](https://cran.r-project.org/).
2.  **R Packages**: The following R packages are required. You can install them from an R console using:
    ```R
    install.packages(c("data.table", "jsonlite"))
    ```
    *   `data.table`: Used by the R agent to execute queries on data frames.
    *   `jsonlite`: Used to extract metadata from R data files (`.Rdata`, `.rda`) during the upload process.

### Python Pandas Agent Requirements

This agent uses Python's `pandas` library to execute queries on CSV and Parquet files.
The necessary Python dependencies, including `pandas` and `pyarrow` (for Parquet support), are listed in `requirements.txt` and will be installed when you run:
```bash
pip install -r requirements.txt
```
Key dependencies include Flask, Pandas, DuckDB, and an increasing reliance on `google-adk` for agent functionality and `litellm` for unified LLM API access.

## Architecture Overview (Transitioning to ADK)

The application consists of:
- A **Flask-based backend** (`app/backend/app.py`) that handles API requests, file uploads, and orchestrates the data analysis process.
- **Code Execution Utilities** (`app/backend/code_execution.py`) for running SQL, R, and Python (Pandas) scripts.
- **Google Agent Development Kit (ADK) components** (under `app/backend/adk_components/`): This is the new core for defining intelligent agents, managing tools (like code execution), and interacting with various LLMs. The custom LLM provider and topology engine previously developed are being refactored into or replaced by ADK constructs.
- A **frontend** (HTML templates in `app/templates/` and static assets) for user interaction.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure LLM API Access:**
    The application leverages Large Language Models (LLMs) for natural language understanding and code generation. With the integration of Google ADK, it aims to be model-agnostic.
    **Do NOT hardcode your API keys in the code.** Use environment variables by creating a `.env` file.

    1.  Copy the example configuration:
        ```bash
        cp .env.example .env
        ```
    2.  Edit the `.env` file with your API keys and endpoints. Refer to `.env.example` for the full list of supported variables. Key variables include:
        *   `OPENAI_API_KEY`: For OpenAI models. Also used for Azure OpenAI (where it serves as `AZURE_API_KEY` for LiteLLM/ADK).
        *   `AZURE_OPENAI_ENDPOINT`, `AZURE_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_VERSION`: For Azure OpenAI.
        *   `GEMINI_API_KEY`: For Google Gemini models. ADK uses this directly.
        *   `LOCAL_HF_...` variables: For connecting to locally hosted HuggingFace models via an OpenAI-compatible API (e.g., Ollama). ADK's support for these local models is under development; currently, the custom framework path might use these via LiteLLM.

    The application's LLM factories (both the older custom one and the newer ADK-based one) will attempt to use these environment variables to configure the selected LLM provider at runtime. Ensure the correct keys are set for the `llm_choice` you intend to use.

5.  **Run the Flask application:**
    ```bash
    # Ensure your virtual environment is active
    # source venv/bin/activate
    cd app/backend
    python app.py
    ```
    The application will typically be available at `http://127.0.0.1:5000`.

## Running Tests

The project uses `pytest` for testing.

1.  **Ensure all dependencies, including test dependencies, are installed:**
    ```bash
    pip install -r requirements.txt 
    ```
    (This includes `pytest` and `pytest-mock`).

2.  **Navigate to the project root directory** (the one containing `app/` and `tests/`).

3.  **Run the tests using the following command:**
    ```bash
    PYTHONPATH=. pytest
    ```
    Or simply:
    ```bash
    pytest
    ```
    This command will discover and run all tests.

## Sample Data

This directory contains sample data files in various formats for testing purposes.

- **R:** `sample_data/sample_data.Rdata` - An R data file containing a sample dataframe. Generated by `sample_data/create_sample_r_data.R`.
- **SQL:** `sample_data/sample_data.db` - A SQLite database file with a sample table. Generated by `sample_data/create_sample_sql_data.py`.
- **Python Pandas:**
    - `sample_data/sample_data.csv` - Sample data in CSV format.
    - `sample_data/sample_data.parquet` - Sample data in Parquet format.
    - Both generated by `sample_data/create_sample_pandas_data.py`.

These files can be used to test the agent's ability to process and analyze data from different sources.

## Contributing

[Guidelines for contributing to the project]

## License

[Information about the project's license]
