# Project Title

This project is a tool that allows users to query structured data files (CSV, Parquet, RData, SQLite) using natural language. It translates natural language questions into SQL, R (data.table), or Python (Pandas) code, executes it, and returns the results.

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

4.  **Configure OpenAI API Access:**
    To enable the natural language to code (SQL, R, Python Pandas) generation feature, you need to provide access to an OpenAI model.
    **Do NOT hardcode your API keys in the code.** Use environment variables.

    Create a `.env` file in the project root directory by copying `.env.example` and filling in your API keys and desired configurations.
    ```bash
    cp .env.example .env
    # Now edit .env with your actual keys/endpoints
    ```
    The `.env.example` file provides placeholders for:
    *   **OpenAI API:**
        *   `OPENAI_API_KEY`: Your standard OpenAI API key.
        *   `OPENAI_API_BASE` (Optional): For custom OpenAI-compatible endpoints.
    *   **Azure OpenAI Service:** (These are used if `AZURE_OPENAI_ENDPOINT` is set)
        *   `OPENAI_API_KEY`: Your Azure OpenAI API key. (LiteLLM will also look for `AZURE_API_KEY`)
        *   `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI resource endpoint.
        *   `AZURE_DEPLOYMENT_NAME`: The name of your model deployment on Azure.
        *   `AZURE_OPENAI_API_VERSION` (Optional): API version, e.g., "2023-07-01-preview".
    *   **Google Gemini API:**
        *   `GEMINI_API_KEY`: Your Google AI Studio API key for Gemini models.
    *   **Local HuggingFace Models (via an OpenAI-compatible server like Ollama):**
        *   `LOCAL_HF_OLLAMA_MODEL_PREFIX`: e.g., "ollama/" (used by LiteLLM to identify Ollama models).
        *   `LOCAL_HF_OLLAMA_API_BASE`: e.g., "http://localhost:11434" (default Ollama API base).
        *   `LOCAL_HF_OLLAMA_API_KEY` (Optional): Usually not needed for local Ollama.
        *   (Similar variables for a generic local server: `LOCAL_HF_GENERIC_MODEL_PREFIX`, `LOCAL_HF_GENERIC_API_BASE`, `LOCAL_HF_GENERIC_API_KEY`)

    The application uses these environment variables to configure the selected LLM provider at runtime.

5.  **Run the Flask application:**
    ```bash
    # Ensure your virtual environment is active
    # source venv/bin/activate
    cd app/backend
    python app.py
    ```
    The application will typically be available at `http://127.0.0.1:5000`.

## Running Tests

The project uses Python's built-in `unittest` framework for testing.

1.  **Ensure all dependencies are installed:**
    ```bash
    pip install -r requirements.txt 
    # (No specific test dependencies needed if using unittest with Python 3.3+)
    ```

2.  **Navigate to the project root directory** (the one containing `app/` and `tests/`).

3.  **Run the tests using the following command:**
    ```bash
    python -m unittest discover -s tests -p "test_*.py"
    ```
    This command will discover and run all files named `test_*.py` within the `tests` directory.

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
