# Project Title

This project is a [brief description of the project].

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

### R Agent Requirements

To use the R data.table agent for querying `.Rdata` or `.rda` files, you also need:

1.  **R Installation**: R must be installed on the system and the `Rscript` command should be accessible from your system's PATH. You can download R from [CRAN (The Comprehensive R Archive Network)](https://cran.r-project.org/).
2.  **R Packages**: The following R packages are required. You can install them from an R console using:
    ```R
    install.packages(c("data.table", "jsonlite"))
    ```
    *   `data.table`: Used by the R agent to execute queries on data frames.
    *   `jsonlite`: Used to extract metadata from R data files (`.Rdata`, `.rda`) during the upload process.

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
    To enable the natural language to SQL generation feature, you need to provide access to an OpenAI model.
    **Do NOT hardcode your API keys in the code.** Use environment variables.

    *   **For standard OpenAI API:**
        Set the following environment variables:
        ```bash
        export OPENAI_API_KEY="your_openai_api_key"
        # Optional: If using a custom endpoint or proxy
        # export OPENAI_API_BASE="your_custom_openai_api_base_url" 
        ```

    *   **For Azure OpenAI Service:**
        Set the following environment variables:
        ```bash
        export AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint" 
        export OPENAI_API_KEY="your_azure_openai_api_key"
        # Also ensure you set AZURE_DEPLOYMENT_NAME in app/backend/app.py or as an environment variable
        # if it's specific to your setup, e.g.:
        export AZURE_DEPLOYMENT_NAME="your_deployment_name" 
        ```
        *Note: The application currently looks for `AZURE_DEPLOYMENT_NAME` as an environment variable if using Azure. You might need to adjust `app/backend/app.py` if your Azure setup differs in how the deployment/engine is specified.*


5.  **Run the Flask application:**
    ```bash
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

## Contributing

[Guidelines for contributing to the project]

## License

[Information about the project's license]
