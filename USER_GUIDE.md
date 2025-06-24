# Welcome to Your SQL Data Assistant!

Hello! This guide will help you use the SQL Data Assistant. Think of this tool as a helpful assistant that can look at your data and answer your questions about it in plain English.

**What this tool does:**
*   Lets you upload your data files (like spreadsheets or simple databases).
*   Understands your questions about your data when you type them in plain English.
*   Shows you the answers in English, as well as the data itself in a table.
*   Can even create simple charts from your data!

**Who this guide is for:**
This guide is made for everyone, especially if you don't have a coding background. We'll walk you through each step. We assume you're using Visual Studio Code (VS Code) as your main tool.

**What you'll be able to do:**
By the end of this guide, you'll know how to:
*   Set up the SQL Data Assistant on your computer.
*   Upload your own data.
*   Ask questions about your data and get answers.
*   See simple charts of your data.

Let's get started!

---

## 2. Getting Ready: What You Need

Before we begin, you'll need a couple of things installed on your computer:

*   **Visual Studio Code (VS Code):**
    *   This is a very popular and free tool that helps people work with code and projects. We'll use it to manage the assistant's files and run a couple of simple commands.
    *   **Download VS Code here:** [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
    *   After installing VS Code, please also install the official **Python extension** from Microsoft.
        *   Open VS Code.
        *   Click on the Extensions icon on the left sidebar (it looks like four squares, with one detached). `[Screenshot: VS Code Extensions icon]`
        *   In the search bar, type `Python` and look for the one published by `Microsoft`. Click "Install". `[Screenshot: Python extension by Microsoft in VS Code Marketplace]`

*   **Python:**
    *   This is the programming language the SQL Data Assistant is built with. You need it on your computer for the assistant to work.
    *   **Download Python here:** [https://www.python.org/downloads/](https://www.python.org/downloads/) (We recommend downloading the latest stable version for Windows, macOS, or Linux).
    *   **Important for Windows users:** When you run the Python installer, make sure to check the box that says **"Add Python to PATH"** or **"Add python.exe to PATH"** before you click "Install Now". This will make things much easier. `[Screenshot: Python installer on Windows with "Add Python to PATH" checked]`

---

## 3. Setting Up Your SQL Data Assistant

Now let's get the assistant's files onto your computer.

### Step 3.1: Get the Project Files

You have two main options:

*   **Option A: Download as a ZIP file (Recommended for most users)**
    1.  Go to the project's page (e.g., on GitHub, where you found this guide).
    2.  Look for a "Code" button, and then click "Download ZIP". `[Screenshot: GitHub "Code" button with "Download ZIP" option]`
    3.  Save the ZIP file to a place you can easily find, like your `Documents` folder or `Desktop`.
    4.  Go to where you saved the ZIP file, right-click on it, and choose "Extract All..." or "Unzip". Choose a clear folder name for the extracted files, for example, `SQL_Data_Assistant`.

*   **Option B: Using Git (If you're familiar with it)**
    1.  If you have Git installed, you can open a terminal or command prompt and type:
        ```bash
        git clone <URL_OF_THE_PROJECT_REPOSITORY>
        ```
        Replace `<URL_OF_THE_PROJECT_REPOSITORY>` with the actual web address of the project. This will create a folder with the project files.

### Step 3.2: Open the Project in VS Code

1.  Open Visual Studio Code.
2.  Go to the menu and click `File > Open Folder...`.
3.  Navigate to the folder where you extracted or cloned the project files (e.g., `Documents/SQL_Data_Assistant`) and click "Select Folder".
    `[Screenshot: VS Code "Open Folder" dialog]`

### Step 3.3: Install Required Tools (Dependencies)

The assistant needs some extra "helper tools" (called dependencies) to work correctly. We'll install them using a feature in VS Code called the Terminal.

1.  In VS Code, open the Terminal by clicking `Terminal > New Terminal` in the top menu. A panel should appear at the bottom of the VS Code window. `[Screenshot: VS Code Terminal menu option and the Terminal panel]`
2.  In the Terminal, type the following command exactly and press Enter:
    ```bash
    pip install -r requirements.txt
    ```
    *   (What's `pip`? It's Python's tool for installing packages. `requirements.txt` is a list of all the helper tools our assistant needs.)
3.  You'll see text appearing in the terminal as the tools are downloaded and installed. This might take a few minutes. Wait for it to finish. You should see a message indicating success, and the command prompt should reappear.

### Step 3.4: Set Up Your OpenAI API Key

The "brain" of our SQL Data Assistant is a powerful language model from OpenAI. To use it, you need an "API Key," which is like a private password.

1.  **Get an OpenAI API Key:**
    *   Go to the OpenAI platform: [https://platform.openai.com/](https://platform.openai.com/)
    *   You'll need to sign up for an account if you don't have one.
    *   Once logged in, navigate to the API key section (usually under your account settings or a dedicated "API Keys" menu). `[Screenshot: OpenAI API Key section on their website]`
    *   Create a new secret key. Copy this key immediately and save it somewhere safe temporarily (like Notepad), as OpenAI might not show it to you again. **Treat this key like a password! Don't share it.**

2.  **Tell the Assistant Your API Key (Using a `.env` file - Recommended for VS Code)**
    This is the easiest and safest way for most users within VS Code. We'll create a special file called `.env` directly in your project to store the API key. This file is already listed in `.gitignore`, so you won't accidentally share your key if you use Git.

    1.  In VS Code, make sure you are in the main project folder (e.g., `SQL_Data_Assistant`).
    2.  Create a new file: Click `File > New File`.
    3.  Save this empty file immediately: Click `File > Save As...` and name it exactly `.env` (dot env). Make sure it's in the main project folder, at the same level as `requirements.txt`. `[Screenshot: VS Code explorer showing the .env file at the root]`
    4.  Open the `.env` file and type the following, replacing `"your_actual_key_here"` with the API key you copied from OpenAI:
        ```env
        # For OpenAI (Standard or Azure)
        OPENAI_API_KEY="your_actual_openai_or_azure_api_key"

        # If using Azure OpenAI, also uncomment and fill these:
        # AZURE_OPENAI_ENDPOINT="your_azure_endpoint_here"
        # AZURE_DEPLOYMENT_NAME="your_azure_deployment_name_here"
        # AZURE_OPENAI_API_VERSION="2023-07-01-preview" # Or your specific version

        # If using Google Gemini
        # GEMINI_API_KEY="your_gemini_api_key"

        # If using a local HuggingFace model (e.g., via Ollama)
        # Ensure your local model server is running.
        # LOCAL_HF_OLLAMA_API_BASE="http://localhost:11434" # Default for Ollama
        # LOCAL_HF_OLLAMA_MODEL_PREFIX="ollama/"
        # (The actual model like 'mistral' is chosen in the UI)
        ```
        Refer to the `.env.example` file in the project for a full list of possible environment variables.
    5.  Save the `.env` file. The application is set up to read this file automatically when it starts.

    *(Alternative for advanced users: You can also set environment variables directly in your operating system. Instructions for this are in the main `README.md`.)*

### Step 3.5: Choosing Your AI Model and Strategy (New!)

With recent updates, you can now choose which AI model provider and which "thinking strategy" (topology) the assistant uses!

*   **LLM Choice:** In the UI, you'll find a dropdown to select your preferred Large Language Model provider:
    *   `OpenAI` (uses your OpenAI or Azure OpenAI key)
    *   `Gemini` (uses your Google Gemini key)
    *   `Local Ollama` (connects to a locally running Ollama instance for HuggingFace models - an advanced option)
    *   *(Make sure you have configured the corresponding API key in your `.env` file for your chosen LLM.)*
*   **Topology Choice:** This dropdown lets you pick how the AI tackles your query:
    *   `Sequential Reflect` (Default): The AI generates code, runs it. If there's an error, it tries to fix it once. Then summarizes. (This was the original behavior).
    *   `Parallel Ensemble`: The AI asks several models (or model configurations) to generate code at the same time and picks the first one that works.
    *   `Iterative Reason+Act`: The AI thinks step-by-step, deciding what to do (like generate code, run code, or give an answer) in a loop, much like a human detective.
    *   `Default Fallback`: Tries the `Parallel Ensemble` first. If that doesn't find a working solution, it falls back to the `Sequential Reflect` method.

    For most users, starting with `OpenAI` as the LLM and `Default Fallback` or `Sequential Reflect` as the topology is a good balance. Experiment to see what works best for your data and questions!
    `[Screenshot: UI showing new dropdowns for LLM Choice and Topology Choice]`


---

## 4. Running Your SQL Data Assistant

You're all set up! Now let's start the assistant.

### Step 4.1: Start the Application

1.  Make sure you have the project open in VS Code.
2.  Open the Terminal in VS Code if it's not already open (`Terminal > New Terminal`).
3.  In the Terminal, you need to navigate into the `app/backend` folder where the main application script is. Type this command and press Enter:
    ```bash
    cd app/backend
    ```
    Your terminal prompt should change to show you're in the `app/backend` directory.
4.  Now, type the following command and press Enter to start the assistant:
    ```bash
    python app.py
    ```
5.  You should see some messages in the terminal. Look for a line that says something like:
    `* Running on http://127.0.0.1:5000` (or `http://localhost:5000`)
    This means the assistant is running! `[Screenshot: VS Code terminal showing the Flask app running and the URL]`
    Keep this terminal open. If you close it, the assistant will stop.

### Step 4.2: Open the Assistant in Your Web Browser

1.  Open your favorite web browser (like Chrome, Firefox, Edge, or Safari).
2.  In the address bar, type the address shown in the terminal: `http://127.0.0.1:5000` (or `http://localhost:5000`) and press Enter.
3.  You should see the SQL Data Assistant's main page! `[Screenshot: Main UI of the SQL agent in a web browser]`

---

## 5. Using the SQL Data Assistant: A Walkthrough

Let's explore how to use the assistant.

### Step 5.1: The Main Screen

You'll see a few key areas:
*   **Upload Data:** For choosing your data file.
*   **Ask a question...:** Where you'll type your questions.
*   **Chat History:** Where your questions and the assistant's answers (including data tables and summaries) will appear.
*   **Plot Display:** Where charts will show up if you request them.

### Step 5.2: Uploading Your Data

1.  Click the **"Choose File"** button. A dialog will open.
2.  Navigate to your data file and select it. The assistant can understand:
    *   `.csv` (Comma Separated Values - common for spreadsheets)
    *   `.parquet` (a modern, efficient data format)
    *   `.sqlite` (a simple database file)
3.  After selecting a file, click the **"Upload Data"** button.
4.  **Metadata Display:** Once uploaded, you'll see some "Metadata" information.
    *   **What is Metadata?** It's just information *about* your data. The assistant tries to guess your table's name (usually from the filename), its column names, and the type of data in each column (like Text, Number, etc.).
    *   You don't usually need to change this, but it's what the assistant uses to understand your data. `[Screenshot: Metadata section of the UI showing table name, column names, and types]`

### Step 5.3: Asking Questions (Natural Language Queries)

1.  In the text box that says **"Ask a question about the loaded data..."**, type your question in plain English.
    *   Examples:
        *   "How many entries are there?"
        *   "What are the different categories in the 'product_type' column?"
        *   "Show me all data where 'city' is 'New York'."
        *   "What is the average value for 'sales_amount'?"
2.  Click the **"Generate SQL & Get Answer"** button.

### Step 5.4: Understanding the Response

The assistant will reply in the "Chat History" area. You'll typically see:

1.  **Natural Language Summary:** A plain English sentence or two answering your question. `[Screenshot: Chat history showing the natural language summary]`
2.  **Generated SQL Query:** This is the special command the assistant wrote to find the answer in your data. You don't need to understand this, but it's there for those who are curious. `[Screenshot: Chat history showing the generated SQL query]`
3.  **Table Results:** A table showing the actual data rows that match your question. `[Screenshot: Chat history showing the tabular results]`

### Step 5.5: Visualizing Your Data (Plots)

If your query results are suitable for a chart, you can try to visualize them.

1.  After you get a response with a data table, click the **"Request Plot"** button.
2.  If the assistant can make a sensible chart from your results (like a bar chart, line chart, or histogram), it will appear in the "Plot Display" area. `[Screenshot: Plot display area showing a sample chart]`
    *   *Note:* Not all data can be easily plotted, or the assistant might not always guess the best chart type.

---

## 6. Troubleshooting Common Issues

If things aren't working as expected, here are a few common fixes:

*   **"I see an error message about API Key / No response from assistant."**
    *   **Check your API Key:** Double-check that your `.env` file has the correct `OPENAI_API_KEY` and that the file is saved. **Crucially, ensure the `.env` file is in the main project folder (e.g., `SQL_Data_Assistant`), not inside the `app` or `app/backend` folders.**
    *   **Restart the Application:** Stop the application in the VS Code terminal (Ctrl+C) and start it again (`python app.py` in the `app/backend` folder). This helps pick up new `.env` file settings.
    *   **Internet Connection:** Ensure you are connected to the internet (OpenAI is an online service).
*   **"The application (Python script) isn't starting in the VS Code terminal."**
    *   **Correct Folder?:** Make sure your terminal is in the `app/backend` folder before running `python app.py`. You can type `cd ../` to go up one folder or `cd folder_name` to go into one.
    *   **Python Installed Correctly?:** Did you check "Add Python to PATH" during installation (for Windows)? Try closing and reopening VS Code.
    *   **Dependencies Installed?:** Did the `pip install -r requirements.txt` command complete without errors?
*   **"File not uploading / Error on upload."**
    *   **Supported Format?:** Is your file a `.csv`, `.parquet`, or `.sqlite` file?
    *   **File Corrupted?:** The file itself might have issues. Try opening it in another program (like Excel for CSVs) to see if it's readable.
*   **"The plot isn't showing or looks weird."**
    *   The current plotting feature is basic. It tries its best to guess a good chart. Some data is harder to visualize automatically. Try asking a question that results in simpler data.

---

## 7. Stopping the Application

When you're done using the SQL Data Assistant:

1.  Go back to the VS Code window where the Python application is running in the Terminal.
2.  Click into the Terminal panel.
3.  Press `Ctrl + C` (hold down the Ctrl key and press C).
4.  You might be asked "Terminate batch job (Y/N)?". Type `Y` and press Enter.
5.  The application will stop. You can then close your web browser tab and VS Code.

---

Congratulations! You've learned how to set up and use your SQL Data Assistant. We hope it helps you explore and understand your data more easily!
