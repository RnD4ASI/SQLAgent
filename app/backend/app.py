from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
import openai
import duckdb
import matplotlib
matplotlib.use('Agg') # Use Agg backend for web server
import matplotlib.pyplot as plt
import io
import base64
import subprocess
import json
import tempfile
import sys
import pyarrow as pa

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')) # Load from project root

app = Flask(__name__, template_folder='../templates', static_folder='../static')
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
ALLOWED_EXTENSIONS = {'csv', 'parquet', 'sqlite', 'rdata', 'rda'}
# Global variable to cache the last successful DataFrame
last_successful_df = None
# Store the last uploaded filename globally for query execution
# In a real app, this should be managed per session or via a more robust mechanism
current_uploaded_filename = None
current_uploaded_filepath = None # Store full path for convenience

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Attempt to get OpenAI API key and endpoint from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE") # For self-hosted or Azure
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT") # For Azure

if AZURE_OPENAI_ENDPOINT:
    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = "2023-07-01-preview" # Specify a valid API version for Azure
    # For Azure, API key is typically set as openai.api_key or handled by Azure AD
    if OPENAI_API_KEY: # Some Azure setups might still use an API key
        openai.api_key = OPENAI_API_KEY
elif OPENAI_API_BASE: # For standard OpenAI or other compatible APIs that require a base URL
    openai.api_base = OPENAI_API_BASE
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
    else:
        print("Warning: OPENAI_API_KEY not set for custom OpenAI endpoint.")
elif OPENAI_API_KEY: # For standard OpenAI API
     openai.api_key = OPENAI_API_KEY
else:
    print("Warning: OpenAI API key or endpoint not configured. The /query endpoint will not work.")

# Store metadata globally for simplicity in this example
current_metadata = None
# Global variable to cache the last successful DataFrame
last_successful_df = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def execute_duckdb_query(sql_query: str, file_path: str, table_name: str):
    """
    Executes a SQL query against a data file (CSV, Parquet, or SQLite) using DuckDB.

    Args:
        sql_query (str): The SQL query to execute.
        file_path (str): The full path to the data file.
        table_name (str): The name of the table to be used in DuckDB (often derived from filename).
                          For SQLite, this is the table name within the .sqlite file.

    Returns:
        tuple: (pandas.DataFrame, None) if successful, or (None, str) if an error occurred.
    """
    if not file_path:
        return None, "File path is missing."
        
    file_ext = file_path.rsplit('.', 1)[1].lower()

    try:
        if file_ext == 'sqlite':
            # For SQLite, connect directly to the file. DuckDB handles this.
            # The table_name parameter is crucial here as it refers to the table within the SQLite DB.
            con = duckdb.connect(database=file_path, read_only=True) # Read-only is safer for existing SQLite DBs
        else:
            # For CSV/Parquet, use an in-memory DuckDB and load the file.
            con = duckdb.connect(database=':memory:', read_only=False)
            if file_ext == 'csv':
                # DuckDB uses the filename (without extension) as table name by default if not specified
                # Or we can explicitly name it using the table_name parameter
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
            elif file_ext == 'parquet':
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')")
            else:
                return None, f"Unsupported file type for querying: {file_ext}"
        
        result_df = con.execute(sql_query).fetchdf()
        con.close()
        return result_df, None # Success
    except duckdb.Error as e:
        # Attempt to close connection if it was opened
        try:
            if 'con' in locals() and con:
                con.close()
        except Exception:
            pass # Ignore errors during close on error
        return None, f"DuckDB SQL execution error: {str(e)}"
    except FileNotFoundError:
        return None, f"Data file not found: {file_path}"
    except Exception as e:
        # Attempt to close connection
        try:
            if 'con' in locals() and con:
                con.close()
        except Exception:
            pass
        return None, f"An error occurred during SQL execution: {str(e)}"

def execute_r_script(r_code_string: str, rdata_file_path: str, target_object_name: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Executes R data.table commands on an object within an Rdata file and returns the result as a Pandas DataFrame.

    Args:
        r_code_string (str): The R code (data.table commands) to execute on target_object_name.
        rdata_file_path (str): Full path to the .Rdata or .rda file.
        target_object_name (str): Name of the data object within the Rdata file.

    Returns:
        tuple: (pandas.DataFrame, None) if successful, or (None, str) if an error occurred.
    """
    temp_r_script_file = None
    temp_csv_file = None

    try:
        # Sanitize paths for R (forward slashes)
        rdata_file_path_r = rdata_file_path.replace('\\', '/')

        # Create a temporary file for the R script
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.R', encoding='utf-8') as temp_r_script_file_obj:
            temp_r_script_path = temp_r_script_file_obj.name
            temp_r_script_path_r = temp_r_script_path.replace('\\', '/') # For Rscript execution if on Windows

        # Create a temporary file path for the output CSV (R will write to this)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_csv_file_obj:
            temp_csv_path = temp_csv_file_obj.name
            temp_csv_path_r = temp_csv_path.replace('\\', '/') # For R script

        r_script_content = f"""
        options(error = function() {{
            cat(geterrmessage(), file = stderr())
            quit(save = "no", status = 1, runLast = FALSE)
        }})

        if (!requireNamespace("data.table", quietly = TRUE)) {{
            write("Error: data.table package is not installed.", stderr())
            quit(save = "no", status = 1, runLast = FALSE)
        }}
        library(data.table)

        tryCatch({{
            load_env <- new.env()
            load("{rdata_file_path_r}", envir=load_env)

            if (!exists("{target_object_name}", envir=load_env)) {{
                stop(paste0("Object '", "{target_object_name}", "' not found in the Rdata file."))
            }}

            active_df <- load_env[["{target_object_name}"]]

            if (!is.data.table(active_df)) {{
                if (is.data.frame(active_df)) {{
                    active_df <- as.data.table(active_df)
                }} else {{
                    stop(paste0("Object '", "{target_object_name}", "' is not a data.frame or data.table."))
                }}
            }}

            # User's R code is executed here. It's expected to modify active_df or create a result.
            # For simplicity, we assume the user's code assigns the final result back to 'active_df'.
            # More complex scenarios might need active_df <- eval(parse(text=...)) if r_code_string is an expression
            eval(parse(text = {repr(r_code_string)})) # Use repr to correctly escape the r_code_string

            if (!exists("active_df")){{
                 stop("The R code did not result in an 'active_df' object.")
            }}

            if (nrow(active_df) == 0) {{
                # fwrite creates an empty file for a 0-row data.table, which is fine.
                # However, if it's not a data.table at this point, it's an issue.
                 if (!is.data.table(active_df)) {{
                    stop("Result of R code is not a data.table and is empty.")
                }}
            }}

            fwrite(active_df, file="{temp_csv_path_r}", row.names=FALSE)

        }}, error = function(e) {{
            write(paste("R script execution error:", e$message), stderr())
            quit(save = "no", status = 1, runLast = FALSE)
        }})

        quit(save = "no", status = 0, runLast = FALSE) # Explicit success exit
        """

        with open(temp_r_script_path, 'w', encoding='utf-8') as f:
            f.write(r_script_content)

        # Execute the R script
        process = subprocess.run(
            ['Rscript', temp_r_script_path],
            capture_output=True, text=True, check=False, encoding='utf-8'
        )

        if process.returncode == 0:
            if os.path.exists(temp_csv_path) and os.path.getsize(temp_csv_path) > 0:
                try:
                    df = pd.read_csv(temp_csv_path)
                    return df, None
                except pd.errors.EmptyDataError:
                     # Can happen if R wrote an empty file (e.g. 0-row data.table but with columns)
                    return pd.DataFrame(), None # Return empty DataFrame
                except Exception as e_read:
                    return None, f"Error reading R script output CSV: {str(e_read)}. R stderr: {process.stderr.strip()}"
            elif os.path.exists(temp_csv_path) and os.path.getsize(temp_csv_path) == 0: # Empty file means empty dataframe
                return pd.DataFrame(), None
            else:
                return None, f"R script executed successfully but output CSV not found or empty. R stderr: {process.stderr.strip()}"
        else:
            error_message = f"R script execution failed (return code {process.returncode}). Error: {process.stderr.strip()}"
            if not process.stderr.strip():
                 error_message = f"R script execution failed (return code {process.returncode}) with no specific error message."
            return None, error_message

    except FileNotFoundError: # Rscript not found
        return None, "Rscript command not found. Please ensure R is installed and in PATH."
    except Exception as e:
        return None, f"Python error during R script execution: {str(e)}"
    finally:
        # Clean up temporary files
        # temp_r_script_path and temp_csv_path are defined if the with blocks were entered.
        if 'temp_r_script_path' in locals() and os.path.exists(temp_r_script_path):
            try:
                os.remove(temp_r_script_path)
            except Exception as e_clean_r:
                print(f"Warning: Could not delete temporary R script {temp_r_script_path}: {e_clean_r}")
        if 'temp_csv_path' in locals() and os.path.exists(temp_csv_path):
            try:
                os.remove(temp_csv_path)
            except Exception as e_clean_csv:
                 print(f"Warning: Could not delete temporary CSV file {temp_csv_path}: {e_clean_csv}")


def execute_python_pandas_code(python_code_string: str, data_file_path: str, dataframe_name: str = 'df') -> tuple[pd.DataFrame | None, str | None]:
    """
    Executes Python Pandas code securely using a subprocess.

    Args:
        python_code_string (str): The Python code string to execute.
                                  This code is expected to operate on a DataFrame
                                  named `dataframe_name`.
        data_file_path (str): Full path to the data file (CSV or Parquet).
        dataframe_name (str): The name of the Pandas DataFrame variable in the executed code.

    Returns:
        tuple: (pandas.DataFrame, None) if successful, or (None, str) if an error occurred.
    """
    temp_script_file = None
    temp_output_csv_file = None
    temp_user_code_file = None

    try:
        # Create a temporary file for the user's Python code string
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as temp_user_code_file_obj:
            temp_user_code_path = temp_user_code_file_obj.name
            temp_user_code_file_obj.write(python_code_string)
            temp_user_code_file_obj.flush() # Ensure it's written

        # Create a temporary file for the main Python script to execute
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as temp_script_file_obj:
            temp_script_path = temp_script_file_obj.name

        # Create a temporary file path that the main script will use to save its output CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_output_csv_file_obj:
            temp_output_csv_path = temp_output_csv_file_obj.name


        # Sanitize paths for use in the script string (especially for Windows)
        data_file_path_script = data_file_path.replace('\\', '/')
        output_csv_path_script = temp_output_csv_path.replace('\\', '/')
        user_code_path_script = temp_user_code_path.replace('\\', '/')

        # Constructing script content line by line to avoid potential issues with large f-string blocks
        script_lines = [
            "import pandas as pd",
            "import sys",
            "import os",
            "",
            f"data_path = r'{data_file_path_script}'",
            f"df_name = '{dataframe_name}'",
            f"output_csv_path = r'{output_csv_path_script}'",
            f"user_code_path = r'{user_code_path_script}'",
            "",
            "try:",
            "    with open(user_code_path, 'r', encoding='utf-8') as f:",
            "        user_code = f.read()",
            "",
            "    # Load the dataframe",
            "    if data_path.endswith('.csv'):",
            f"        globals()[df_name] = pd.read_csv(data_path)",
            "    elif data_path.endswith('.parquet'):",
            f"        globals()[df_name] = pd.read_parquet(data_path)",
            "    else:",
            "        raise ValueError(f\"Unsupported file type: {data_path}. Only CSV and Parquet are supported.\")", # Note: escaped quote for f-string within f-string
            "",
            "    # Execute the user's code",
            "    exec(user_code, globals())",
            "",
            "    # After execution, retrieve the DataFrame by `df_name`.",
            "    if df_name not in globals():",
            "        print(f\"Error: DataFrame '{df_name}' not found after code execution. Did you delete or rename it?\", file=sys.stderr)",
            "        sys.exit(1)",
            "",
            "    result_df = globals()[df_name]",
            "",
            "    if isinstance(result_df, pd.DataFrame):",
            "        result_df.to_csv(output_csv_path, index=False)",
            "        print(output_csv_path) # Success: print the path of the output CSV to stdout",
            "    else:",
            "        print(f\"Error: Resulting object '{df_name}' is not a Pandas DataFrame (type: {type(result_df)}).\", file=sys.stderr)",
            "        sys.exit(1)",
            "",
            "except FileNotFoundError as e_fnf:",
            "    print(f\"Error loading data: {e_fnf}\", file=sys.stderr)",
            "    sys.exit(1)",
            "except pd.errors.EmptyDataError as e_ede:",
            "    print(f\"Error loading data: The file '{data_path}' is empty or contains no data.\", file=sys.stderr)",
            "    sys.exit(1)",
            "except ValueError as e_ve: # Catch our unsupported file type error",
            "    print(f\"Error: {e_ve}\", file=sys.stderr)",
            "    sys.exit(1)",
            "except Exception as e:",
            "    print(f\"Error during Python code execution: {str(e)}\", file=sys.stderr)",
            "    sys.exit(1)",
        ]
        script_content = "\n".join(script_lines)

        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Execute the temporary Python script using the same Python interpreter
        # sys.executable ensures we use the same python that runs the main app
        process = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True, text=True, check=False, encoding='utf-8'
        )

        if process.returncode == 0:
            # Successfully executed, stdout should contain the path to the output CSV
            output_file_from_script = process.stdout.strip()
            if os.path.exists(output_file_from_script):
                try:
                    # Read the resulting DataFrame from the script's output CSV
                    returned_df = pd.read_csv(output_file_from_script)
                    return returned_df, None
                except pd.errors.EmptyDataError:
                    # If the script outputs an empty CSV (e.g., df with 0 rows)
                    return pd.DataFrame(), None
                except Exception as e_read_csv:
                    return None, f"Error reading result CSV from script: {str(e_read_csv)}. Stderr: {process.stderr.strip()}"
            else:
                # This case should ideally not be reached if returncode is 0 and script prints path
                return None, f"Script executed successfully but output file '{output_file_from_script}' not found. Stderr: {process.stderr.strip()}"
        else:
            # Script execution failed
            error_message = f"Python script execution failed (return code {process.returncode}). Error: {process.stderr.strip()}"
            if not process.stderr.strip(): # Provide a generic message if stderr is empty
                 error_message = f"Python script execution failed (return code {process.returncode}) with no specific error message from stderr."
            return None, error_message

    except FileNotFoundError: # For issues finding python interpreter or script itself (less likely with NamedTemporaryFile)
        return None, "Error: Python interpreter or temporary script file not found."
    except Exception as e:
        # Catch-all for errors in this function itself (e.g., temp file creation issues)
        return None, f"Python error in 'execute_python_pandas_code' function: {str(e)}"
    finally:
        # Clean up temporary files
        if temp_script_path and os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
            except Exception as e_clean_script:
                print(f"Warning: Could not delete temporary Python script {temp_script_path}: {e_clean_script}")

        if temp_user_code_path and os.path.exists(temp_user_code_path):
            try:
                os.remove(temp_user_code_path)
            except Exception as e_clean_user_code:
                print(f"Warning: Could not delete temporary user code file {temp_user_code_path}: {e_clean_user_code}")

        # temp_output_csv_path is the path *string*. The actual file is temp_output_csv_file_obj.name
        # We need to ensure it's cleaned up if it was created.
        # The variable temp_output_csv_file_obj from the with statement might not be in scope here.
        # We stored its name in temp_output_csv_path.
        if temp_output_csv_path and os.path.exists(temp_output_csv_path): # Check path string
             try:
                 os.remove(temp_output_csv_path)
             except Exception as e_clean_csv:
                 print(f"Warning: Could not delete temporary output CSV file {temp_output_csv_path}: {e_clean_csv}")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_uploaded_filename, current_metadata, current_uploaded_filepath
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_uploaded_filename = filename
        current_uploaded_filepath = filepath # Store full path

        current_metadata = None

        metadata_inferred = {}
        if filename.rsplit('.', 1)[1].lower() == 'csv':
            try:
                df = pd.read_csv(filepath, nrows=100)
                metadata_inferred['table_name'] = filename.rsplit('.', 1)[0]
                metadata_inferred['columns'] = []
                for column in df.columns:
                    col_type = 'TEXT'
                    try:
                        if pd.api.types.is_integer_dtype(df[column].dropna()):
                             if (df[column].dropna() % 1 == 0).all():
                                col_type = 'INTEGER'
                             else: # Should not happen if is_integer_dtype is true
                                col_type = 'REAL' # Fallback
                        elif pd.api.types.is_float_dtype(df[column].dropna()):
                            col_type = 'REAL'
                        elif pd.api.types.is_numeric_dtype(df[column].dropna()): # General numeric check
                            # Check if it can be integer after dropping NA
                            temp_series = df[column].dropna()
                            if (temp_series % 1 == 0).all():
                                col_type = 'INTEGER'
                            else:
                                col_type = 'REAL'
                        # else: keep as TEXT
                    except Exception: # Broad exception for various conversion/type issues
                        pass # Keep as TEXT
                    metadata_inferred['columns'].append({'name': column, 'type': col_type})
                
                current_metadata = metadata_inferred # Store inferred metadata
                return jsonify({'message': 'File uploaded successfully', 'metadata': metadata_inferred}), 200
            except Exception as e:
                return jsonify({'error': f'Error processing CSV file: {str(e)}'}), 500

        elif filename.rsplit('.', 1)[1].lower() == 'parquet':
            try:
                pq_schema = pd.io.parquet.read_schema(filepath)
                metadata_inferred['table_name'] = filename.rsplit('.', 1)[0]
                metadata_inferred['columns'] = []

                for i in range(len(pq_schema)):
                    field = pq_schema.field(i)
                    col_name = field.name
                    arrow_type = field.type
                    col_type = 'TEXT' # Default

                    if pa.types.is_integer(arrow_type):
                        col_type = 'INTEGER'
                    elif pa.types.is_floating(arrow_type):
                        col_type = 'REAL'
                    elif pa.types.is_boolean(arrow_type):
                        col_type = 'BOOLEAN'
                    elif pa.types.is_temporal(arrow_type): # Covers date, time, timestamp
                        col_type = 'DATETIME'
                    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type) or pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
                        col_type = 'TEXT'
                    # Add more specific pyarrow types if needed, otherwise default to TEXT

                    metadata_inferred['columns'].append({'name': col_name, 'type': col_type})

                current_metadata = metadata_inferred
                return jsonify({'message': 'File uploaded successfully (Parquet)', 'metadata': metadata_inferred}), 200
            except Exception as e:
                return jsonify({'error': f'Error processing Parquet file: {str(e)}'}), 500
        
        elif filename.rsplit('.', 1)[1].lower() in ['rdata', 'rda']:
            try:
                # R script to extract metadata
                r_script = f"""
                tryCatch({{
                    load('{filepath_r}') # Use R-compatible path
                    obj_name <- ls()[1] # Assume first object is the target
                    data_obj <- get(obj_name)

                    if (is.data.frame(data_obj) || inherits(data_obj, "data.table")) {{
                        cols <- colnames(data_obj)
                        types <- sapply(data_obj, class)

                        # Prepare types for JSON, handling cases where a column might have multiple classes
                        formatted_types <- lapply(types, function(t) {{
                            if (is.array(t) || is.list(t)) {{
                                return(paste(t, collapse=", ")) # Join multiple classes if any
                            }} else {{
                                return(t)
                            }}
                        }})

                        metadata_json <- jsonlite::toJSON(list(
                            object_name = obj_name,
                            columns = mapply(function(n, t) list(name=n, type=t), cols, formatted_types, SIMPLIFY=FALSE)
                        ), auto_unbox = TRUE)
                        cat(metadata_json)
                    }} else {{
                        stop("First object in Rdata is not a data.frame or data.table.")
                    }}
                }}, error = function(e) {{
                    write(paste("R script error:", e$message), stderr())
                    stop(e) # Ensure R script exits with error status
                }})
                """

                filepath_r = filepath.replace('\\', '/') # Ensure forward slashes for R
                process = subprocess.run(['Rscript', '-', filepath_r], input=r_script.format(filepath_r=filepath_r), text=True, capture_output=True, check=False)

                if process.returncode == 0 and process.stdout:
                    r_metadata = json.loads(process.stdout)

                    # Map R types to SQL-like types
                    type_mapping = {{
                        'integer': 'INTEGER',
                        'numeric': 'REAL',
                        'character': 'TEXT',
                        'factor': 'TEXT', # Factors are often treated as text
                        'logical': 'BOOLEAN', # Or TEXT, depending on preference
                        'Date': 'DATE',
                        'POSIXct': 'TIMESTAMP', # Common for datetime
                        'POSIXlt': 'TIMESTAMP'
                        # Add more mappings as needed
                    }}

                    metadata_inferred = {{
                        'table_name': r_metadata.get('object_name', filename.rsplit('.', 1)[0]),
                        'columns': []
                    }}
                    for col in r_metadata.get('columns', []):
                        # Handle multi-class types from R (e.g., "ordered, factor")
                        r_type_primary = col.get('type', 'character').split(',')[0].strip()
                        sql_type = type_mapping.get(r_type_primary, 'TEXT') # Default to TEXT
                        metadata_inferred['columns'].append({'name': col['name'], 'type': sql_type})

                    current_metadata = metadata_inferred
                    return jsonify({{
                        'message': 'RData processed successfully. Metadata extracted.',
                        'metadata': current_metadata
                    }}), 200
                else:
                    error_message = f"Error executing R script: {process.stderr}" if process.stderr else "R script execution failed with no specific error message."
                    if process.returncode != 0:
                         error_message += f" (Return code: {process.returncode})"
                    print(f"R script stderr: {process.stderr}") # Log R errors to server console
                    current_metadata = {{'table_name': filename.rsplit('.',1)[0], 'columns': [], 'message': f'Failed to process R data file: {error_message}'}}
                    return jsonify({'error': error_message, 'metadata': current_metadata}), 500

            except json.JSONDecodeError:
                current_metadata = {{'table_name': filename.rsplit('.',1)[0], 'columns': [], 'message': 'Error parsing metadata from R script output.'}}
                return jsonify({'error': 'Error parsing R script output.', 'metadata': current_metadata}), 500
            except FileNotFoundError: # Rscript not found
                 current_metadata = {{'table_name': filename.rsplit('.',1)[0], 'columns': [], 'message': 'Rscript command not found. Please ensure R is installed and in PATH.'}}
                 return jsonify({'error': 'Rscript not found. Cannot process R data files.', 'metadata': current_metadata}), 500
            except Exception as e:
                current_metadata = {{'table_name': filename.rsplit('.',1)[0], 'columns': [], 'message': f'An unexpected error occurred: {str(e)}'}}
                return jsonify({'error': f'Error processing R data file: {str(e)}', 'metadata': current_metadata}), 500

        # For other file types, metadata remains None or minimal
        current_metadata = {'table_name': filename.rsplit('.',1)[0], 'columns': [], 'message': 'Metadata not automatically inferred for this file type. Please provide or approve manually.'}
        return jsonify({'message': 'File uploaded successfully. Metadata may need manual input.', 'metadata': current_metadata }), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/query', methods=['POST'])
def handle_query():
    global current_metadata, current_uploaded_filepath, current_uploaded_filename, last_successful_df
    
    if not (OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT):
        return jsonify({'error': 'OpenAI API not configured on the server.', 'executed_query_text': None, 'results': None, 'natural_language_response': None}), 500

    data = request.get_json()
    natural_language_query = data.get('naturalLanguageQuery')
    agent_type = data.get('agent_type', 'sql') # Default to 'sql'

    # Use metadata passed from frontend if available, otherwise use globally stored one
    metadata_for_prompt = data.get('metadata', current_metadata) 

    # Initialize variables
    sql_query = ""
    r_code_generated = ""
    python_code_generated = ""
    executed_query_text = ""
    results_df = None
    error_message = None
    results_json = []
    nl_summary = "Could not generate natural language summary." # Default

    if not natural_language_query:
        return jsonify({'error': 'No natural language query provided.', 'executed_query_text': None, 'results': None, 'natural_language_response': None}), 400
    
    if not current_uploaded_filepath:
         return jsonify({'error': 'No file has been uploaded yet or file context lost.', 'executed_query_text': None, 'results': None, 'natural_language_response': None}), 400

    # Determine table name for the query (common for both SQL and R)
    # For R, this is the object name within the Rdata file.
    table_name_for_query = metadata_for_prompt.get('table_name') if metadata_for_prompt else None
    if not table_name_for_query and current_uploaded_filename: # Fallback to filename if not in metadata
        table_name_for_query = current_uploaded_filename.rsplit('.', 1)[0]
    
    if not table_name_for_query:
        return jsonify({'error': 'Could not determine table name for query.', 'executed_query_text': None, 'results': None, 'natural_language_response': None}), 400

    # Prerequisite check for column metadata (common for SQL, R, and Python Pandas, unless SQLite for SQL)
    is_sqlite = current_uploaded_filename.rsplit('.', 1)[1].lower() == 'sqlite'
    # Python Pandas and R DataTable always need column metadata. SQL needs it if not SQLite.
    if agent_type == 'python_pandas' or agent_type == 'r_datatable' or (agent_type == 'sql' and not is_sqlite):
        if not metadata_for_prompt or not metadata_for_prompt.get('columns'):
            return jsonify({'error': 'Column metadata is missing or not provided for the selected agent. Please ensure metadata is processed or provided.', 'executed_query_text': None, 'results': None, 'natural_language_response': None}), 400

    try:
        deployment_name = os.environ.get("AZURE_DEPLOYMENT_NAME")
        if openai.api_type == "azure" and not deployment_name:
            return jsonify({'error': 'AZURE_DEPLOYMENT_NAME environment variable not set for Azure OpenAI.', 'executed_query_text': None, 'results': None, 'natural_language_response': None}), 500

        if agent_type == 'sql':
            # --- SQL Agent Logic ---
            prompt_parts = [
                "Given the table schema below and the user question, generate a valid SQL query to answer the question."
            ]
            prompt_parts.append(f"Table Name: {table_name_for_query}")

            if metadata_for_prompt and metadata_for_prompt.get('columns'):
                prompt_parts.append("Columns:")
                for column in metadata_for_prompt['columns']:
                    prompt_parts.append(f"- {column['name']} ({column['type']})")
            elif not is_sqlite : # Only append if not SQLite and no columns given (SQLite schema is in file)
                prompt_parts.append("Columns: (Schema not fully provided. Ensure your query references correct table and column names based on the file's actual schema.)")


            prompt_parts.append(f"\nUser Question: {natural_language_query}")
            prompt_parts.append("SQL Query:")
            current_prompt = "\n".join(prompt_parts)

            if openai.api_type == "azure":
                response = openai.Completion.create(engine=deployment_name, prompt=current_prompt, max_tokens=150, temperature=0.1)
            else:
                response = openai.Completion.create(model="text-davinci-003", prompt=current_prompt, max_tokens=150, temperature=0.1)
            
            sql_query = response.choices[0].text.strip()
            executed_query_text = sql_query
            if not sql_query:
                error_message = 'LLM did not return a SQL query.'
                # Return early as there's nothing to execute or correct
                return jsonify({'error': error_message, 'executed_query_text': executed_query_text, 'results': None, 'natural_language_response': None}), 500
            
            results_df, error_message = execute_duckdb_query(sql_query, current_uploaded_filepath, table_name_for_query)

            if error_message: # SQL execution failed, try to correct
                print(f"Initial SQL query failed: {sql_query}. Error: {error_message}. Attempting correction...")
                reflection_prompt_parts = [
                    "The following SQL query resulted in an error. Please correct it.",
                    f"Original Question: {natural_language_query}",
                    f"Table Name: {table_name_for_query}"
                ]
                if metadata_for_prompt and metadata_for_prompt.get('columns'):
                    reflection_prompt_parts.append("Columns:")
                    for column in metadata_for_prompt['columns']:
                        reflection_prompt_parts.append(f"- {column['name']} ({column['type']})")
                elif not is_sqlite:
                     reflection_prompt_parts.append("Columns: (Schema not fully provided or embedded)")

                reflection_prompt_parts.extend([
                    f"Failed SQL: {sql_query}",
                    f"Error Message: {error_message}",
                    "Corrected SQL Query:"
                ])
                correction_prompt = "\n".join(reflection_prompt_parts)

                if openai.api_type == "azure":
                    correction_response = openai.Completion.create(engine=deployment_name, prompt=correction_prompt, max_tokens=150, temperature=0.15)
                else:
                    correction_response = openai.Completion.create(model="text-davinci-003", prompt=correction_prompt, max_tokens=150, temperature=0.15)

                corrected_sql_query = correction_response.choices[0].text.strip()
                if not corrected_sql_query:
                    # Stick with the original error message if LLM gives up
                    error_message = f'LLM did not return a corrected SQL query. Original error: {error_message}'
                    return jsonify({'error': error_message, 'executed_query_text': executed_query_text, 'results': None, 'natural_language_response': None}), 500

                sql_query = corrected_sql_query
                executed_query_text = sql_query
                print(f"Attempting corrected SQL query: {sql_query}")
                results_df, error_message = execute_duckdb_query(sql_query, current_uploaded_filepath, table_name_for_query)

            # If error_message is still present, it will be handled before summarization

        elif agent_type == 'r_datatable':
            # --- R data.table Agent Logic ---
            # Prerequisite check (already done for columns, table_name_for_query is R object name)

            r_prompt_parts = [
                "You are an R programming assistant. Generate R code using the `data.table` package to answer the user's question.",
                "The data is loaded into an R object named `active_df` which is already a data.table.",
                f"R Object (data.table) Name: active_df (derived from: {table_name_for_query})",
                "Columns in `active_df`:"
            ]
            for column in metadata_for_prompt['columns']: # This check is now done above
                r_prompt_parts.append(f"- {column['name']} (type: {column['type']})") # Type info might help LLM

            r_prompt_parts.extend([
                f"\nUser Question: {natural_language_query}",
                "Generate only the R `data.table` code that performs the query on `active_df` and assigns the result back to `active_df`.",
                "For example: active_df <- active_df[some_condition, .(new_col = sum(another_col))]",
                "R data.table Code:"
            ])
            current_r_prompt = "\n".join(r_prompt_parts)

            if openai.api_type == "azure":
                response = openai.Completion.create(engine=deployment_name, prompt=current_r_prompt, max_tokens=200, temperature=0.1)
            else:
                response = openai.Completion.create(model="text-davinci-003", prompt=current_r_prompt, max_tokens=200, temperature=0.1)

            r_code_generated = response.choices[0].text.strip()
            executed_query_text = r_code_generated
            if not r_code_generated:
                error_message = 'LLM did not return R code.'
                return jsonify({'error': error_message, 'executed_query_text': executed_query_text, 'results': None, 'natural_language_response': None}), 500

            results_df, error_message = execute_r_script(r_code_generated, current_uploaded_filepath, table_name_for_query)

            if error_message: # R execution failed, try to correct
                print(f"Initial R code failed: {r_code_generated}. Error: {error_message}. Attempting correction...")
                r_correction_prompt_parts = [
                    "The following R data.table code resulted in an error. Please correct it.",
                    "The data is in a data.table named `active_df`.",
                    f"Original Question: {natural_language_query}",
                    "Columns in `active_df`:"
                ]
                for column in metadata_for_prompt['columns']:
                    r_correction_prompt_parts.append(f"- {column['name']} (type: {column['type']})")

                r_correction_prompt_parts.extend([
                    f"Failed R Code:\n{r_code_generated}",
                    f"Error Message: {error_message}",
                    "Corrected R data.table Code (assign result back to active_df):"
                ])
                r_correction_prompt = "\n".join(r_correction_prompt_parts)

                if openai.api_type == "azure":
                    r_correction_response = openai.Completion.create(engine=deployment_name, prompt=r_correction_prompt, max_tokens=250, temperature=0.15)
                else:
                    r_correction_response = openai.Completion.create(model="text-davinci-003", prompt=r_correction_prompt, max_tokens=250, temperature=0.15)
                
                corrected_r_code = r_correction_response.choices[0].text.strip()
                if not corrected_r_code:
                    error_message = f'LLM did not return corrected R code. Original error: {error_message}'
                    return jsonify({'error': error_message, 'executed_query_text': executed_query_text, 'results': None, 'natural_language_response': None}), 500

                r_code_generated = corrected_r_code
                executed_query_text = r_code_generated
                print(f"Attempting corrected R code: {r_code_generated}")
                results_df, error_message = execute_r_script(r_code_generated, current_uploaded_filepath, table_name_for_query)

        elif agent_type == 'python_pandas':
            # --- Python Pandas Agent Logic ---
            # Prerequisite checks already done (current_uploaded_filepath, table_name_for_query, metadata_for_prompt['columns'])

            pandas_prompt_parts = [
                "You are a Python programming assistant. Generate Python code using the Pandas library to answer the user's question.",
                f"The data is loaded into a Pandas DataFrame named `{table_name_for_query}`.",
                "Columns in the DataFrame:"
            ]
            for column in metadata_for_prompt['columns']:
                pandas_prompt_parts.append(f"- {column['name']} (type: {column['type']})")

            pandas_prompt_parts.extend([
                f"\nUser Question: {natural_language_query}",
                f"Generate only the Python Pandas code that performs the query on the DataFrame named `{table_name_for_query}` and assigns the result back to the same DataFrame variable.",
                f"For example: {table_name_for_query} = {table_name_for_query}[{table_name_for_query}['some_column'] > 10]",
                "Python Pandas Code:"
            ])
            current_pandas_prompt = "\n".join(pandas_prompt_parts)

            if openai.api_type == "azure":
                response = openai.Completion.create(engine=deployment_name, prompt=current_pandas_prompt, max_tokens=300, temperature=0.1)
            else:
                response = openai.Completion.create(model="text-davinci-003", prompt=current_pandas_prompt, max_tokens=300, temperature=0.1)

            python_code_generated = response.choices[0].text.strip()
            executed_query_text = python_code_generated

            if not python_code_generated:
                error_message = 'LLM did not return Python Pandas code.'
                return jsonify({'error': error_message, 'executed_query_text': executed_query_text, 'results': None, 'natural_language_response': None}), 500

            results_df, error_message = execute_python_pandas_code(python_code_generated, current_uploaded_filepath, dataframe_name=table_name_for_query)

            if error_message: # Python code execution failed, try to correct
                print(f"Initial Python Pandas code failed: {python_code_generated}. Error: {error_message}. Attempting correction...")
                pandas_correction_prompt_parts = [
                    "The following Python Pandas code resulted in an error. Please correct it.",
                    f"The data is in a Pandas DataFrame named `{table_name_for_query}`.",
                    f"Original Question: {natural_language_query}",
                    "Columns in DataFrame:"
                ]
                for column in metadata_for_prompt['columns']:
                    pandas_correction_prompt_parts.append(f"- {column['name']} (type: {column['type']})")

                pandas_correction_prompt_parts.extend([
                    f"Failed Python Code:\n{python_code_generated}",
                    f"Error Message: {error_message}",
                    f"Corrected Python Pandas Code (assign result back to `{table_name_for_query}`):"
                ])
                pandas_correction_prompt = "\n".join(pandas_correction_prompt_parts)

                if openai.api_type == "azure":
                    correction_response = openai.Completion.create(engine=deployment_name, prompt=pandas_correction_prompt, max_tokens=350, temperature=0.15)
                else:
                    correction_response = openai.Completion.create(model="text-davinci-003", prompt=pandas_correction_prompt, max_tokens=350, temperature=0.15)

                corrected_python_code = correction_response.choices[0].text.strip()
                if not corrected_python_code:
                    error_message = f'LLM did not return corrected Python Pandas code. Original error: {error_message}'
                    return jsonify({'error': error_message, 'executed_query_text': executed_query_text, 'results': None, 'natural_language_response': None}), 500

                python_code_generated = corrected_python_code
                executed_query_text = python_code_generated
                print(f"Attempting corrected Python Pandas code: {python_code_generated}")
                results_df, error_message = execute_python_pandas_code(python_code_generated, current_uploaded_filepath, dataframe_name=table_name_for_query)
        else:
            error_message = f"Unsupported agent_type: {agent_type}"
            # Return early as this is a fundamental configuration issue
            return jsonify({'error': error_message, 'executed_query_text': None, 'results': None, 'natural_language_response': None}), 400

        # --- Common Result Processing & Summarization ---
        if error_message:
            # This error is from the execution (or correction attempt) of SQL or R code
            return jsonify({'executed_query_text': executed_query_text, 'error': error_message, 'results': None, 'natural_language_response': None}), 400

        # If we reach here, results_df should be populated from SQL or R execution
        if results_df is not None:
            last_successful_df = results_df.copy()
            results_json = results_df.to_dict(orient='records')
        else: # Should ideally not happen if error_message was not set, but as a safeguard
            results_json = []
            last_successful_df = pd.DataFrame()


        # --- Third LLM Call to generate Natural Language Summary ---
        # This part is common, using executed_query_text and results_json
        try:
            result_summary_for_prompt = ""
            if not results_json:
                result_summary_for_prompt = "The query returned no results."
            elif len(results_json) <= 5:
                result_summary_for_prompt = f"The query returned the following results:\n{pd.DataFrame(results_json).to_string()}"
            else:
                result_summary_for_prompt = f"The query returned {len(results_json)} rows. Here are the first 5:\n{pd.DataFrame(results_json[:5]).to_string()}\n...and {len(results_json)-5} more rows."

            summary_prompt_parts = [
                f"Based on the user's question '{natural_language_query}', the executed query '{executed_query_text}', and the following query results, provide a concise natural language answer:",
                result_summary_for_prompt,
                "\nNatural Language Answer:"
            ]
            summary_prompt = "\n".join(summary_prompt_parts)

            if openai.api_type == "azure":
                summary_response = openai.Completion.create(engine=deployment_name, prompt=summary_prompt, max_tokens=200, temperature=0.3)
            else:
                summary_response = openai.Completion.create(model="text-davinci-003", prompt=summary_prompt, max_tokens=200, temperature=0.3)

            nl_summary = summary_response.choices[0].text.strip()
            if not nl_summary:
                nl_summary = "LLM did not provide a natural language summary."

        except openai.APIError as e_sum: # For openai SDK v1.x
            print(f"OpenAI API error during summary generation: {str(e_sum)}")
            # Don't overwrite data results if only summary fails
            nl_summary = f"Error generating natural language summary: {str(e_sum)}"
        except Exception as e_sum_gen:
            print(f"Unexpected error during summary generation: {str(e_sum_gen)}")
            nl_summary = f"Unexpected error generating natural language summary: {str(e_sum_gen)}"

        return jsonify({
            'executed_query_text': executed_query_text,
            'results': results_json,
            'error': None, # Explicitly None if execution was successful up to this point
            'natural_language_response': nl_summary
        }), 200

    except openai.APIError as e: # For openai SDK v1.x
        # executed_query_text might hold the last attempted query
        return jsonify({'error': f'OpenAI API error: {str(e)}', 'executed_query_text': executed_query_text, 'results': None, 'natural_language_response': None}), 500
    except Exception as e: # Catch any other unexpected errors
        return jsonify({'error': f'An unexpected error occurred: {str(e)}', 'executed_query_text': executed_query_text, 'results': None, 'natural_language_response': None}), 500

@app.route('/plot_data', methods=['POST'])
def plot_data():
    global last_successful_df
    if last_successful_df is None or last_successful_df.empty:
        return jsonify({'error': 'No data available to plot. Please execute a query first.'}), 400

    df_to_plot = last_successful_df.copy()
    
    # Basic plotting logic
    plt.figure(figsize=(8, 6)) # Create a new figure for each plot
    
    try:
        if len(df_to_plot.columns) == 1 and pd.api.types.is_numeric_dtype(df_to_plot.iloc[:, 0]):
            # Histogram for single numerical column
            df_to_plot.iloc[:, 0].plot(kind='hist', bins=20)
            plt.title(f'Histogram of {df_to_plot.columns[0]}')
            plt.xlabel(df_to_plot.columns[0])
            plt.ylabel('Frequency')
        elif len(df_to_plot.columns) >= 2:
            # Attempt to find suitable columns for a bar or line chart
            numeric_cols = df_to_plot.select_dtypes(include=pd.np.number).columns.tolist()
            categorical_cols = df_to_plot.select_dtypes(include='object').columns.tolist() # Basic categorical check
            
            if not numeric_cols:
                 return jsonify({'error': 'No numeric columns found for plotting.'}), 400

            y_col = numeric_cols[0] # Take the first numeric column as Y-axis
            
            if categorical_cols:
                x_col = categorical_cols[0] # First categorical as X-axis for bar chart
                # Summarize data for bar chart if too many categories
                if df_to_plot[x_col].nunique() > 20:
                    top_20_categories = df_to_plot[x_col].value_counts().nlargest(20).index
                    plot_df_agg = df_to_plot[df_to_plot[x_col].isin(top_20_categories)].groupby(x_col)[y_col].sum().reset_index()
                    plot_df_agg.plot(kind='bar', x=x_col, y=y_col)
                    plt.title(f'Bar Chart: {y_col} by top 20 {x_col}')
                else:
                    df_to_plot.plot(kind='bar', x=x_col, y=y_col)
                    plt.title(f'Bar Chart: {y_col} by {x_col}')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.xticks(rotation=45, ha='right')
            elif len(numeric_cols) >=2 : # If no clear categorical, try line plot with first two numeric
                x_col = numeric_cols[1] if numeric_cols[0] == y_col and len(numeric_cols) > 1 else numeric_cols[0]
                if df_to_plot.shape[0] > 100: # Sample if too many points for line plot
                    df_to_plot.sample(100).sort_values(by=x_col).plot(kind='line', x=x_col, y=y_col)
                else:
                    df_to_plot.sort_values(by=x_col).plot(kind='line', x=x_col, y=y_col)
                plt.title(f'Line Plot: {y_col} vs {x_col}')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
            else: # Fallback to histogram of the first numeric column if other plots not suitable
                df_to_plot[y_col].plot(kind='hist', bins=20)
                plt.title(f'Histogram of {y_col}')
                plt.xlabel(y_col)
                plt.ylabel('Frequency')

        else:
            return jsonify({'error': 'Plotting logic not implemented for this data structure (e.g., no numeric columns or too few columns).'}), 400

        plt.tight_layout() # Adjust layout to prevent labels from being cut off
        
        # Save plot to memory buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close() # Close the figure to free memory

        # Encode image to base64
        base64_img = base64.b64encode(img_buffer.read()).decode('utf-8')
        return jsonify({'plot_image': f'data:image/png;base64,{base64_img}'}), 200

    except Exception as e_plot:
        plt.close() # Ensure figure is closed on error
        return jsonify({'error': f'Error generating plot: {str(e_plot)}'}), 500


@app.route('/execute_sql', methods=['POST'])
def execute_sql_query_route(): # Renamed to avoid conflict
    global current_uploaded_filepath, current_metadata, current_uploaded_filename, last_successful_df
    data = request.get_json()
    sql_query = data.get('sql_query')
    
    if not sql_query:
        return jsonify({'error': 'No SQL query provided.'}), 400
    
    if not current_uploaded_filepath:
        return jsonify({'error': 'No file has been uploaded or file context lost.'}), 400

    # Determine table name
    table_name_in_db = current_metadata.get('table_name') if current_metadata else None
    if not table_name_in_db and current_uploaded_filename: # Fallback to filename
        table_name_in_db = current_uploaded_filename.rsplit('.', 1)[0]
    
    if not table_name_in_db:
        return jsonify({'error': 'Could not determine table name for query execution.'}), 400
    
    results_df, error_message = execute_duckdb_query(sql_query, current_uploaded_filepath, table_name_in_db)

    if error_message:
        return jsonify({'error': error_message}), 400
    else:
        last_successful_df = results_df.copy() if results_df is not None else pd.DataFrame() # Cache for plotting
        results_json = results_df.to_dict(orient='records') if results_df is not None else []
        return jsonify({'results': results_json}), 200


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
