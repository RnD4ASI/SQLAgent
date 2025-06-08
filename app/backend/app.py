from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np # Import numpy
import openai
import duckdb
import matplotlib
matplotlib.use('Agg') # Use Agg backend for web server
import matplotlib.pyplot as plt
import io
import base64

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')) # Load from project root

app = Flask(__name__, template_folder='../templates', static_folder='../static')
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
ALLOWED_EXTENSIONS = {'csv', 'parquet', 'sqlite'}
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

# Global OpenAI client setup (moved here for clarity)
if AZURE_OPENAI_ENDPOINT:
    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = "2023-07-01-preview"
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
elif OPENAI_API_BASE:
    openai.api_base = OPENAI_API_BASE
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
    else:
        print("Warning: OPENAI_API_KEY not set for custom OpenAI endpoint.")
elif OPENAI_API_KEY:
     openai.api_key = OPENAI_API_KEY
else:
    print("Warning: OpenAI API key or endpoint not configured. The /query endpoint will not work.")

# Store metadata globally for simplicity in this example
current_metadata = None # Stores metadata from file upload
# last_successful_df is already defined above


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
    con = None # Initialize con to None

    try:
        if file_ext == 'sqlite':
            con = duckdb.connect(database=file_path, read_only=True)
        else:
            con = duckdb.connect(database=':memory:', read_only=False)
            if file_ext == 'csv':
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
            elif file_ext == 'parquet':
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')")
            else:
                return None, f"Unsupported file type for querying: {file_ext}"
        
        result_df = con.execute(sql_query).fetchdf()
        return result_df, None # Success
    except duckdb.Error as e:
        return None, f"DuckDB SQL execution error: {str(e)}"
    except FileNotFoundError:
        return None, f"Data file not found: {file_path}"
    except Exception as e:
        return None, f"An error occurred during SQL execution: {str(e)}"
    finally:
        if con:
            con.close()


def handle_sql_agent_query(natural_language_query, metadata_for_prompt, current_uploaded_filepath_param, table_name_for_query_param):
    """
    Handles the SQL agent logic: LLM calls for SQL generation, execution, correction, and summary.
    """
    # Initialize return values
    sql_query = ""
    results_df = None
    results_json = []
    nl_summary = ""
    error_message_to_return = None
    
    import textwrap # Ensure textwrap is imported

    # --- First LLM Call to generate SQL ---
    schema_description_parts = []
    if metadata_for_prompt and metadata_for_prompt.get('columns'):
        for column in metadata_for_prompt['columns']:
            schema_description_parts.append(f"- {column['name']} ({column['type']})")
    else:
        schema_description_parts.append("(Schema not fully provided or is embedded, e.g., in a SQLite file. Ensure your query references correct table and column names based on the file's actual schema.)")
    
    schema_description_str = "\n".join(schema_description_parts)

    current_prompt = textwrap.dedent(f"""\
        Given the table schema below and the user question, generate a valid SQL query to answer the question.
        Table Name: {table_name_for_query_param}
        Columns:
        {schema_description_str}

        User Question: {natural_language_query}
        SQL Query:""")

    try:
        deployment_name = os.environ.get("AZURE_DEPLOYMENT_NAME")

        if openai.api_type == "azure" and not deployment_name:
            error_message_to_return = 'AZURE_DEPLOYMENT_NAME environment variable not set for Azure OpenAI.'
            return {'sql_query': None, 'results_json': None, 'nl_summary': None, 'error_message': error_message_to_return, 'results_df': None}

        if openai.api_type == "azure":
            response = openai.Completion.create(engine=deployment_name, prompt=current_prompt, max_tokens=150, temperature=0.1)
        else:
            response = openai.Completion.create(model="text-davinci-003", prompt=current_prompt, max_tokens=150, temperature=0.1)
        
        sql_query = response.choices[0].text.strip()
        if not sql_query:
            error_message_to_return = 'LLM did not return a SQL query.'
            return {'sql_query': '', 'results_json': None, 'nl_summary': None, 'error_message': error_message_to_return, 'results_df': None}

        # --- Attempt to execute the first generated SQL ---
        results_df, db_error_message = execute_duckdb_query(sql_query, current_uploaded_filepath_param, table_name_for_query_param)

        if db_error_message: # SQL execution failed
            print(f"Initial SQL query failed: {sql_query}. Error: {db_error_message}. Attempting correction...")
            
            # schema_description_str is already available from the initial SQL generation part
            correction_prompt = textwrap.dedent(f"""\
                The following SQL query resulted in an error. Please correct it.
                Original Question: {natural_language_query}
                Table Name: {table_name_for_query_param}
                Columns:
                {schema_description_str}
                Failed SQL: {sql_query}
                Error Message: {db_error_message}
                Corrected SQL Query:""")

            if openai.api_type == "azure":
                correction_response = openai.Completion.create(engine=deployment_name, prompt=correction_prompt, max_tokens=150, temperature=0.15)
            else:
                correction_response = openai.Completion.create(model="text-davinci-003", prompt=correction_prompt, max_tokens=150, temperature=0.15)
            
            corrected_sql_query = correction_response.choices[0].text.strip()
            if not corrected_sql_query:
                error_message_to_return = f'LLM did not return a corrected SQL query. Original error: {db_error_message}'
                return {'sql_query': sql_query, 'results_json': None, 'nl_summary': None, 'error_message': error_message_to_return, 'results_df': None}
            
            sql_query = corrected_sql_query 
            print(f"Attempting corrected SQL query: {sql_query}")
            results_df, db_error_message = execute_duckdb_query(sql_query, current_uploaded_filepath_param, table_name_for_query_param)
        
        if db_error_message:
            error_message_to_return = f'SQL execution failed after correction attempt: {db_error_message}'
            return {'sql_query': sql_query, 'results_json': None, 'nl_summary': None, 'error_message': error_message_to_return, 'results_df': None}
        else:
            # If successful execution
            results_json = results_df.to_dict(orient='records') if results_df is not None else []

            # --- Third LLM Call to generate Natural Language Summary ---
            nl_summary = "Could not generate natural language summary." # Default
            try:
                result_summary_for_prompt = ""
                if not results_json:
                    result_summary_for_prompt = "The query returned no results."
                elif len(results_json) <= 5:
                    result_summary_for_prompt = f"The query returned the following results:\n{pd.DataFrame(results_json).to_string()}"
                else:
                    result_summary_for_prompt = f"The query returned {len(results_json)} rows. Here are the first 5:\n{pd.DataFrame(results_json[:5]).to_string()}\n...and {len(results_json)-5} more rows."

                summary_prompt = textwrap.dedent(f"""\
                    Based on the user's question '{natural_language_query}', the SQL query '{sql_query}', and the following SQL query results, provide a concise natural language answer:
                    {result_summary_for_prompt}

                    Natural Language Answer:""")

                if openai.api_type == "azure":
                    summary_response = openai.Completion.create(engine=deployment_name, prompt=summary_prompt, max_tokens=200, temperature=0.3)
                else:
                    summary_response = openai.Completion.create(model="text-davinci-003", prompt=summary_prompt, max_tokens=200, temperature=0.3)
                
                nl_summary = summary_response.choices[0].text.strip()
                if not nl_summary:
                    nl_summary = "LLM did not provide a natural language summary."

            except openai.OpenAIError as e_sum: # Updated OpenAIError
                print(f"OpenAI API error during summary generation: {str(e_sum)}")
                # Keep the results and SQL, but indicate summary failure
                nl_summary = f"Error generating natural language summary: {str(e_sum)}"
            except Exception as e_sum_gen:
                print(f"Unexpected error during summary generation: {str(e_sum_gen)}")
                nl_summary = f"Unexpected error generating natural language summary: {str(e_sum_gen)}"

            return {'sql_query': sql_query, 'results_json': results_json, 'nl_summary': nl_summary, 'error_message': None, 'results_df': results_df}

    except openai.OpenAIError as e: # Updated OpenAIError
        error_message_to_return = f'OpenAI API error: {str(e)}'
        return {'sql_query': sql_query, 'results_json': None, 'nl_summary': None, 'error_message': error_message_to_return, 'results_df': None}
    except Exception as e:
        error_message_to_return = f'An unexpected error occurred in SQL agent: {str(e)}'
        return {'sql_query': sql_query, 'results_json': None, 'nl_summary': None, 'error_message': error_message_to_return, 'results_df': None}


def generate_pandas_code_prompt(natural_language_query: str, metadata: dict) -> str:
    """
    Generates a prompt for the LLM to create pandas code.
    """
    import textwrap # Ensure textwrap is imported

    column_description_parts = []
    if metadata and metadata.get('columns'):
        for column in metadata['columns']:
            column_description_parts.append(f"- {column['name']} ({column['type']})")
    else:
        column_description_parts.append("(Column information not fully available. Generate code assuming 'df' is the DataFrame.)")

    columns_description_str = "\n".join(column_description_parts)

    prompt = textwrap.dedent(f"""\
        You are a helpful assistant that generates Python pandas code.
        Given a pandas DataFrame named 'df' with the following columns:
        {columns_description_str}

        Please generate a Python pandas code snippet (do not include any explanation or markdown) to perform the following task based on the user's question.
        The final result should be assigned to a variable named 'result_df'.

        User Question: "{natural_language_query}"
        Pandas Code:""")
    return prompt

def execute_pandas_code(code_string: str, input_df: pd.DataFrame):
    """
    Executes a string of pandas code on a given DataFrame.
    The code is expected to assign its result to a variable 'result_df'.
    """
    if not isinstance(input_df, pd.DataFrame):
        return None, "Invalid input: input_df must be a pandas DataFrame."

    local_scope = {}
    # Pass a copy of the DataFrame to avoid modification by reference if code alters 'df'
    # globals for exec are restricted to only what's necessary.
    exec_globals = {'df': input_df.copy(), 'pd': pd}

    try:
        exec(code_string, exec_globals, local_scope)

        if 'result_df' in local_scope:
            result_df_val = local_scope['result_df']
            if not isinstance(result_df_val, pd.DataFrame):
                 # Attempt to convert if it's a Series (common in pandas operations)
                if isinstance(result_df_val, pd.Series):
                    result_df_val = result_df_val.to_frame()
                else:
                    return None, "Result ('result_df') is not a pandas DataFrame or Series."
            return result_df_val, None
        else:
            return None, "Pandas code did not assign its result to 'result_df'."

    except SyntaxError as e:
        return None, f"Pandas code SyntaxError: {e}"
    except NameError as e:
        return None, f"Pandas code NameError: {e} (Possibly refers to an undefined variable or column)"
    except KeyError as e:
        return None, f"Pandas code KeyError: {e} (Likely an invalid column name)"
    except AttributeError as e:
        return None, f"Pandas code AttributeError: {e} (Possibly an issue with DataFrame operations)"
    except Exception as e:
        return None, f"An unexpected error occurred during pandas code execution: {type(e).__name__} - {e}"


def handle_pandas_agent_query(natural_language_query: str, metadata_for_prompt: dict, current_uploaded_filepath_param: str, table_name_for_query_param: str):
    """
    Handles the Pandas agent logic: data loading, LLM call for code generation, and code execution.
    """
    global last_successful_df # To update the global cache
    generated_code = ""
    results_json = None
    nl_summary = "Pandas agent processing."
    error_message_to_return = None
    input_df = None

    # --- 1. Load Data ---
    if not current_uploaded_filepath_param:
        error_message_to_return = "No file path available for pandas agent."
        # This return structure matches what /query expects
        return {'generated_code': None, 'results_json': None, 'nl_summary': "Error: Missing file path.", 'error_message': error_message_to_return}

    try:
        file_ext = current_uploaded_filepath_param.rsplit('.', 1)[1].lower()
        if file_ext == 'csv':
            input_df = pd.read_csv(current_uploaded_filepath_param)
        elif file_ext == 'parquet':
            input_df = pd.read_parquet(current_uploaded_filepath_param)
        elif file_ext == 'sqlite':
            import sqlite3 # Local import for this specific use case
            conn = sqlite3.connect(current_uploaded_filepath_param)
            # Use table_name_for_query_param, which should be derived from metadata or filename
            if not table_name_for_query_param:
                 error_message_to_return = "Table name for SQLite database not specified."
                 return {'generated_code': None, 'results_json': None, 'nl_summary': "Error: Missing table name for SQLite.", 'error_message': error_message_to_return}
            input_df = pd.read_sql_query(f"SELECT * FROM \"{table_name_for_query_param}\"", conn) # Ensure table name is quoted
            conn.close()
        else:
            error_message_to_return = f"Unsupported file type for pandas agent: {file_ext}"
            return {'generated_code': None, 'results_json': None, 'nl_summary': f"Error: Unsupported file type {file_ext}.", 'error_message': error_message_to_return}
    except Exception as e:
        error_message_to_return = f"Error loading data into DataFrame: {str(e)}"
        return {'generated_code': None, 'results_json': None, 'nl_summary': "Error during data loading.", 'error_message': error_message_to_return}

    # --- 2. Generate Pandas Code (LLM Call) ---
    try:
        prompt = generate_pandas_code_prompt(natural_language_query, metadata_for_prompt)
        deployment_name = os.environ.get("AZURE_DEPLOYMENT_NAME")

        if openai.api_type == "azure" and not deployment_name:
            error_message_to_return = 'AZURE_DEPLOYMENT_NAME environment variable not set for Azure OpenAI.'
        elif not (OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT or openai.api_key): # Check if API is configured
             error_message_to_return = 'OpenAI API not configured.'

        if error_message_to_return: # If config error from above
             return {'generated_code': None, 'results_json': None, 'nl_summary': "OpenAI API configuration error.", 'error_message': error_message_to_return}

        if openai.api_type == "azure":
            response = openai.Completion.create(engine=deployment_name, prompt=prompt, max_tokens=250, temperature=0.1)
        else:
            response = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=250, temperature=0.1)

        generated_code = response.choices[0].text.strip()
        if not generated_code:
            error_message_to_return = 'LLM did not return any pandas code.'
            nl_summary = "LLM failed to generate pandas code."
        else:
            nl_summary = "Pandas code generated. Attempting execution."

    except openai.OpenAIError as e:
        error_message_to_return = f'OpenAI API error during pandas code generation: {str(e)}'
        nl_summary = "Error during pandas code generation."
    except Exception as e: # Catch other unexpected errors during LLM phase
        error_message_to_return = f'An unexpected error occurred during pandas code generation: {str(e)}'
        nl_summary = "Unexpected error during pandas code generation."

    if error_message_to_return: # If LLM call failed
        return {'generated_code': generated_code, 'results_json': None, 'nl_summary': nl_summary, 'error_message': error_message_to_return}

    # --- 3. Execute Pandas Code ---
    if input_df is not None and generated_code:
        result_df, exec_error = execute_pandas_code(generated_code, input_df)
        if exec_error:
            error_message_to_return = exec_error
            nl_summary = "Error executing pandas code."
            results_json = None # Ensure no results if execution fails
        else:
            if result_df is not None:
                results_json = result_df.to_dict(orient='records')
                last_successful_df = result_df.copy()

                # --- 4. Generate Natural Language Summary for Pandas Result ---
                try:
                    result_summary_for_prompt = ""
                    if result_df.empty:
                        result_summary_for_prompt = "The pandas code returned an empty DataFrame."
                    elif len(result_df) <= 5:
                        result_summary_for_prompt = f"The pandas code returned the following DataFrame:\n{result_df.to_string()}"
                    else:
                        result_summary_for_prompt = f"The pandas code returned a DataFrame with {len(result_df)} rows. Here are the first 5:\n{result_df.head().to_string()}\n...and {len(result_df)-5} more rows."

                    import textwrap # Ensure textwrap is imported for pandas summary
                    summary_prompt = textwrap.dedent(f"""\
                        Based on the user's question '{natural_language_query}',
                        the executed pandas code:
                        ```python
                        {generated_code}
                        ```
                        and the following results:
                        {result_summary_for_prompt}

                        Provide a concise natural language answer:
                        Natural Language Answer:""")

                    deployment_name = os.environ.get("AZURE_DEPLOYMENT_NAME") # Re-check for safety, though should be set if code gen worked
                    if openai.api_type == "azure" and not deployment_name:
                        nl_summary = "Successfully executed pandas code, but Azure deployment name not found for summary."
                    elif not (OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT or openai.api_key):
                        nl_summary = "Successfully executed pandas code, but OpenAI API not configured for summary."
                    else:
                        if openai.api_type == "azure":
                            summary_response = openai.Completion.create(engine=deployment_name, prompt=summary_prompt, max_tokens=200, temperature=0.3)
                        else:
                            summary_response = openai.Completion.create(model="text-davinci-003", prompt=summary_prompt, max_tokens=200, temperature=0.3)

                        nl_summary = summary_response.choices[0].text.strip()
                        if not nl_summary:
                            nl_summary = "Successfully executed pandas code, but LLM did not provide a summary."
                except openai.OpenAIError as e_sum:
                    print(f"OpenAI API error during pandas summary generation: {str(e_sum)}")
                    nl_summary = f"Successfully executed pandas code, but error generating summary: {str(e_sum)}"
                except Exception as e_sum_gen:
                    print(f"Unexpected error during pandas summary generation: {str(e_sum_gen)}")
                    nl_summary = f"Successfully executed pandas code, but unexpected error generating summary: {str(e_sum_gen)}"

            else: # result_df is None (should be caught by exec_error but defensive)
                results_json = []
                last_successful_df = pd.DataFrame()
                nl_summary = "Pandas code executed, but no DataFrame was returned to summarize."
    else:
        if not generated_code and not error_message_to_return: # If no code and no prior error
             nl_summary = "No pandas code was generated to execute."
        # If error_message_to_return is already set, nl_summary would have been set accordingly.
        # error_message_to_return might already be set if input_df loading failed.

    return {
        'generated_code': generated_code,
        'results_json': results_json,
        'nl_summary': nl_summary,
        'error_message': error_message_to_return
    }


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
        # Ensure UPLOAD_FOLDER exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_uploaded_filename = filename
        current_uploaded_filepath = filepath

        current_metadata = None # Reset metadata
        metadata_inferred = {}
        if filename.rsplit('.', 1)[1].lower() == 'csv':
            try:
                df = pd.read_csv(filepath, nrows=100)
                metadata_inferred['table_name'] = filename.rsplit('.', 1)[0]
                metadata_inferred['columns'] = []
                for column in df.columns:
                    col_type = 'TEXT' # Default
                    try:
                        # Attempt to infer more specific types
                        temp_series_dropna = df[column].dropna()
                        if temp_series_dropna.empty: # If all are NA, keep TEXT
                            pass
                        elif pd.api.types.is_integer_dtype(temp_series_dropna):
                             if (temp_series_dropna % 1 == 0).all(): # Check if actually integers
                                col_type = 'INTEGER'
                             else: # Should not happen if is_integer_dtype is true and all are integers
                                col_type = 'REAL'
                        elif pd.api.types.is_float_dtype(temp_series_dropna):
                            col_type = 'REAL'
                        elif pd.api.types.is_numeric_dtype(temp_series_dropna):
                            if (temp_series_dropna % 1 == 0).all():
                                col_type = 'INTEGER'
                            else:
                                col_type = 'REAL'
                        # Other types (datetime, boolean) could be added here
                    except Exception:
                        pass # Keep as TEXT if any error during type inference
                    metadata_inferred['columns'].append({'name': column, 'type': col_type})

                current_metadata = metadata_inferred
                return jsonify({'message': 'File uploaded successfully', 'metadata': metadata_inferred}), 200
            except Exception as e:
                return jsonify({'error': f'Error processing CSV file: {str(e)}'}), 500

        # For other file types (Parquet, SQLite), provide a basic metadata structure.
        # More sophisticated inference might be needed for these in a real app.
        table_name_default = filename.rsplit('.',1)[0]
        current_metadata = {
            'table_name': table_name_default,
            'columns': [], # For Parquet/SQLite, columns are often best read from the file directly during query.
            'message': 'Metadata for this file type is minimal. Schema will be inferred by DuckDB during query if possible.'
        }
        if filename.rsplit('.', 1)[1].lower() == 'sqlite':
             current_metadata['message'] = f"SQLite file uploaded. Query against table names within the file (e.g., '{table_name_default}' if it exists, or other tables)."

        return jsonify({'message': 'File uploaded successfully. Metadata may need manual input or is inferred at query time.', 'metadata': current_metadata }), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/query', methods=['POST'])
def handle_query():
    global current_metadata, current_uploaded_filepath, current_uploaded_filename, last_successful_df

    if not (OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT or openai.api_key): # Check if API key is set
        return jsonify({'error': 'OpenAI API not configured on the server.'}), 500

    data = request.get_json()
    natural_language_query = data.get('naturalLanguageQuery')
    metadata_from_request = data.get('metadata', current_metadata)
    agent_type = data.get('agent_type', 'sql')

    if not natural_language_query:
        return jsonify({'error': 'No natural language query provided.'}), 400

    if not current_uploaded_filepath:
         return jsonify({'error': 'No file has been uploaded yet or file context lost.'}), 400

    table_name_for_query = metadata_from_request.get('table_name') if metadata_from_request else None
    if not table_name_for_query and current_uploaded_filename:
        table_name_for_query = current_uploaded_filename.rsplit('.', 1)[0]

    if not table_name_for_query: # Still needed for context
        return jsonify({'error': 'Could not determine table name for context.'}), 400

    # Specific metadata checks for SQL agent
    if agent_type == 'sql' and current_uploaded_filename.rsplit('.', 1)[1].lower() not in ['sqlite']:
        if not metadata_from_request or not metadata_from_request.get('columns'):
            if current_uploaded_filename.rsplit('.', 1)[1].lower() == 'csv':
                try:
                    df_temp = pd.read_csv(current_uploaded_filepath, nrows=5)
                    cols = [{'name': col, 'type': 'TEXT'} for col in df_temp.columns]
                    metadata_from_request = {'table_name': table_name_for_query, 'columns': cols}
                    current_metadata = metadata_from_request
                    print("Warning: Column metadata was missing for SQL agent; basic re-inference attempted.")
                except Exception as e:
                    print(f"Error during re-inference for SQL agent: {e}")
                    return jsonify({'error': 'Column metadata is missing for SQL agent.'}), 400
            else:
                 return jsonify({'error': 'Column metadata is missing for SQL agent.'}), 400

    if agent_type == 'sql':
        sql_agent_result = handle_sql_agent_query(
            natural_language_query,
            metadata_from_request,
            current_uploaded_filepath,
            table_name_for_query
        )

        if sql_agent_result['error_message']:
            return jsonify({
                'code_type': 'sql',
                'sql_query': sql_agent_result['sql_query'],
                'results': None,
                'error': sql_agent_result['error_message'],
                'natural_language_response': None
            }), 400 if "SQL execution failed" in sql_agent_result['error_message'] else 500

        if sql_agent_result['results_df'] is not None:
            last_successful_df = sql_agent_result['results_df'].copy()
        else:
            last_successful_df = pd.DataFrame()

        return jsonify({
            'code_type': 'sql',
            'sql_query': sql_agent_result['sql_query'],
            'results': sql_agent_result['results_json'],
            'error': None,
            'natural_language_response': sql_agent_result['nl_summary']
        }), 200

    elif agent_type == 'pandas':
        pandas_agent_result = handle_pandas_agent_query(
            natural_language_query,
            metadata_from_request,
            current_uploaded_filepath,
            table_name_for_query
        )

        if pandas_agent_result['error_message']:
            # If LLM fails or other error in pandas handler, return 500
            return jsonify({
                'code_type': 'pandas',
                'generated_code': pandas_agent_result['generated_code'],
                'results': None, # No results if error
                'error': pandas_agent_result['error_message'],
                'natural_language_response': pandas_agent_result['nl_summary'] # Use summary from pandas_agent_result
            }), 500

        # last_successful_df is not updated by pandas agent yet as execution is not implemented
        # This part is for successful code generation by the pandas agent
        return jsonify({
            'code_type': 'pandas',
            'generated_code': pandas_agent_result['generated_code'],
            'results': pandas_agent_result['results_json'], # Will be None for now
            'error': None, # No error if code generation was successful
            'natural_language_response': pandas_agent_result['nl_summary']
        }), 200

    else:
        return jsonify({'error': f"Unknown agent type: {agent_type}"}), 400


@app.route('/plot_data', methods=['POST'])
def plot_data():
    global last_successful_df
    if last_successful_df is None or last_successful_df.empty:
        return jsonify({'error': 'No data available to plot. Please execute a query first.'}), 400

    df_to_plot = last_successful_df.copy()
    fig = None # Initialize fig to None for finally block

    try:
        fig, ax = plt.subplots(figsize=(8, 6)) # Create a figure and an axes.

        if len(df_to_plot.columns) == 1 and pd.api.types.is_numeric_dtype(df_to_plot.iloc[:, 0]):
            df_to_plot.iloc[:, 0].plot(kind='hist', bins=20, ax=ax)
            ax.set_title(f'Histogram of {df_to_plot.columns[0]}')
            ax.set_xlabel(df_to_plot.columns[0])
            ax.set_ylabel('Frequency')
        elif len(df_to_plot.columns) >= 2:
            # Using numpy for numeric type check
            numeric_cols = df_to_plot.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df_to_plot.select_dtypes(include='object').columns.tolist()
            
            if not numeric_cols:
                 return jsonify({'error': 'No numeric columns found for plotting.'}), 400

            y_col = numeric_cols[0]
            
            if categorical_cols:
                x_col = categorical_cols[0]
                if df_to_plot[x_col].nunique() > 20:
                    top_20_categories = df_to_plot[x_col].value_counts().nlargest(20).index
                    plot_df_agg = df_to_plot[df_to_plot[x_col].isin(top_20_categories)].groupby(x_col)[y_col].sum().reset_index()
                    plot_df_agg.plot(kind='bar', x=x_col, y=y_col, ax=ax)
                    ax.set_title(f'Bar Chart: {y_col} by top 20 {x_col}')
                else:
                    df_to_plot.plot(kind='bar', x=x_col, y=y_col, ax=ax)
                    ax.set_title(f'Bar Chart: {y_col} by {x_col}')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                plt.xticks(rotation=45, ha='right')
            elif len(numeric_cols) >= 2:
                x_col = numeric_cols[1] if numeric_cols[0] == y_col and len(numeric_cols) > 1 else numeric_cols[0]
                # Avoid plotting identical columns if x_col and y_col ended up being the same
                if x_col == y_col and len(numeric_cols) > 1:
                    x_col = numeric_cols[1]
                elif x_col == y_col and len(numeric_cols) <=1:
                     return jsonify({'error': 'Need at least two distinct numeric columns for a line plot if no categorical columns are present.'}), 400


                if df_to_plot.shape[0] > 100:
                    df_to_plot.sample(100).sort_values(by=x_col).plot(kind='line', x=x_col, y=y_col, ax=ax)
                else:
                    df_to_plot.sort_values(by=x_col).plot(kind='line', x=x_col, y=y_col, ax=ax)
                ax.set_title(f'Line Plot: {y_col} vs {x_col}')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
            else:
                df_to_plot[y_col].plot(kind='hist', bins=20, ax=ax)
                ax.set_title(f'Histogram of {y_col}')
                ax.set_xlabel(y_col)
                ax.set_ylabel('Frequency')
        else:
            return jsonify({'error': 'Plotting logic could not determine a suitable plot for this data structure.'}), 400

        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode('utf-8')
        return jsonify({'plot_image': f'data:image/png;base64,{base64_img}'}), 200

    except Exception as e_plot:
        return jsonify({'error': f'Error generating plot: {str(e_plot)}'}), 500
    finally:
        if fig: # Ensure figure is closed if it was created
            plt.close(fig)


@app.route('/execute_sql', methods=['POST'])
def execute_sql_query_route():
    global current_uploaded_filepath, current_metadata, current_uploaded_filename, last_successful_df
    data = request.get_json()
    sql_query = data.get('sql_query')
    
    if not sql_query:
        return jsonify({'error': 'No SQL query provided.'}), 400
    
    if not current_uploaded_filepath:
        return jsonify({'error': 'No file has been uploaded or file context lost.'}), 400

    table_name_in_db = current_metadata.get('table_name') if current_metadata else None
    if not table_name_in_db and current_uploaded_filename:
        table_name_in_db = current_uploaded_filename.rsplit('.', 1)[0]
    
    if not table_name_in_db:
        return jsonify({'error': 'Could not determine table name for query execution.'}), 400
    
    results_df, error_message = execute_duckdb_query(sql_query, current_uploaded_filepath, table_name_in_db)

    if error_message:
        return jsonify({'error': error_message}), 400 # Return 400 for query errors
    else:
        if results_df is not None:
            last_successful_df = results_df.copy()
        else:
            last_successful_df = pd.DataFrame() # Ensure it's a DataFrame even if empty

        results_json = results_df.to_dict(orient='records') if results_df is not None else []
        return jsonify({'results': results_json}), 200


if __name__ == '__main__':
    # Ensure UPLOAD_FOLDER exists when running directly
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
