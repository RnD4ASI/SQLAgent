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
        
        # For other file types, metadata remains None or minimal
        current_metadata = {'table_name': filename.rsplit('.',1)[0], 'columns': [], 'message': 'Metadata not automatically inferred for this file type. Please provide or approve manually.'}
        return jsonify({'message': 'File uploaded successfully. Metadata may need manual input.', 'metadata': current_metadata }), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/query', methods=['POST'])
def handle_query():
    global current_metadata, current_uploaded_filepath, current_uploaded_filename
    
    if not (OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT):
        return jsonify({'error': 'OpenAI API not configured on the server.', 'sql_query': None, 'results': None}), 500

    data = request.get_json()
    natural_language_query = data.get('naturalLanguageQuery')
    # Use metadata passed from frontend if available, otherwise use globally stored one
    # This allows for potential manual correction of metadata on the UI side later
    metadata_for_prompt = data.get('metadata', current_metadata) 

    if not natural_language_query:
        return jsonify({'error': 'No natural language query provided.', 'sql_query': None, 'results': None}), 400
    
    if not current_uploaded_filepath:
         return jsonify({'error': 'No file has been uploaded yet or file context lost.', 'sql_query': None, 'results': None}), 400

    # Determine table name for the query
    table_name_for_query = metadata_for_prompt.get('table_name') if metadata_for_prompt else None
    if not table_name_for_query and current_uploaded_filename: # Fallback to filename if not in metadata
        table_name_for_query = current_uploaded_filename.rsplit('.', 1)[0]
    
    if not table_name_for_query:
        return jsonify({'error': 'Could not determine table name for query.', 'sql_query': None, 'results': None}), 400

    # Check for columns, especially if not SQLite (for which schema is in file)
    if current_uploaded_filename.rsplit('.', 1)[1].lower() not in ['sqlite']:
        if not metadata_for_prompt or not metadata_for_prompt.get('columns'):
            return jsonify({'error': 'Column metadata is missing for this file type. Please ensure metadata is processed.', 'sql_query': None, 'results': None}), 400

    # --- First LLM Call to generate SQL ---
    prompt_parts = [
        "Given the table schema below and the user question, generate a valid SQL query to answer the question."
    ]
    prompt_parts.append(f"Table Name: {table_name_for_query}")

    if metadata_for_prompt and metadata_for_prompt.get('columns'):
        prompt_parts.append("Columns:")
        for column in metadata_for_prompt['columns']:
            prompt_parts.append(f"- {column['name']} ({column['type']})")
    else:
        prompt_parts.append("Columns: (Schema not fully provided or is embedded, e.g., in a SQLite file. Ensure your query references correct table and column names based on the file's actual schema.)")

    prompt_parts.append(f"\nUser Question: {natural_language_query}")
    prompt_parts.append("SQL Query:")
    current_prompt = "\n".join(prompt_parts)
    
    sql_query = "" # Initialize sql_query to ensure it's always defined

    try:
        deployment_name = os.environ.get("AZURE_DEPLOYMENT_NAME")

        if openai.api_type == "azure" and not deployment_name:
            return jsonify({'error': 'AZURE_DEPLOYMENT_NAME environment variable not set for Azure OpenAI.', 'sql_query': None, 'results': None}), 500

        if openai.api_type == "azure":
            response = openai.Completion.create(engine=deployment_name, prompt=current_prompt, max_tokens=150, temperature=0.1)
        else:
            response = openai.Completion.create(model="text-davinci-003", prompt=current_prompt, max_tokens=150, temperature=0.1)
        
        sql_query = response.choices[0].text.strip()
        if not sql_query:
            return jsonify({'error': 'LLM did not return a SQL query.', 'sql_query': '', 'results': None}), 500

        # --- Attempt to execute the first generated SQL ---
        results_df, error_message = execute_duckdb_query(sql_query, current_uploaded_filepath, table_name_for_query)

        if error_message: # SQL execution failed
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
            else:
                reflection_prompt_parts.append("Columns: (Schema not fully provided or embedded)")
            
            reflection_prompt_parts.extend([
                f"Failed SQL: {sql_query}",
                f"Error Message: {error_message}",
                "Corrected SQL Query:"
            ])
            correction_prompt = "\n".join(reflection_prompt_parts)

            if openai.api_type == "azure":
                correction_response = openai.Completion.create(engine=deployment_name, prompt=correction_prompt, max_tokens=150, temperature=0.15) # Slightly higher temp
            else:
                correction_response = openai.Completion.create(model="text-davinci-003", prompt=correction_prompt, max_tokens=150, temperature=0.15)
            
            corrected_sql_query = correction_response.choices[0].text.strip()
            if not corrected_sql_query:
                 return jsonify({'sql_query': sql_query, 'error': f'LLM did not return a corrected SQL query. Original error: {error_message}', 'results': None}), 500
            
            sql_query = corrected_sql_query 
            print(f"Attempting corrected SQL query: {sql_query}")
            results_df, error_message = execute_duckdb_query(sql_query, current_uploaded_filepath, table_name_for_query)
        
        if error_message: 
            return jsonify({'sql_query': sql_query, 'error': f'SQL execution failed after correction attempt: {error_message}', 'results': None, 'natural_language_response': None}), 400
        else:
            global last_successful_df # To cache the dataframe
            last_successful_df = results_df.copy() if results_df is not None else pd.DataFrame() # Cache a copy
            results_json = results_df.to_dict(orient='records') if results_df is not None else []
            
            # --- Third LLM Call to generate Natural Language Summary ---
            nl_summary = "Could not generate natural language summary." # Default
            try:
                # Summarize results for the prompt
                result_summary_for_prompt = ""
                if not results_json:
                    result_summary_for_prompt = "The query returned no results."
                elif len(results_json) <= 5:
                    result_summary_for_prompt = f"The query returned the following results:\n{pd.DataFrame(results_json).to_string()}"
                else:
                    result_summary_for_prompt = f"The query returned {len(results_json)} rows. Here are the first 5:\n{pd.DataFrame(results_json[:5]).to_string()}\n...and {len(results_json)-5} more rows."

                summary_prompt_parts = [
                    f"Based on the user's question '{natural_language_query}', the SQL query '{sql_query}', and the following SQL query results, provide a concise natural language answer:",
                    result_summary_for_prompt,
                    "\nNatural Language Answer:"
                ]
                summary_prompt = "\n".join(summary_prompt_parts)

                if openai.api_type == "azure":
                    summary_response = openai.Completion.create(engine=deployment_name, prompt=summary_prompt, max_tokens=200, temperature=0.3) # Temp might be higher for creative summary
                else:
                    summary_response = openai.Completion.create(model="text-davinci-003", prompt=summary_prompt, max_tokens=200, temperature=0.3)
                
                nl_summary = summary_response.choices[0].text.strip()
                if not nl_summary:
                    nl_summary = "LLM did not provide a natural language summary."

            except openai.error.OpenAIError as e_sum:
                print(f"OpenAI API error during summary generation: {str(e_sum)}")
                nl_summary = f"Error generating natural language summary: {str(e_sum)}"
            except Exception as e_sum_gen:
                print(f"Unexpected error during summary generation: {str(e_sum_gen)}")
                nl_summary = f"Unexpected error generating natural language summary: {str(e_sum_gen)}"

            return jsonify({'sql_query': sql_query, 'results': results_json, 'error': None, 'natural_language_response': nl_summary}), 200

    except openai.error.OpenAIError as e:
        return jsonify({'error': f'OpenAI API error: {str(e)}', 'sql_query': sql_query, 'results': None, 'natural_language_response': None}), 500
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}', 'sql_query': sql_query, 'results': None, 'natural_language_response': None}), 500

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
