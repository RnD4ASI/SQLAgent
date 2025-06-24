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

# ----- Existing OpenAI SDK Setup (will be replaced by LLMProvider logic later) ----
# Attempt to get OpenAI API key and endpoint from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE") # For self-hosted or Azure
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT") # For Azure

if AZURE_OPENAI_ENDPOINT:
    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = "2023-07-01-preview" # Specify a valid API version for Azure
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
    print("Warning: OpenAI API key or endpoint not configured. The /query endpoint's old logic will not work.")
# ----- End of old OpenAI SDK Setup -----


# Import new code execution functions
from .code_execution import execute_duckdb_query, execute_r_script, execute_python_pandas_code

# Import new LLM and Topology components
from .llm_providers.factory import LLMFactory
from .topologies.factory import TopologyFactory, TopologyFactoryError
from .llm_providers.base import LLMProvider # For type hinting


# Store metadata globally for simplicity in this example
current_metadata = None # Holds schema of the uploaded file
# Global variable to cache the last successful DataFrame for plotting
last_successful_df = None # Updated by the /query endpoint
# Store the last uploaded filename and filepath globally
# In a real app, this should be managed per session or via a more robust mechanism
current_uploaded_filename = None
current_uploaded_filepath = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# execute_duckdb_query, execute_r_script, execute_python_pandas_code
# are now imported from .code_execution


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
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON payload'}), 400

    natural_language_query = data.get('naturalLanguageQuery')
    agent_type = data.get('agent_type', 'sql') # Default: sql
    llm_choice = data.get('llm_choice', 'openai') # Default: openai
    topology_choice = data.get('topology_choice', 'sequential_reflect') # Default: sequential_reflect
    # Frontend might pass topology_specific_config if needed in future
    topology_specific_config = data.get('topology_config', {})

    # Use metadata passed from frontend if available, otherwise use globally stored one
    # This metadata should ideally include table_name and columns for the agent
    metadata_for_query = data.get('metadata', current_metadata)

    if not natural_language_query:
        return jsonify({'error': 'No natural language query provided.'}), 400
    
    if not current_uploaded_filepath or not current_uploaded_filename:
         return jsonify({'error': 'No file has been uploaded yet or file context lost.'}), 400

    if not metadata_for_query or not metadata_for_query.get('table_name'):
        # Try to infer table_name if missing in metadata but present in older global current_metadata
        # or from filename; this is a fallback. Ideally, frontend sends complete metadata.
        table_name_from_global_meta = current_metadata.get('table_name') if current_metadata else None
        table_name_from_filename = current_uploaded_filename.rsplit('.', 1)[0]

        inferred_table_name = metadata_for_query.get('table_name') or \
                              table_name_from_global_meta or \
                              table_name_from_filename

        if not inferred_table_name:
            return jsonify({'error': 'Could not determine table name for query from metadata or filename.'}), 400

        # If metadata_for_query was incomplete, update it with inferred table_name for the topology
        if not metadata_for_query.get('table_name'):
            metadata_for_query['table_name'] = inferred_table_name
        # Also ensure 'columns' key exists, even if empty, as topologies might expect it.
        if 'columns' not in metadata_for_query:
             # Try to get columns from global current_metadata if available
            metadata_for_query['columns'] = current_metadata.get('columns', []) if current_metadata else []


    # The 'table_name' for the topology's execute method is the contextual name
    # (SQL table alias, R object name, Python DataFrame variable name).
    # This should be present in metadata_for_query['table_name'].
    contextual_table_name = metadata_for_query.get('table_name')
    if not contextual_table_name: # Should be caught by above, but as a safeguard
        return jsonify({'error': 'Contextual table name for query execution is missing in metadata.'}), 400


    # Prerequisite check for column metadata (still important for many agents/prompts)
    # This check might be nuanced depending on the topology or specific LLM's needs
    is_sqlite_file = current_uploaded_filename.rsplit('.', 1)[1].lower() == 'sqlite'
    requires_column_metadata = agent_type in ['python_pandas', 'r_datatable'] or \
                               (agent_type == 'sql' and not is_sqlite_file)

    if requires_column_metadata and (not metadata_for_query.get('columns') or not isinstance(metadata_for_query['columns'], list)):
        # If columns are missing or not a list, this is problematic.
        # Attempt to fill from global current_metadata if it seems valid.
        if current_metadata and current_metadata.get('columns') and isinstance(current_metadata['columns'], list):
            print(f"Warning: Columns missing in query metadata for {agent_type}, using globally cached columns.")
            metadata_for_query['columns'] = current_metadata['columns']
        else:
            return jsonify({'error': f'Column metadata is missing or invalid for agent type {agent_type}. Please ensure metadata is processed or provided correctly.'}), 400


    try:
        # 1. Get LLM Provider
        llm_provider_instance: LLMProvider = LLMFactory.get_llm_provider(llm_choice)

        # 2. Get Topology
        # Pass the chosen llm_provider_instance and any specific config for this topology
        topology_instance = TopologyFactory.get_topology(
            topology_name=topology_choice,
            llm_provider=llm_provider_instance,
            topology_specific_config=topology_specific_config
        )

        # 3. Execute Topology
        # The topology's execute method will handle the core logic.
        # It needs all relevant context.
        topology_result = topology_instance.execute(
            natural_language_query=natural_language_query,
            metadata=metadata_for_query, # This contains table_name, columns
            agent_type=agent_type,
            file_path=current_uploaded_filepath,
            table_name=contextual_table_name, # Pass the contextual table name
            original_uploaded_filename=current_uploaded_filename # For specific checks like SQLite
        )

        # Update last_successful_df for plotting if results are present and valid
        if topology_result.get('error') is None and topology_result.get('results'):
            try:
                # Ensure results are in a format that can be made into a DataFrame
                # The topology should return results as list of dicts.
                df_from_results = pd.DataFrame(topology_result['results'])
                if not df_from_results.empty:
                    last_successful_df = df_from_results.copy()
                else:
                    last_successful_df = pd.DataFrame() # Reset if results were empty
            except Exception as e_df:
                print(f"Warning: Could not form DataFrame from topology results for caching: {e_df}")
                last_successful_df = pd.DataFrame() # Reset on error
        elif topology_result.get('error'):
            # If there was an error, clear last_successful_df to prevent plotting stale/incorrect data
            last_successful_df = pd.DataFrame()


        # Return the result from the topology
        # The topology result should already be a dictionary with keys like:
        # 'executed_query_text', 'results', 'error', 'natural_language_response', 'intermediate_steps'
        return jsonify(topology_result), 200

    except (ValueError, TopologyFactoryError) as e_factory: # Catch errors from factories
        print(f"Factory Error: {str(e_factory)}")
        return jsonify({'error': f'Configuration or setup error: {str(e_factory)}', 'results': None, 'natural_language_response': None, 'executed_query_text': None}), 400
    except Exception as e:
        # Catch any other unexpected errors during the new flow
        print(f"Unexpected error in /query endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}', 'results': None, 'natural_language_response': None, 'executed_query_text': None}), 500


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
