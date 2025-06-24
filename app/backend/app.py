from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
import openai # Still needed for the old custom framework path
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
import traceback

# ADK components
from app.backend.adk_components.main_agent import create_basic_gemini_agent
from google.adk.sessions import Session as AdkSession
try:
    from google.adk.runtime import AutoSession
except ImportError:
    print("WARN: google.adk.runtime.AutoSession not found. ADK basic test path will try session=None.")
    AutoSession = None


load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

app = Flask(__name__, template_folder='../templates', static_folder='../static')
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
ALLOWED_EXTENSIONS = {'csv', 'parquet', 'sqlite', 'rdata', 'rda'}

last_successful_df = None
current_uploaded_filename = None
current_uploaded_filepath = None
current_metadata = None

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

if AZURE_OPENAI_ENDPOINT:
    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = "2023-07-01-preview"
    if OPENAI_API_KEY: openai.api_key = OPENAI_API_KEY
elif OPENAI_API_BASE:
    openai.api_base = OPENAI_API_BASE
    if OPENAI_API_KEY: openai.api_key = OPENAI_API_KEY
    else: print("Warning: OPENAI_API_KEY not set for custom OpenAI endpoint (old framework).")
elif OPENAI_API_KEY: openai.api_key = OPENAI_API_KEY
else: print("Warning: OpenAI API key or endpoint not configured for old framework. The /query endpoint old path may not work.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def execute_duckdb_query(sql_query: str, file_path: str, table_name: str):
    if not file_path: return None, "File path is missing."
    file_ext = file_path.rsplit('.', 1)[1].lower()
    try:
        if file_ext == 'sqlite': con = duckdb.connect(database=file_path, read_only=True)
        else:
            con = duckdb.connect(database=':memory:', read_only=False)
            if file_ext == 'csv': con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
            elif file_ext == 'parquet': con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')")
            else: return None, f"Unsupported file type for querying: {file_ext}"
        result_df = con.execute(sql_query).fetchdf(); con.close(); return result_df, None
    except duckdb.Error as e:
        try:
            if 'con' in locals() and con: con.close()
        except Exception: pass
        return None, f"DuckDB SQL execution error: {str(e)}"
    except FileNotFoundError: return None, f"Data file not found: {file_path}"
    except Exception as e:
        try:
            if 'con' in locals() and con: con.close()
        except Exception: pass
        return None, f"An error occurred during SQL execution: {str(e)}"

def execute_r_script(r_code_string: str, rdata_file_path: str, target_object_name: str) -> tuple[pd.DataFrame | None, str | None]:
    temp_r_script_path, temp_csv_path = "", ""
    try:
        rdata_file_path_r = rdata_file_path.replace('\\', '/')
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.R', encoding='utf-8') as o: temp_r_script_path = o.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as o: temp_csv_path = o.name
        temp_csv_path_r = temp_csv_path.replace('\\', '/')
        r_script_content = f"""
        options(error = function() {{ cat(geterrmessage(), file = stderr()); quit(save = "no", status = 1, runLast = FALSE); }})
        if (!requireNamespace("data.table", quietly = TRUE)) {{ write("Error: data.table package is not installed.", stderr()); quit(save = "no", status = 1, runLast = FALSE); }}
        library(data.table)
        tryCatch({{
            load_env <- new.env(); load("{rdata_file_path_r}", envir=load_env)
            if (!exists("{target_object_name}", envir=load_env)) {{ stop(paste0("Object '", "{target_object_name}", "' not found in the Rdata file.")) }}
            active_df <- load_env[["{target_object_name}"]]
            if (!is.data.table(active_df)) {{
                if (is.data.frame(active_df)) {{ active_df <- as.data.table(active_df) }}
                else {{ stop(paste0("Object '", "{target_object_name}", "' is not a data.frame or data.table.")) }}
            }}
            eval(parse(text = {repr(r_code_string)}))
            if (!exists("active_df")){{ stop("The R code did not result in an 'active_df' object.") }}
            if (nrow(active_df) == 0 && !is.data.table(active_df)) {{ stop("Result of R code is not a data.table and is empty.") }}
            fwrite(active_df, file="{temp_csv_path_r}", row.names=FALSE)
        }}, error = function(e) {{ write(paste("R script execution error:", e$message), stderr()); quit(save = "no", status = 1, runLast = FALSE); }})
        quit(save = "no", status = 0, runLast = FALSE)
        """
        with open(temp_r_script_path, 'w', encoding='utf-8') as f: f.write(r_script_content)
        process = subprocess.run(['Rscript', temp_r_script_path], capture_output=True, text=True, check=False, encoding='utf-8')
        if process.returncode == 0:
            if os.path.exists(temp_csv_path):
                if os.path.getsize(temp_csv_path) > 0:
                    try: return pd.read_csv(temp_csv_path), None
                    except pd.errors.EmptyDataError: return pd.DataFrame(), None
                    except Exception as e_read: return None, f"Error reading R script output CSV: {str(e_read)}. R stderr: {process.stderr.strip()}"
                else: return pd.DataFrame(), None
            else: return None, f"R script executed successfully but output CSV not found or empty. R stderr: {process.stderr.strip()}"
        else:
            error_message = f"R script execution failed (return code {process.returncode}). Error: {process.stderr.strip()}"
            if not process.stderr.strip(): error_message = f"R script execution failed (return code {process.returncode}) with no specific error message."
            return None, error_message
    except FileNotFoundError: return None, "Rscript command not found. Please ensure R is installed and in PATH."
    except Exception as e: return None, f"Python error during R script execution: {str(e)}"
    finally:
        for p in [temp_r_script_path, temp_csv_path]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception as e_clean: print(f"Warning: Could not delete temporary file {p}: {e_clean}")

def execute_python_pandas_code(python_code_string: str, data_file_path: str, dataframe_name: str = 'df') -> tuple[pd.DataFrame | None, str | None]:
    temp_script_path, temp_output_csv_path, temp_user_code_path = "", "", ""
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as o: temp_user_code_path = o.name; o.write(python_code_string); o.flush()
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as o: temp_script_path = o.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as o: temp_output_csv_path = o.name
        data_file_path_script = data_file_path.replace('\\', '/'); output_csv_path_script = temp_output_csv_path.replace('\\', '/'); user_code_path_script = temp_user_code_path.replace('\\', '/')
        script_lines = [
            "import pandas as pd", "import sys", "import os", "",
            f"data_path = r'{data_file_path_script}'", f"df_name = '{dataframe_name}'",
            f"output_csv_path = r'{output_csv_path_script}'", f"user_code_path = r'{user_code_path_script}'", "",
            "try:",
            "    with open(user_code_path, 'r', encoding='utf-8') as f: user_code = f.read()", "",
            "    if data_path.endswith('.csv'): globals()[df_name] = pd.read_csv(data_path)",
            "    elif data_path.endswith('.parquet'): globals()[df_name] = pd.read_parquet(data_path)",
            "    else: raise ValueError(f\"Unsupported file type: {{data_path}}. Only CSV and Parquet are supported.\")", "",
            "    exec(user_code, globals())", "",
            "    if df_name not in globals(): print(f\"Error: DataFrame '{{df_name}}' not found after code execution. Did you delete or rename it?\", file=sys.stderr); sys.exit(1)", "",
            "    result_df = globals()[df_name]", "",
            "    if isinstance(result_df, pd.DataFrame): result_df.to_csv(output_csv_path, index=False); print(output_csv_path)",
            "    else: print(f\"Error: Resulting object '{{df_name}}' is not a Pandas DataFrame (type: {{type(result_df)}}).\", file=sys.stderr); sys.exit(1)", "",
            "except FileNotFoundError as e_fnf: print(f\"Error loading data: {{e_fnf}}\", file=sys.stderr); sys.exit(1)",
            "except pd.errors.EmptyDataError as e_ede: print(f\"Error loading data: The file '{{data_path}}' is empty or contains no data.\", file=sys.stderr); sys.exit(1)",
            "except ValueError as e_ve: print(f\"Error: {{e_ve}}\", file=sys.stderr); sys.exit(1)",
            "except Exception as e: print(f\"Error during Python code execution: {{str(e)}}\", file=sys.stderr); sys.exit(1)",
        ]
        script_content = "\n".join(script_lines)
        with open(temp_script_path, 'w', encoding='utf-8') as f: f.write(script_content)
        process = subprocess.run([sys.executable, temp_script_path], capture_output=True, text=True, check=False, encoding='utf-8')
        if process.returncode == 0:
            output_file_from_script = process.stdout.strip()
            if os.path.exists(output_file_from_script):
                try: return pd.read_csv(output_file_from_script), None
                except pd.errors.EmptyDataError: return pd.DataFrame(), None
                except Exception as e_csv: return None, f"Error reading result CSV from script: {{str(e_csv)}}. Stderr: {{process.stderr.strip()}}"
            else: return None, f"Script executed successfully but output file '{{output_file_from_script}}' not found. Stderr: {{process.stderr.strip()}}"
        else:
            error_message = f"Python script execution failed (return code {{process.returncode}}). Error: {{process.stderr.strip()}}"
            if not process.stderr.strip(): error_message = f"Python script execution failed (return code {{process.returncode}}) with no specific error message from stderr."
            return None, error_message
    except FileNotFoundError: return None, "Error: Python interpreter or temporary script file not found."
    except Exception as e: return None, f"Python error in 'execute_python_pandas_code' function: {{str(e)}}"
    finally:
        for p in [temp_script_path, temp_user_code_path, temp_output_csv_path]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception as e_clean: print(f"Warning: Could not delete temp file {{p}}: {{e_clean}}")

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_uploaded_filename, current_metadata, current_uploaded_filepath
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = file.filename; filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try: file.save(filepath)
        except Exception as e_save: return jsonify({'error': f'Failed to save file: {e_save}'}), 500
        current_uploaded_filename = filename; current_uploaded_filepath = filepath; current_metadata = None
        metadata_inferred = {}
        file_ext_lower = filename.rsplit('.', 1)[1].lower()

        if file_ext_lower == 'csv':
            try:
                df = pd.read_csv(filepath, nrows=100)
                metadata_inferred['table_name'] = filename.rsplit('.', 1)[0]
                metadata_inferred['columns'] = []
                for column in df.columns:
                    col_type = 'TEXT'
                    try:
                        if pd.api.types.is_integer_dtype(df[column].dropna()):
                             if (df[column].dropna() % 1 == 0).all(): col_type = 'INTEGER'
                             else: col_type = 'REAL'
                        elif pd.api.types.is_float_dtype(df[column].dropna()): col_type = 'REAL'
                        elif pd.api.types.is_numeric_dtype(df[column].dropna()):
                            temp_series = df[column].dropna()
                            if (temp_series % 1 == 0).all(): col_type = 'INTEGER'
                            else: col_type = 'REAL'
                    except Exception: pass
                    metadata_inferred['columns'].append({'name': column, 'type': col_type})
                current_metadata = metadata_inferred
                return jsonify({'message': 'File uploaded successfully', 'metadata': metadata_inferred}), 200
            except Exception as e: return jsonify({'error': f'Error processing CSV file: {str(e)}'}), 500
        elif file_ext_lower == 'parquet':
            try:
                pq_schema = pd.io.parquet.read_schema(filepath)
                metadata_inferred['table_name'] = filename.rsplit('.', 1)[0]; metadata_inferred['columns'] = []
                for i in range(len(pq_schema)):
                    field = pq_schema.field(i); col_name = field.name; arrow_type = field.type; col_type = 'TEXT'
                    if pa.types.is_integer(arrow_type): col_type = 'INTEGER'
                    elif pa.types.is_floating(arrow_type): col_type = 'REAL'
                    elif pa.types.is_boolean(arrow_type): col_type = 'BOOLEAN'
                    elif pa.types.is_temporal(arrow_type): col_type = 'DATETIME'
                    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type) or pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type): col_type = 'TEXT'
                    metadata_inferred['columns'].append({'name': col_name, 'type': col_type})
                current_metadata = metadata_inferred
                return jsonify({'message': 'File uploaded successfully (Parquet)', 'metadata': metadata_inferred}), 200
            except Exception as e: return jsonify({'error': f'Error processing Parquet file: {str(e)}'}), 500
        elif file_ext_lower in ['rdata', 'rda']:
            try:
                r_script = f"""
                tryCatch({{ load('{filepath.replace('\\\\', '/')}') # Use R-compatible path
                    obj_name <- ls()[1]; data_obj <- get(obj_name)
                    if (is.data.frame(data_obj) || inherits(data_obj, "data.table")) {{
                        cols <- colnames(data_obj); types <- sapply(data_obj, class)
                        formatted_types <- lapply(types, function(t) {{ if (is.array(t) || is.list(t)) paste(t, collapse=", ") else t }})
                        cat(jsonlite::toJSON(list(object_name=obj_name, columns=mapply(function(n,t) list(name=n,type=t),cols,formatted_types,SIMPLIFY=FALSE)),auto_unbox=TRUE))
                    }} else {{ stop("First object in Rdata is not a data.frame or data.table.") }}
                }}, error = function(e) {{ write(paste("R script error:", e$message), stderr()); stop(e); }})"""
                process = subprocess.run(['Rscript', '-', filepath.replace('\\', '/')], input=r_script, text=True, capture_output=True, check=False)
                if process.returncode == 0 and process.stdout:
                    r_metadata = json.loads(process.stdout)
                    type_mapping = {'integer':'INTEGER', 'numeric':'REAL', 'character':'TEXT', 'factor':'TEXT', 'logical':'BOOLEAN', 'Date':'DATE', 'POSIXct':'TIMESTAMP', 'POSIXlt':'TIMESTAMP'}
                    metadata_inferred = {'table_name': r_metadata.get('object_name', filename.rsplit('.', 1)[0]), 'columns': []}
                    for col in r_metadata.get('columns', []):
                        sql_type = type_mapping.get(col.get('type', 'character').split(',')[0].strip(), 'TEXT')
                        metadata_inferred['columns'].append({'name': col['name'], 'type': sql_type})
                    current_metadata = metadata_inferred
                    return jsonify({'message': 'RData processed successfully. Metadata extracted.', 'metadata': current_metadata}), 200
                else:
                    error_message = f"Error executing R script: {process.stderr.strip()}" if process.stderr.strip() else f"R script failed (code {process.returncode})"
                    current_metadata = {'table_name': filename.rsplit('.',1)[0], 'columns': [], 'message': f'Failed to process R data file: {error_message}'}
                    return jsonify({'error': error_message, 'metadata': current_metadata}), 500
            except Exception as e: return jsonify({'error': f'Error processing R data file: {str(e)}'}),500
        current_metadata = {'table_name': filename.rsplit('.',1)[0], 'columns': [], 'message': 'Metadata not auto-inferred.'}
        return jsonify({'message': 'File uploaded. Metadata needs manual input.', 'metadata': current_metadata }), 200
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/query', methods=['POST'])
def handle_query():
    global current_metadata, current_uploaded_filepath, current_uploaded_filename, last_successful_df
    data = request.get_json()
    if not data: return jsonify({'error': 'Invalid JSON payload'}), 400

    natural_language_query = data.get('naturalLanguageQuery')
    agent_type = data.get('agent_type', 'sql')
    llm_choice = data.get('llm_choice', 'openai')
    topology_choice = data.get('topology_choice', 'sequential_reflect')
    topology_specific_config = data.get('topology_config', {})
    metadata_for_query = data.get('metadata', current_metadata)

    if not natural_language_query: return jsonify({'error': 'No natural language query provided.'}), 400
    if not current_uploaded_filepath or not current_uploaded_filename: return jsonify({'error': 'No file uploaded or context lost.'}), 400

    if not metadata_for_query or not metadata_for_query.get('table_name'):
        metadata_for_query['table_name'] = current_uploaded_filename.rsplit('.', 1)[0]
        if 'columns' not in metadata_for_query: metadata_for_query['columns'] = []

    contextual_table_name = metadata_for_query.get('table_name')
    if not contextual_table_name: return jsonify({'error': 'Contextual table name missing.'}), 400

    if llm_choice == "adk_gemini_test":
        try:
            if not os.environ.get("GEMINI_API_KEY"):
                return jsonify({'error': 'GEMINI_API_KEY not set for ADK Gemini agent.'}), 500
            adk_agent = create_basic_gemini_agent()
            session_to_pass = None
            adk_response = None

            if AutoSession:
                print("INFO: Using AutoSession for ADK.")
                with AutoSession(appName="DataAnalyticsApp", userId="default_user", id="adk_test_session") as session:
                    session_to_pass = session
                    runtime_context = {"natural_language_query": natural_language_query, "metadata": metadata_for_query,
                                     "agent_type": agent_type, "file_path": current_uploaded_filepath,
                                     "table_name": contextual_table_name, "original_uploaded_filename": current_uploaded_filename}
                    adk_response = adk_agent.send_sync(session=session_to_pass, message=natural_language_query, context=runtime_context)
            else:
                print("WARN: AutoSession not found. Attempting send_sync with session=None for ADK agent.")
                runtime_context = {"natural_language_query": natural_language_query, "metadata": metadata_for_query,
                                 "agent_type": agent_type, "file_path": current_uploaded_filepath,
                                 "table_name": contextual_table_name, "original_uploaded_filename": current_uploaded_filename}
                adk_response = adk_agent.send_sync(session=None, message=natural_language_query, context=runtime_context)

            return jsonify({
                'executed_query_text': None, 'results': [], 'error': None,
                'natural_language_response': adk_response.message if adk_response else "ADK agent did not return a response.",
                'intermediate_steps': [{"step": "ADK Basic Agent Invoked", "response_message": adk_response.message if adk_response else "N/A"}]
            }), 200
        except Exception as e_adk:
            print(f"Error during ADK agent execution: {str(e_adk)}")
            traceback.print_exc()
            return jsonify({'error': f'ADK agent execution failed: {str(e_adk)}'}), 500
    else: # Fallback to Old Custom Framework
        # This is the old logic using the custom LLMFactory and TopologyFactory.
        # It requires OPENAI_API_KEY to be set if 'openai' is chosen.
        if not (OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT) and llm_choice == 'openai':
             return jsonify({'error': 'OpenAI API not configured for custom framework path.'}), 500

        is_sqlite = current_uploaded_filename.rsplit('.', 1)[1].lower() == 'sqlite'
        if agent_type == 'python_pandas' or agent_type == 'r_datatable' or (agent_type == 'sql' and not is_sqlite):
            if not metadata_for_query or not metadata_for_query.get('columns'):
                 return jsonify({'error': f'Column metadata is missing or not provided for the selected agent (custom framework).' }), 400

        print(f"INFO: Using custom framework for llm_choice '{llm_choice}' and topology '{topology_choice}'")
        try:
            from .llm_providers.factory import LLMFactory as CustomLLMFactory
            from .topologies.factory import TopologyFactory as CustomTopologyFactory
            from .topologies.factory import TopologyFactoryError as CustomTopologyFactoryError
            from .llm_providers.base import LLMProvider as CustomLLMProvider

            llm_provider_instance: CustomLLMProvider = CustomLLMFactory.get_llm_provider(llm_choice)
            topology_instance = CustomTopologyFactory.get_topology(
                topology_name=topology_choice, llm_provider=llm_provider_instance,
                topology_specific_config=topology_specific_config
            )
            topology_result = topology_instance.execute(
                natural_language_query=natural_language_query, metadata=metadata_for_query,
                agent_type=agent_type, file_path=current_uploaded_filepath,
                table_name=contextual_table_name, original_uploaded_filename=current_uploaded_filename
            )
            if topology_result.get('error') is None and topology_result.get('results'):
                try:
                    df_from_results = pd.DataFrame(topology_result['results'])
                    last_successful_df = df_from_results.copy() if not df_from_results.empty else pd.DataFrame()
                except Exception: last_successful_df = pd.DataFrame()
            elif topology_result.get('error'): last_successful_df = pd.DataFrame()
            return jsonify(topology_result), 200
        except (ValueError, CustomTopologyFactoryError) as e_factory:
            print(f"Custom Framework Factory Error: {str(e_factory)}")
            return jsonify({'error': f'Custom framework config error: {str(e_factory)}'}), 400
        except Exception as e_custom:
            print(f"Unexpected error in custom framework: {str(e_custom)}")
            traceback.print_exc()
            return jsonify({'error': f'Unexpected server error in custom framework: {str(e_custom)}'}), 500

@app.route('/plot_data', methods=['POST'])
def plot_data():
    global last_successful_df
    if last_successful_df is None or last_successful_df.empty:
        return jsonify({'error': 'No data available to plot. Please execute a query first.'}), 400
    df_to_plot = last_successful_df.copy()
    plt.figure(figsize=(8, 6))
    try:
        if len(df_to_plot.columns) == 1 and pd.api.types.is_numeric_dtype(df_to_plot.iloc[:, 0]):
            df_to_plot.iloc[:, 0].plot(kind='hist', bins=20); plt.title(f'Histogram of {df_to_plot.columns[0]}')
            plt.xlabel(df_to_plot.columns[0]); plt.ylabel('Frequency')
        elif len(df_to_plot.columns) >= 2:
            numeric_cols = df_to_plot.select_dtypes(include=pd.np.number).columns.tolist()
            if not numeric_cols: return jsonify({'error': 'No numeric columns found for plotting.'}), 400
            y_col = numeric_cols[0]
            categorical_cols = df_to_plot.select_dtypes(include='object').columns.tolist()
            if categorical_cols:
                x_col = categorical_cols[0]
                if df_to_plot[x_col].nunique() > 20:
                    top_20 = df_to_plot[x_col].value_counts().nlargest(20).index
                    plot_df_agg = df_to_plot[df_to_plot[x_col].isin(top_20)].groupby(x_col)[y_col].sum().reset_index()
                    plot_df_agg.plot(kind='bar', x=x_col, y=y_col); plt.title(f'Bar Chart: {y_col} by top 20 {x_col}')
                else: df_to_plot.plot(kind='bar', x=x_col, y=y_col); plt.title(f'Bar Chart: {y_col} by {x_col}')
                plt.xlabel(x_col); plt.ylabel(y_col); plt.xticks(rotation=45, ha='right')
            elif len(numeric_cols) >=2:
                x_col = numeric_cols[1] if numeric_cols[0] == y_col and len(numeric_cols) > 1 else numeric_cols[0]
                df_to_plot.sample(min(100, df_to_plot.shape[0])).sort_values(by=x_col).plot(kind='line', x=x_col, y=y_col)
                plt.title(f'Line Plot: {y_col} vs {x_col}'); plt.xlabel(x_col); plt.ylabel(y_col)
            else:
                df_to_plot[y_col].plot(kind='hist', bins=20); plt.title(f'Histogram of {y_col}')
                plt.xlabel(y_col); plt.ylabel('Frequency')
        else:
            return jsonify({'error': 'Plotting logic not implemented for this data structure (e.g., no numeric columns or too few columns).'}), 400
        plt.tight_layout(); img_buffer = io.BytesIO(); plt.savefig(img_buffer, format='png'); img_buffer.seek(0); plt.close()
        return jsonify({'plot_image': f'data:image/png;base64,{base64.b64encode(img_buffer.read()).decode("utf-8")}'}), 200
    except Exception as e_plot:
        plt.close()
        return jsonify({'error': f'Error generating plot: {str(e_plot)}'}), 500

@app.route('/execute_sql', methods=['POST'])
def execute_sql_query_route():
    global current_uploaded_filepath, current_metadata, current_uploaded_filename, last_successful_df
    data = request.get_json(); sql_query = data.get('sql_query')
    if not sql_query: return jsonify({'error': 'No SQL query provided.'}), 400
    if not current_uploaded_filepath: return jsonify({'error': 'No file uploaded or context lost.'}), 400
    table_name_in_db = (current_metadata.get('table_name') if current_metadata else None) or \
                       (current_uploaded_filename.rsplit('.', 1)[0] if current_uploaded_filename else None)
    if not table_name_in_db: return jsonify({'error': 'Could not determine table name for query execution.'}), 400
    results_df, error_message = execute_duckdb_query(sql_query, current_uploaded_filepath, table_name_in_db)
    if error_message: return jsonify({'error': error_message}), 400
    last_successful_df = results_df.copy() if results_df is not None else pd.DataFrame()
    return jsonify({'results': results_df.to_dict(orient='records') if results_df is not None else []}), 200

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
