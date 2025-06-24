import pandas as pd
import duckdb
import subprocess
import tempfile
import os
import sys
import json # For R script metadata extraction, though not directly used by execution itself

# Matplotlib and base64 are for plotting, not direct code execution utils, so keep them in app.py for now.
# pyarrow is used by app.py for parquet metadata, not directly by execution helpers.

# --- SQL Execution (DuckDB) ---
def execute_duckdb_query(sql_query: str, file_path: str, table_name: str) -> tuple[pd.DataFrame | None, str | None]:
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
        con.close()
        return result_df, None # Success
    except duckdb.Error as e:
        try:
            if 'con' in locals() and con:
                con.close()
        except Exception:
            pass
        return None, f"DuckDB SQL execution error: {str(e)}"
    except FileNotFoundError:
        return None, f"Data file not found: {file_path}"
    except Exception as e:
        try:
            if 'con' in locals() and con:
                con.close()
        except Exception:
            pass
        return None, f"An error occurred during SQL execution: {str(e)}"

# --- R Script Execution ---
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
    temp_r_script_path = "" # Initialize to satisfy finally block if NamedTemporaryFile fails
    temp_csv_path = ""      # Initialize

    try:
        rdata_file_path_r = rdata_file_path.replace('\\', '/')

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.R', encoding='utf-8') as temp_r_script_file_obj:
            temp_r_script_path = temp_r_script_file_obj.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_csv_file_obj:
            temp_csv_path = temp_csv_file_obj.name

        temp_csv_path_r = temp_csv_path.replace('\\', '/')

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

            # User's R code is executed here.
            eval(parse(text = {repr(r_code_string)}))

            if (!exists("active_df")){{
                 stop("The R code did not result in an 'active_df' object.")
            }}

            if (nrow(active_df) == 0 && !is.data.table(active_df)) {{
                stop("Result of R code is not a data.table and is empty.")
            }}
            # fwrite handles 0-row data.tables correctly by creating an empty file with headers.

            fwrite(active_df, file="{temp_csv_path_r}", row.names=FALSE)

        }}, error = function(e) {{
            write(paste("R script execution error:", e$message), stderr())
            quit(save = "no", status = 1, runLast = FALSE)
        }})
        quit(save = "no", status = 0, runLast = FALSE)
        """

        with open(temp_r_script_path, 'w', encoding='utf-8') as f:
            f.write(r_script_content)

        process = subprocess.run(
            ['Rscript', temp_r_script_path],
            capture_output=True, text=True, check=False, encoding='utf-8'
        )

        if process.returncode == 0:
            if os.path.exists(temp_csv_path): # Check existence before size
                if os.path.getsize(temp_csv_path) > 0:
                    try:
                        df = pd.read_csv(temp_csv_path)
                        return df, None
                    except pd.errors.EmptyDataError: # Should be caught by getsize == 0 ideally
                        return pd.DataFrame(), None
                    except Exception as e_read:
                        return None, f"Error reading R script output CSV: {str(e_read)}. R stderr: {process.stderr.strip()}"
                else: # File exists but is empty (e.g. 0-row data.table)
                    return pd.DataFrame(), None
            else: # File does not exist (should not happen if Rscript succeeded and fwrite was called)
                 return None, f"R script executed successfully but output CSV not found. R stderr: {process.stderr.strip()}"
        else:
            error_message = f"R script execution failed (return code {process.returncode}). Error: {process.stderr.strip()}"
            if not process.stderr.strip():
                 error_message = f"R script execution failed (return code {process.returncode}) with no specific error message."
            return None, error_message

    except FileNotFoundError:
        return None, "Rscript command not found. Please ensure R is installed and in PATH."
    except Exception as e:
        return None, f"Python error during R script execution: {str(e)}"
    finally:
        if temp_r_script_path and os.path.exists(temp_r_script_path):
            try:
                os.remove(temp_r_script_path)
            except Exception as e_clean_r:
                print(f"Warning: Could not delete temporary R script {temp_r_script_path}: {e_clean_r}")
        if temp_csv_path and os.path.exists(temp_csv_path):
            try:
                os.remove(temp_csv_path)
            except Exception as e_clean_csv:
                 print(f"Warning: Could not delete temporary CSV file {temp_csv_path}: {e_clean_csv}")

# --- Python Pandas Code Execution ---
def execute_python_pandas_code(python_code_string: str, data_file_path: str, dataframe_name: str = 'df') -> tuple[pd.DataFrame | None, str | None]:
    """
    Executes Python Pandas code securely using a subprocess.

    Args:
        python_code_string (str): The Python code string to execute.
        data_file_path (str): Full path to the data file (CSV or Parquet).
        dataframe_name (str): The name of the Pandas DataFrame variable in the executed code.

    Returns:
        tuple: (pandas.DataFrame, None) if successful, or (None, str) if an error occurred.
    """
    temp_script_path = ""
    temp_output_csv_path = ""
    temp_user_code_path = ""

    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as temp_user_code_file_obj:
            temp_user_code_path = temp_user_code_file_obj.name
            temp_user_code_file_obj.write(python_code_string)

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as temp_script_file_obj:
            temp_script_path = temp_script_file_obj.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_output_csv_file_obj:
            temp_output_csv_path = temp_output_csv_file_obj.name

        data_file_path_script = data_file_path.replace('\\', '/')
        output_csv_path_script = temp_output_csv_path.replace('\\', '/')
        user_code_path_script = temp_user_code_path.replace('\\', '/')

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
            "    if data_path.endswith('.csv'):",
            f"        globals()[df_name] = pd.read_csv(data_path)",
            "    elif data_path.endswith('.parquet'):",
            f"        globals()[df_name] = pd.read_parquet(data_path)",
            "    else:",
            "        raise ValueError(f\"Unsupported file type: {data_path}. Only CSV and Parquet are supported.\")",
            "",
            "    exec(user_code, globals())",
            "",
            "    if df_name not in globals():",
            "        print(f\"Error: DataFrame '{df_name}' not found after code execution.\", file=sys.stderr)",
            "        sys.exit(1)",
            "",
            "    result_df = globals()[df_name]",
            "",
            "    if isinstance(result_df, pd.DataFrame):",
            "        result_df.to_csv(output_csv_path, index=False)",
            "        print(output_csv_path) ", # Output path to stdout on success
            "    else:",
            "        print(f\"Error: Resulting object '{df_name}' is not a Pandas DataFrame.\", file=sys.stderr)",
            "        sys.exit(1)",
            "",
            "except FileNotFoundError as e_fnf:",
            "    print(f\"Error loading data: {e_fnf}\", file=sys.stderr)",
            "    sys.exit(1)",
            "except pd.errors.EmptyDataError as e_ede:",
            "    print(f\"Error loading data: The file '{data_path}' is empty.\", file=sys.stderr)",
            "    sys.exit(1)",
            "except ValueError as e_ve:",
            "    print(f\"Error: {e_ve}\", file=sys.stderr)",
            "    sys.exit(1)",
            "except Exception as e:",
            "    print(f\"Error during Python code execution: {str(e)}\", file=sys.stderr)",
            "    sys.exit(1)",
        ]
        script_content = "\n".join(script_lines)

        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        process = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True, text=True, check=False, encoding='utf-8'
        )

        if process.returncode == 0:
            output_file_from_script = process.stdout.strip()
            if os.path.exists(output_file_from_script):
                try:
                    returned_df = pd.read_csv(output_file_from_script)
                    return returned_df, None
                except pd.errors.EmptyDataError:
                    return pd.DataFrame(), None
                except Exception as e_read_csv:
                    return None, f"Error reading result CSV from script: {str(e_read_csv)}. Stderr: {process.stderr.strip()}"
            else:
                return None, f"Script executed successfully but output file '{output_file_from_script}' not found. Stderr: {process.stderr.strip()}"
        else:
            error_message = f"Python script execution failed (return code {process.returncode}). Error: {process.stderr.strip()}"
            if not process.stderr.strip():
                 error_message = f"Python script execution failed (return code {process.returncode}) with no specific error message."
            return None, error_message

    except FileNotFoundError:
        return None, "Error: Python interpreter or temporary script file not found."
    except Exception as e:
        return None, f"Python error in 'execute_python_pandas_code' function: {str(e)}"
    finally:
        for f_path in [temp_script_path, temp_user_code_path, temp_output_csv_path]:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception as e_clean:
                    print(f"Warning: Could not delete temporary file {f_path}: {e_clean}")
