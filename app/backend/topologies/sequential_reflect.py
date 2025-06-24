import pandas as pd
from typing import Dict, Any, Tuple
from app.backend.llm_providers.base import LLMProvider
from app.backend.topologies.base import Topology
from app.backend.code_execution import (
    execute_duckdb_query,
    execute_r_script,
    execute_python_pandas_code
)

class SequentialReflectTopology(Topology):
    """
    Implements a sequential execution topology with a self-reflection/correction step.
    1. Generate code based on natural language query.
    2. Execute the code.
    3. If execution fails, use LLM to reflect on the error and generate corrected code.
    4. Execute the corrected code.
    5. Summarize the results using LLM.
    """

    def __init__(self, llm_provider: LLMProvider, topology_config: Dict[str, Any] | None = None):
        super().__init__(llm_provider, topology_config)
        # Default model names, can be overridden by topology_config
        self.default_code_gen_model = self._get_config_value("code_gen_model", "gpt-3.5-turbo") # Example default
        self.default_correction_model = self._get_config_value("correction_model", "gpt-3.5-turbo")
        self.default_summary_model = self._get_config_value("summary_model", "gpt-3.5-turbo")


    def _generate_initial_code(self, natural_language_query: str, metadata: Dict[str, Any], agent_type: str, table_name: str) -> str:
        """Generates the initial code/query."""
        prompt = self._construct_initial_prompt(natural_language_query, metadata, agent_type, table_name)
        # print(f"DEBUG: Initial generation prompt:\n{prompt}")
        code = self.llm_provider.generate_code(
            prompt=prompt,
            model_name=self._get_config_value("code_gen_model", self.default_code_gen_model),
            # Temperature, max_tokens can also be configured via topology_config or have defaults here
        )
        if not code:
            raise ValueError("LLM failed to generate initial code.")
        return code

    def _execute_generated_code(
        self,
        generated_code: str,
        agent_type: str,
        file_path: str,
        table_name_for_query: str, # This is the table/object name for the query execution context
        original_uploaded_filename: str # Used to determine if it's SQLite for SQL agent
    ) -> Tuple[pd.DataFrame | None, str | None]:
        """Executes the generated code based on agent type."""
        if agent_type == 'sql':
            return execute_duckdb_query(generated_code, file_path, table_name_for_query)
        elif agent_type == 'r_datatable':
            # For R, table_name_for_query is the R object name
            return execute_r_script(generated_code, file_path, table_name_for_query)
        elif agent_type == 'python_pandas':
            # For Python Pandas, table_name_for_query is the dataframe variable name
            return execute_python_pandas_code(generated_code, file_path, dataframe_name=table_name_for_query)
        else:
            return None, f"Unsupported agent_type for execution: {agent_type}"

    def _generate_corrected_code(
        self,
        natural_language_query: str,
        metadata: Dict[str, Any],
        agent_type: str,
        table_name: str,
        failed_code: str,
        error_message: str
    ) -> str:
        """Generates corrected code based on the failure."""
        prompt = self._construct_correction_prompt(
            natural_language_query, metadata, agent_type, table_name, failed_code, error_message
        )
        # print(f"DEBUG: Correction prompt:\n{prompt}")
        corrected_code = self.llm_provider.generate_code(
            prompt=prompt,
            model_name=self._get_config_value("correction_model", self.default_correction_model),
            temperature=0.15 # Potentially slightly higher temp for correction
        )
        if not corrected_code:
            raise ValueError("LLM failed to generate corrected code.")
        return corrected_code

    def _summarize_results(
        self,
        natural_language_query: str,
        executed_query_text: str,
        results_df: pd.DataFrame | None
    ) -> str:
        """Summarizes the execution results."""
        result_summary_for_prompt = ""
        if results_df is None or results_df.empty:
            result_summary_for_prompt = "The query returned no results or an error occurred before results could be obtained."
        elif len(results_df) <= 5:
            result_summary_for_prompt = f"The query returned the following results:\n{results_df.to_string()}"
        else:
            result_summary_for_prompt = f"The query returned {len(results_df)} rows. Here are the first 5:\n{results_df.head().to_string()}\n...and {len(results_df)-5} more rows."

        summary_prompt_parts = [
            f"Based on the user's question '{natural_language_query}', the executed query/code '{executed_query_text}', and the following query results, provide a concise natural language answer:",
            result_summary_for_prompt,
            "\nNatural Language Answer:"
        ]
        summary_prompt = "\n".join(summary_prompt_parts)
        # print(f"DEBUG: Summary prompt:\n{summary_prompt}")

        nl_summary = self.llm_provider.generate_summary(
            prompt=summary_prompt,
            model_name=self._get_config_value("summary_model", self.default_summary_model)
        )
        return nl_summary if nl_summary else "LLM did not provide a natural language summary."


    def execute(
        self,
        natural_language_query: str,
        metadata: Dict[str, Any],
        agent_type: str,
        file_path: str,
        table_name: str, # This is the table_name from metadata (e.g. R object name, SQL table name)
        original_uploaded_filename: str, # Needed to determine if SQL is against SQLite
        **kwargs: Any # To catch any other params like llm_model (for specific step)
    ) -> Dict[str, Any]:

        executed_code = ""
        results_df = None
        error_msg = None
        nl_response = "Could not generate a response."
        intermediate_steps = []

        # Determine table name for query context (used by execution helpers)
        # For R, this is the object name. For SQL (CSV/Parquet), it's the table alias.
        # For Python, it's the DataFrame variable name.
        # 'table_name' argument to this method should be this context-specific name.

        try:
            # Step 1: Generate initial code
            intermediate_steps.append({"step": "Generate Initial Code", "query": natural_language_query})
            generated_code = self._generate_initial_code(natural_language_query, metadata, agent_type, table_name)
            executed_code = generated_code
            intermediate_steps[-1]["generated_code"] = generated_code
            # print(f"DEBUG: Initial generated code ({agent_type}):\n{generated_code}")

            # Step 2: Execute initial code
            intermediate_steps.append({"step": "Execute Initial Code", "code": generated_code})
            results_df, error_msg = self._execute_generated_code(generated_code, agent_type, file_path, table_name, original_uploaded_filename)
            intermediate_steps[-1]["execution_error"] = error_msg
            intermediate_steps[-1]["execution_successful"] = error_msg is None
            # print(f"DEBUG: Initial execution result - Error: {error_msg}, DF empty: {results_df.empty if results_df is not None else 'N/A'}")


            # Step 3 & 4: Reflect and correct if error
            if error_msg:
                intermediate_steps.append({
                    "step": "Attempt Correction",
                    "failed_code": generated_code,
                    "error_message": error_msg
                })
                # print(f"DEBUG: Attempting correction for error: {error_msg}")
                corrected_code = self._generate_corrected_code(
                    natural_language_query, metadata, agent_type, table_name, generated_code, error_msg
                )
                executed_code = corrected_code # Update to the last executed code
                intermediate_steps[-1]["corrected_code"] = corrected_code
                # print(f"DEBUG: Corrected code ({agent_type}):\n{corrected_code}")

                intermediate_steps.append({"step": "Execute Corrected Code", "code": corrected_code})
                results_df, error_msg = self._execute_generated_code(corrected_code, agent_type, file_path, table_name, original_uploaded_filename)
                intermediate_steps[-1]["execution_error"] = error_msg
                intermediate_steps[-1]["execution_successful"] = error_msg is None
                # print(f"DEBUG: Corrected execution result - Error: {error_msg}, DF empty: {results_df.empty if results_df is not None else 'N/A'}")


            # Step 5: Summarize results (if no critical error before this point)
            if not error_msg : # Or if results_df is not None, depending on desired behavior for empty results
                intermediate_steps.append({"step": "Summarize Results"})
                try:
                    nl_response = self._summarize_results(natural_language_query, executed_code, results_df)
                    intermediate_steps[-1]["summary"] = nl_response
                except Exception as e_summary:
                    # print(f"DEBUG: Error during summarization: {str(e_summary)}")
                    nl_response = f"Successfully executed code, but failed to generate summary: {str(e_summary)}"
                    intermediate_steps[-1]["summary_error"] = str(e_summary)
            else:
                # If there's still an error after correction (or initial error if no correction attempted/succeeded)
                nl_response = f"An error occurred: {error_msg}"
                intermediate_steps.append({"step": "Final Error", "message": error_msg})


        except Exception as e:
            # print(f"DEBUG: Critical error in topology execution: {str(e)}")
            error_msg = f"Critical error in SequentialReflectTopology: {str(e)}"
            nl_response = error_msg # Ensure this error is propagated
            intermediate_steps.append({"step": "Critical Topology Error", "error_message": str(e)})


        return {
            'executed_query_text': executed_code,
            'results': results_df.to_dict(orient='records') if results_df is not None else [],
            'error': error_msg,
            'natural_language_response': nl_response,
            'intermediate_steps': intermediate_steps
        }

    # --- Prompt Construction Helpers (similar to what was in app.py) ---
    def _construct_initial_prompt(self, natural_language_query: str, metadata: Dict[str, Any], agent_type: str, table_name: str) -> str:
        is_sqlite = metadata.get('file_type') == 'sqlite' # Assuming file_type is in metadata

        if agent_type == 'sql':
            prompt_parts = ["Given the table schema below and the user question, generate a valid SQL query to answer the question."]
            prompt_parts.append(f"Table Name: {table_name}")
            if metadata.get('columns'):
                prompt_parts.append("Columns:")
                for column in metadata['columns']:
                    prompt_parts.append(f"- {column['name']} ({column.get('type', 'UNKNOWN')})")
            elif not is_sqlite:
                 prompt_parts.append("Columns: (Schema not fully provided. Ensure your query references correct table and column names.)")
            prompt_parts.append(f"\nUser Question: {natural_language_query}")
            prompt_parts.append("SQL Query:")
            return "\n".join(prompt_parts)

        elif agent_type == 'r_datatable':
            prompt_parts = [
                "You are an R programming assistant. Generate R code using the `data.table` package to answer the user's question.",
                f"The data is loaded into an R data.table object named `{table_name}` (originally from the Rdata file).", # table_name is the R object name
                "Ensure the final result of the operations is assigned back to this same variable `active_df`.",
                "Columns in the data.table `active_df` (which is `{table_name}` from the file):"
            ]
            if metadata.get('columns'):
                for column in metadata['columns']:
                    prompt_parts.append(f"- {column['name']} (R type might be {column.get('type', 'UNKNOWN')})") # Type info might help
            else:
                prompt_parts.append("(Column details not fully provided; infer from context if necessary)")
            prompt_parts.extend([
                f"\nUser Question: {natural_language_query}",
                "Generate only the R `data.table` code that performs the query on `active_df` and assigns the result back to `active_df`.",
                "For example: active_df <- active_df[some_condition == TRUE, .(new_col = sum(another_col, na.rm = TRUE))]",
                "R data.table Code (assign to active_df):"
            ])
            return "\n".join(prompt_parts)

        elif agent_type == 'python_pandas':
            prompt_parts = [
                "You are a Python programming assistant. Generate Python code using the Pandas library to answer the user's question.",
                f"The data is loaded into a Pandas DataFrame variable named `{table_name}`.", # table_name is the df variable name
                "Ensure the final result of the operations is assigned back to this same DataFrame variable.",
                "Columns in the DataFrame:"
            ]
            if metadata.get('columns'):
                for column in metadata['columns']:
                    prompt_parts.append(f"- {column['name']} (dtype: {column.get('type', 'object')})")
            else:
                prompt_parts.append("(Column details not fully provided; infer from context if necessary)")
            prompt_parts.extend([
                f"\nUser Question: {natural_language_query}",
                f"Generate only the Python Pandas code that operates on the DataFrame named `{table_name}` and assigns the result back to the *same* variable `{table_name}`.",
                f"For example: {table_name} = {table_name}[{table_name}['some_column'] > 10]",
                "Python Pandas Code:"
            ])
            return "\n".join(prompt_parts)
        else:
            raise ValueError(f"Unsupported agent_type for prompt construction: {agent_type}")

    def _construct_correction_prompt(
        self, natural_language_query: str, metadata: Dict[str, Any], agent_type: str,
        table_name: str, failed_code: str, error_message: str
    ) -> str:
        is_sqlite = metadata.get('file_type') == 'sqlite'

        if agent_type == 'sql':
            prompt_parts = ["The following SQL query resulted in an error. Please correct it."]
            prompt_parts.append(f"Original Question: {natural_language_query}")
            prompt_parts.append(f"Table Name: {table_name}")
            if metadata.get('columns'):
                prompt_parts.append("Columns:")
                for column in metadata['columns']:
                    prompt_parts.append(f"- {column['name']} ({column.get('type', 'UNKNOWN')})")
            elif not is_sqlite:
                 prompt_parts.append("Columns: (Schema not fully provided or embedded)")
            prompt_parts.extend([f"Failed SQL: {failed_code}", f"Error Message: {error_message}", "Corrected SQL Query:"])
            return "\n".join(prompt_parts)

        elif agent_type == 'r_datatable':
            prompt_parts = [
                "The following R data.table code resulted in an error. Please correct it.",
                f"The data is in a data.table named `active_df` (derived from object `{table_name}` in the Rdata file).",
                "Ensure the corrected code assigns the result back to `active_df`.",
                f"Original Question: {natural_language_query}",
                "Columns in `active_df`:"
            ]
            if metadata.get('columns'):
                for column in metadata['columns']:
                     prompt_parts.append(f"- {column['name']} (R type might be {column.get('type', 'UNKNOWN')})")
            prompt_parts.extend([
                f"Failed R Code:\n{failed_code}",
                f"Error Message: {error_message}",
                "Corrected R data.table Code (assign to active_df):"
            ])
            return "\n".join(prompt_parts)

        elif agent_type == 'python_pandas':
            prompt_parts = [
                "The following Python Pandas code resulted in an error. Please correct it.",
                f"The data is in a Pandas DataFrame named `{table_name}`.",
                f"Ensure the corrected code assigns the result back to the *same* DataFrame variable `{table_name}`.",
                f"Original Question: {natural_language_query}",
                "Columns in DataFrame:"
            ]
            if metadata.get('columns'):
                for column in metadata['columns']:
                    prompt_parts.append(f"- {column['name']} (dtype: {column.get('type', 'object')})")
            prompt_parts.extend([
                f"Failed Python Code:\n{failed_code}",
                f"Error Message: {error_message}",
                f"Corrected Python Pandas Code (assign result back to `{table_name}`):"
            ])
            return "\n".join(prompt_parts)
        else:
            raise ValueError(f"Unsupported agent_type for correction prompt: {agent_type}")
