import pandas as pd
from typing import Dict, Any, List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.backend.llm_providers.base import LLMProvider
from app.backend.topologies.base import Topology
from app.backend.code_execution import (
    execute_duckdb_query,
    execute_r_script,
    execute_python_pandas_code
)
# Import SequentialReflectTopology to reuse its prompt construction and summarization logic
from .sequential_reflect import SequentialReflectTopology


class ParallelEnsembleTopology(Topology):
    """
    Implements a parallel ensemble execution topology.
    1. Takes a list of LLM model configurations (e.g., different model names).
    2. Generates code in parallel for the natural language query using these configurations.
    3. Attempts to execute the generated codes in the order of generation preference or all in parallel.
    4. The first code that executes successfully is chosen.
    5. If none execute successfully, an error is returned.
    6. Summarizes the results of the successfully executed code using LLM.
    """

    def __init__(self, llm_provider: LLMProvider, topology_config: Dict[str, Any] | None = None):
        super().__init__(llm_provider, topology_config)
        # Configurable list of model names to try in parallel for code generation
        self.code_gen_models: List[str] = self._get_config_value(
            "code_gen_models",
            # Provide a sensible default list if not configured
            ["gpt-4-turbo-preview", "gpt-3.5-turbo", "gemini-1.5-pro-latest"]
        )
        self.summary_model: str = self._get_config_value("summary_model", "gpt-3.5-turbo")
        self.max_workers: int = self._get_config_value("max_workers", len(self.code_gen_models))

        # For prompt construction and summarization, we can leverage methods from SequentialReflectTopology
        # by instantiating it with the same LLM provider. This avoids code duplication.
        # The config for this internal helper should ensure it uses the correct summary model.
        helper_topology_config = {"summary_model": self.summary_model}
        self._helper_sequential_topology = SequentialReflectTopology(llm_provider, helper_topology_config)


    def _generate_code_parallel(
        self,
        natural_language_query: str,
        metadata: Dict[str, Any],
        agent_type: str,
        table_name: str
    ) -> List[Tuple[str, str | None]]: # List of (model_name, generated_code or None if error)
        """Generates code in parallel using configured models."""
        futures = []
        results: List[Tuple[str, str | None]] = []

        prompt = self._helper_sequential_topology._construct_initial_prompt(
            natural_language_query, metadata, agent_type, table_name
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for model_name in self.code_gen_models:
                futures.append(
                    executor.submit(
                        self.llm_provider.generate_code,
                        prompt=prompt,
                        model_name=model_name
                        # TODO: Allow per-model temperature/max_tokens via config if needed
                    )
                )

            for i, future in enumerate(as_completed(futures)):
                model_used = self.code_gen_models[futures.index(future)] # Get model name based on future index
                try:
                    code = future.result()
                    if code:
                        results.append((model_used, code.strip()))
                    else:
                        results.append((model_used, None)) # LLM returned no code
                except Exception as e:
                    print(f"Error generating code with {model_used}: {e}")
                    results.append((model_used, None)) # Error during generation
        return results

    def _execute_generated_code(
        self,
        generated_code: str,
        agent_type: str,
        file_path: str,
        table_name_for_query: str,
        original_uploaded_filename: str
    ) -> Tuple[pd.DataFrame | None, str | None]:
        """Wraps the execution logic (borrowed from SequentialReflectTopology for consistency)."""
        return self._helper_sequential_topology._execute_generated_code(
            generated_code, agent_type, file_path, table_name_for_query, original_uploaded_filename
        )

    def execute(
        self,
        natural_language_query: str,
        metadata: Dict[str, Any],
        agent_type: str,
        file_path: str,
        table_name: str,
        original_uploaded_filename: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        intermediate_steps = []

        # Step 1: Generate code in parallel
        intermediate_steps.append({"step": "Parallel Code Generation", "models_tried": self.code_gen_models})
        parallel_code_gen_results = self._generate_code_parallel(
            natural_language_query, metadata, agent_type, table_name
        )
        intermediate_steps[-1]["generation_outputs"] = [
            {"model": r[0], "code_generated": bool(r[1]), "code_sample": (r[1][:100] + "..." if r[1] else "None")}
            for r in parallel_code_gen_results
        ]

        # Filter out None results (errors during generation or empty returns)
        valid_generated_codes = [(model, code) for model, code in parallel_code_gen_results if code]

        if not valid_generated_codes:
            error_msg = "All LLM configurations failed to generate code."
            intermediate_steps.append({"step": "Error", "message": error_msg})
            return {
                'executed_query_text': "", 'results': [], 'error': error_msg,
                'natural_language_response': error_msg, 'intermediate_steps': intermediate_steps
            }

        # Step 2: Attempt to execute generated codes sequentially (or could be parallel too)
        # For now, try executing one by one, taking the first success.
        # The order in valid_generated_codes depends on completion order from ThreadPoolExecutor,
        # which might not be the same as self.code_gen_models if some models are much faster.
        # If order of trial matters (e.g. prefer cheaper models first if they succeed),
        # sort valid_generated_codes based on self.code_gen_models preference.

        # Simple sort to match original preference order for trying execution:
        preference_order = {model_name: i for i, model_name in enumerate(self.code_gen_models)}
        sorted_codes_to_try = sorted(valid_generated_codes, key=lambda x: preference_order.get(x[0], float('inf')))

        intermediate_steps.append({"step": "Attempting Execution of Generated Codes", "order": [c[0] for c in sorted_codes_to_try]})

        executed_code = ""
        results_df = None
        final_error_msg = "All generated codes failed to execute." # Default if all fail

        for model_attempted, code_to_execute in sorted_codes_to_try:
            exec_step_info = {"sub_step": f"Execute code from {model_attempted}", "code": code_to_execute}
            intermediate_steps.append(exec_step_info)
            # print(f"DEBUG: Attempting to execute code from {model_attempted}:\n{code_to_execute}")

            current_results_df, current_error_msg = self._execute_generated_code(
                code_to_execute, agent_type, file_path, table_name, original_uploaded_filename
            )
            exec_step_info["execution_error"] = current_error_msg
            exec_step_info["execution_successful"] = current_error_msg is None

            if current_error_msg is None:
                # print(f"DEBUG: Successful execution with model {model_attempted}")
                executed_code = code_to_execute
                results_df = current_results_df
                final_error_msg = None # Success!
                exec_step_info["chosen_for_summary"] = True
                break # Stop on first successful execution
            else:
                # print(f"DEBUG: Failed execution with model {model_attempted}, Error: {current_error_msg}")
                final_error_msg = f"Error from {model_attempted}: {current_error_msg}" # Keep last error

        # Step 3: Summarize if successful
        nl_response = "Could not generate a response."
        if final_error_msg is None and results_df is not None:
            summary_step_info = {"step": "Summarize Results", "executed_code_from_model": model_attempted} # type: ignore
            intermediate_steps.append(summary_step_info)
            try:
                # Use the helper topology's summarize method
                nl_response = self._helper_sequential_topology._summarize_results(
                    natural_language_query, executed_code, results_df
                )
                summary_step_info["summary"] = nl_response
            except Exception as e_summary:
                nl_response = f"Successfully executed code, but failed to generate summary: {str(e_summary)}"
                summary_step_info["summary_error"] = str(e_summary)
        elif final_error_msg:
            nl_response = f"An error occurred: {final_error_msg}"
            # Ensure the last error is captured if no summary happens
            if not any(s.get('step') == "Error" for s in intermediate_steps): # Avoid duplicate "Error" step
                 intermediate_steps.append({"step": "Final Error", "message": final_error_msg})


        return {
            'executed_query_text': executed_code,
            'results': results_df.to_dict(orient='records') if results_df is not None else [],
            'error': final_error_msg,
            'natural_language_response': nl_response,
            'intermediate_steps': intermediate_steps
        }
