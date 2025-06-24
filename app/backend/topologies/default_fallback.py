import pandas as pd
from typing import Dict, Any, List, Tuple

from app.backend.llm_providers.base import LLMProvider
from app.backend.topologies.base import Topology
from .parallel_ensemble import ParallelEnsembleTopology
from .sequential_reflect import SequentialReflectTopology

class DefaultFallbackTopology(Topology):
    """
    A default fallback topology that combines parallelism and sequential reflection.
    1. Tries to generate and execute code using a ParallelEnsembleTopology.
    2. If the parallel approach fails to get an executable result,
       it falls back to using a SequentialReflectTopology with the initial query
       (or potentially with the best failed attempt from parallel phase).
    3. If parallel succeeded but `refine_summary_after_parallel` is true,
       it uses SequentialReflectTopology's summarizer for the final summary.
    """

    def __init__(self, llm_provider: LLMProvider, topology_config: Dict[str, Any] | None = None):
        super().__init__(llm_provider, topology_config)

        # Config for the parallel phase
        parallel_config = self._get_config_value("parallel_config", {}).copy() # Use copy to modify safely

        # Determine provider type to set appropriate default parallel models
        # This is a simple check based on class name. More robust checks might be needed if class names change.
        provider_class_name = llm_provider.__class__.__name__
        default_parallel_code_gen_models = ["gpt-3.5-turbo", "openai/gpt-4-turbo-preview"] # Default to OpenAI models
        if "GeminiLLMProvider" in provider_class_name:
            default_parallel_code_gen_models = ["gemini/gemini-1.5-flash", "gemini/gemini-pro"]
        elif "LocalHFLMMProvider" in provider_class_name:
            # For local, it's harder to set a generic default; user should configure this.
            # Or, the LocalHFConfig could carry a preferred model name.
            # For now, let's assume user configures or it defaults to something generic that might work if server has it.
            default_parallel_code_gen_models = ["ollama/mistral", "ollama/llama2"] # Example if ollama-based

        parallel_config.setdefault("code_gen_models", default_parallel_code_gen_models)
        parallel_config.setdefault("summary_model", self._get_config_value("summary_model", "gpt-3.5-turbo")) # Default summary model
        self.parallel_topology = ParallelEnsembleTopology(llm_provider, parallel_config)

        # Config for the sequential fallback/reflection phase
        sequential_config = self._get_config_value("sequential_config", {}).copy()
        # Default sequential model potentially stronger
        default_sequential_code_model = "openai/gpt-4-turbo-preview"
        if "GeminiLLMProvider" in provider_class_name:
            default_sequential_code_model = "gemini/gemini-1.5-pro-latest"

        sequential_config.setdefault("code_gen_model", self._get_config_value("code_gen_model", default_sequential_code_model))
        sequential_config.setdefault("correction_model", sequential_config["code_gen_model"]) # Use same as code_gen by default
        sequential_config.setdefault("summary_model", self._get_config_value("summary_model", "gpt-3.5-turbo")) # Default summary model
        self.sequential_topology = SequentialReflectTopology(llm_provider, sequential_config)

        self.refine_summary_after_parallel: bool = self._get_config_value("refine_summary_after_parallel", False)
        # Determines if we re-summarize even if parallel part was fully successful.

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

        intermediate_steps = [{"step": "DefaultFallbackTopology Initiated"}]
        final_result: Dict[str, Any] = {
            'executed_query_text': "", 'results': [], 'error': "Topology did not complete.",
            'natural_language_response': "Failed to get a response.",
            'intermediate_steps': intermediate_steps
        }

        # --- Phase 1: Attempt with ParallelEnsembleTopology ---
        intermediate_steps.append({"step": "Attempting Parallel Ensemble Execution"})
        try:
            parallel_result = self.parallel_topology.execute(
                natural_language_query, metadata, agent_type, file_path, table_name, original_uploaded_filename, **kwargs
            )
            intermediate_steps.append({
                "step": "Parallel Ensemble Result",
                "success": parallel_result.get('error') is None,
                "output": parallel_result
            })

            if parallel_result.get('error') is None:
                # Parallel approach succeeded
                final_result = parallel_result # Use this as the base

                if self.refine_summary_after_parallel and final_result.get('results') is not None:
                    intermediate_steps.append({"step": "Refining Summary (Post-Parallel Success)"})
                    try:
                        refined_summary = self.sequential_topology._summarize_results(
                            natural_language_query,
                            final_result['executed_query_text'],
                            pd.DataFrame(final_result['results']) # Recreate DF for summarizer
                        )
                        final_result['natural_language_response'] = refined_summary
                        intermediate_steps[-1]['refined_summary'] = refined_summary
                    except Exception as e_refine:
                        intermediate_steps[-1]['refinement_error'] = str(e_refine)
                        # Keep original summary if refinement fails

                final_result['intermediate_steps'] = intermediate_steps + (parallel_result.get('intermediate_steps', []))
                return final_result

            else:
                # Parallel approach failed to get a clean execution.
                # We will proceed to sequential fallback.
                intermediate_steps.append({
                    "step": "Parallel Ensemble Failed, Proceeding to Sequential Fallback",
                    "parallel_error": parallel_result.get('error')
                })
                # We might want to use the best code attempt from parallel phase for sequential correction,
                # but for simplicity now, sequential will start fresh or with its own initial generation.
                # The current SequentialReflectTopology starts code gen from scratch.

        except Exception as e_parallel:
            intermediate_steps.append({
                "step": "Critical Error during Parallel Ensemble Execution",
                "error_message": str(e_parallel),
                "proceeding_to_sequential": True
            })
            # Continue to sequential fallback even if parallel had a critical error

        # --- Phase 2: Attempt with SequentialReflectTopology (Fallback) ---
        intermediate_steps.append({"step": "Attempting Sequential Reflection Execution (Fallback)"})
        try:
            # Pass any specific model prefs from kwargs if they exist for sequential part
            sequential_kwargs = kwargs.copy()
            # Allow overriding sequential_topology's default models if specified in main call's kwargs
            # e.g. kwargs might contain "sequential_code_gen_model"
            # This requires SequentialReflectTopology to check its own config for these overrides.
            # For now, it uses its init-time config.

            sequential_result = self.sequential_topology.execute(
                natural_language_query, metadata, agent_type, file_path, table_name, original_uploaded_filename, **sequential_kwargs
            )
            intermediate_steps.append({
                "step": "Sequential Reflection Result",
                "success": sequential_result.get('error') is None,
                "output": sequential_result # Contains its own intermediate steps
            })
            # The sequential_result is the final result if we reach here.
            final_result = sequential_result
            # Prepend the DefaultFallbackTopology's own steps to the sequential ones
            final_result['intermediate_steps'] = intermediate_steps + (sequential_result.get('intermediate_steps', []))
            return final_result

        except Exception as e_sequential:
            intermediate_steps.append({
                "step": "Critical Error during Sequential Reflection Execution (Fallback)",
                "error_message": str(e_sequential)
            })
            final_result['error'] = f"Fallback sequential topology also failed critically: {str(e_sequential)}"
            final_result['natural_language_response'] = final_result['error']
            final_result['intermediate_steps'] = intermediate_steps
            return final_result

        # Should not be reached if logic is correct
        return final_result
