import pandas as pd
import json # For parsing LLM's structured output (thought/action)
from typing import Dict, Any, List, Tuple, Callable

from app.backend.llm_providers.base import LLMProvider
from app.backend.topologies.base import Topology
from app.backend.code_execution import (
    execute_duckdb_query,
    execute_r_script,
    execute_python_pandas_code
)
from .sequential_reflect import SequentialReflectTopology # For _summarize_results and prompt helpers

class IterativeReasonActTopology(Topology):
    """
    Implements an iterative reasoning and action topology (like a ReAct agent).
    1. LLM is prompted to "think" and decide an "action" based on the query and history.
    2. Actions can be: generate_code, execute_code, provide_final_answer.
    3. The chosen action is performed.
    4. The result ("observation") is fed back into the LLM for the next iteration.
    5. Loop continues for max_iterations or until provide_final_answer.
    """

    def __init__(self, llm_provider: LLMProvider, topology_config: Dict[str, Any] | None = None):
        super().__init__(llm_provider, topology_config)
        self.max_iterations: int = self._get_config_value("max_iterations", 5)
        self.reason_act_model: str = self._get_config_value("reason_act_model", "gpt-4-turbo-preview") # Needs strong model
        self.code_gen_model: str = self._get_config_value("code_gen_model", self.reason_act_model) # Default to same
        self.summary_model: str = self._get_config_value("summary_model", self.reason_act_model)

        # Helper for code execution and summarization (can also provide prompt construction)
        helper_config = {
            "code_gen_model": self.code_gen_model, # Used by helper's _generate_initial_code
            "summary_model": self.summary_model
        }
        self._helper_sequential_topology = SequentialReflectTopology(llm_provider, helper_config)

    def _parse_llm_response_for_action(self, llm_output: str) -> Dict[str, Any] | None:
        """
        Parses the LLM's output to extract thought and action.
        Expects a JSON block like:
        ```json
        {
            "thought": "I need to first understand what data is available...",
            "action": {
                "tool_name": "generate_code", // or "execute_code", "provide_final_answer"
                "tool_input": {
                    "language": "sql", // if generate_code or execute_code
                    "query_goal": "Get count of rows", // if generate_code
                    "code": "SELECT COUNT(*) FROM table", // if execute_code
                    "summary_text": "The total count is X" // if provide_final_answer
                }
            }
        }
        ```
        """
        try:
            # Try to find JSON block if LLM adds surrounding text
            json_start = llm_output.find("```json")
            json_end = llm_output.rfind("```")

            if json_start != -1 and json_end != -1 and json_start < json_end:
                json_str = llm_output[json_start + 7 : json_end].strip() # +7 for "```json\n"
            else: # Assume the whole output is JSON or try direct parse
                json_str = llm_output

            parsed = json.loads(json_str)
            if "action" in parsed and "tool_name" in parsed["action"]:
                return parsed
            return None
        except json.JSONDecodeError as e:
            print(f"WARN: Failed to parse LLM action JSON: {e}. Output was: {llm_output}")
            return None # Or could try a more lenient parsing / retry with LLM

    def _execute_action(
        self,
        action: Dict[str, Any],
        current_context: Dict[str, Any] # Contains metadata, file_path, table_name etc.
    ) -> Tuple[str, Any | None]: # (observation_text, result_data_for_internal_use)
        """
        Executes the action decided by the LLM.
        `current_context` holds natural_language_query, metadata, agent_type, file_path, table_name, original_uploaded_filename
        """
        tool_name = action.get("tool_name")
        tool_input = action.get("tool_input", {})

        agent_type = current_context["agent_type"] # sql, python_pandas, r_datatable
        metadata = current_context["metadata"]
        table_name = current_context["table_name"] # The contextual table/df/object name
        file_path = current_context["file_path"]
        original_uploaded_filename = current_context["original_uploaded_filename"]

        if tool_name == "generate_code":
            language = tool_input.get("language", agent_type) # Default to main agent_type
            query_goal = tool_input.get("query_goal", current_context["natural_language_query"]) # Use sub-goal if provided

            # Use helper's prompt construction
            prompt = self._helper_sequential_topology._construct_initial_prompt(
                query_goal, metadata, language, table_name
            )
            generated_code = self.llm_provider.generate_code(prompt, model_name=self.code_gen_model)
            if generated_code:
                return f"Generated {language} code:\n{generated_code}", {"type": "code", "language": language, "code": generated_code}
            else:
                return "Error: LLM failed to generate code for the sub-task.", None

        elif tool_name == "execute_code":
            code_to_execute = tool_input.get("code")
            language_of_code = tool_input.get("language", agent_type)
            if not code_to_execute:
                return "Error: No code provided for execution.", None

            df_results, error_msg = self._helper_sequential_topology._execute_generated_code(
                code_to_execute, language_of_code, file_path, table_name, original_uploaded_filename
            )
            if error_msg:
                return f"Error executing code: {error_msg}", {"type": "execution_error", "error": error_msg, "executed_code": code_to_execute}
            else:
                # Limit data preview for observation
                preview = df_results.head().to_string() if df_results is not None else "No results"
                if df_results is not None and len(df_results) > 5:
                    preview += f"\n...and {len(df_results)-5} more rows."
                return f"Code executed. Result preview:\n{preview}", {"type": "execution_result", "dataframe": df_results, "executed_code": code_to_execute}

        elif tool_name == "provide_final_answer":
            summary = tool_input.get("summary_text", "LLM decided to provide a final answer without detailed summary.")
            # This action signals termination. The summary is the observation.
            return f"Final Answer: {summary}", {"type": "final_answer", "summary": summary}

        else:
            return f"Error: Unknown tool_name '{tool_name}'. Valid tools are: generate_code, execute_code, provide_final_answer.", None


    def _build_iteration_prompt(self, natural_language_query: str, metadata: Dict[str, Any], agent_type: str, history: List[Dict[str,str]]) -> str:
        # History items are like: {"thought": "...", "action_taken": "{...}", "observation": "..."}

        prompt = f"You are an expert data analyst assisting a user. Your goal is to answer the user's question about their data.\n"
        prompt += f"User's overall question: {natural_language_query}\n"
        prompt += f"Data agent type to use for code: {agent_type}\n"
        prompt += "Schema of the primary table/dataframe:\n"
        prompt += f"Table/DataFrame Name: {metadata.get('table_name', 'unknown_table')}\nColumns:\n"
        for col in metadata.get('columns', []):
            prompt += f"- {col['name']} ({col.get('type', 'UNKNOWN')})\n"

        prompt += "\nPrevious steps in your reasoning process:\n"
        if not history:
            prompt += "No steps taken yet. This is the first step.\n"
        else:
            for i, entry in enumerate(history):
                prompt += f"--- Step {i+1} ---\n"
                prompt += f"Thought: {entry.get('thought', 'N/A')}\n"
                prompt += f"Action Taken: {entry.get('action_taken_str', 'N/A')}\n" # String representation of action
                prompt += f"Observation: {entry.get('observation', 'N/A')}\n"

        prompt += "\nBased on the user's question and the history of your actions and observations, carefully consider your next step.\n"
        prompt += "Your available tools are:\n"
        prompt += "1. `generate_code`: Use this to generate code (SQL, Python Pandas, R data.table) to query or manipulate data. Specify 'language' and 'query_goal' (a sub-question or description of what the code should achieve).\n"
        prompt += "2. `execute_code`: Use this to run previously generated code. Specify 'language' and 'code'.\n"
        prompt += "3. `provide_final_answer`: Use this when you have enough information to answer the user's overall question. Provide the final answer in 'summary_text'.\n"

        prompt += "\nRespond with a JSON object containing your 'thought' (your reasoning) and your chosen 'action' (including 'tool_name' and 'tool_input').\n"
        prompt += "Example for generate_code: {\"thought\": \"I need to count rows.\", \"action\": {\"tool_name\": \"generate_code\", \"tool_input\": {\"language\": \"sql\", \"query_goal\": \"Count all rows in the table\"}}}\n"
        prompt += "Example for execute_code: {\"thought\": \"The SQL for counting rows is generated, now I need to run it.\", \"action\": {\"tool_name\": \"execute_code\", \"tool_input\": {\"language\": \"sql\", \"code\": \"SELECT COUNT(*) FROM test_table;\"}}}\n"
        prompt += "Example for provide_final_answer: {\"thought\": \"I have the count and can now answer the user.\", \"action\": {\"tool_name\": \"provide_final_answer\", \"tool_input\": {\"summary_text\": \"The table contains 150 rows.\"}}}\n"
        prompt += "\nYour JSON response:\n```json\n"
        # LLM should complete starting from here with its JSON output.
        return prompt

    def execute(
        self,
        natural_language_query: str,
        metadata: Dict[str, Any],
        agent_type: str,
        file_path: str,
        table_name: str, # Contextual table/df/object name for execution
        original_uploaded_filename: str,
        **kwargs: Any
    ) -> Dict[str, Any]:

        history: List[Dict[str, str]] = [] # Stores thought, action_str, observation for each step
        iteration_results: List[Any] = [] # Stores any data results from execute_code actions

        current_executed_code: str = "" # Last successfully executed code snippet by this topology
        final_summary_from_llm: str = "No final answer reached by the agent."
        final_error_msg: str | None = None
        action: Dict[str, Any] = {} # Initialize action to an empty dict

        current_context = {
            "natural_language_query": natural_language_query, "metadata": metadata, "agent_type": agent_type,
            "file_path": file_path, "table_name": table_name, "original_uploaded_filename": original_uploaded_filename
        }

        for i in range(self.max_iterations):
            iteration_prompt = self._build_iteration_prompt(natural_language_query, metadata, agent_type, history)

            try:
                llm_response_str = self.llm_provider.generate_text(
                    prompt=iteration_prompt,
                    model_name=self.reason_act_model,
                    max_tokens=500 # Allow more tokens for thought + JSON action
                )
            except Exception as e_llm:
                final_error_msg = f"LLM call failed during iteration {i+1}: {e_llm}"
                history.append({"thought": "LLM call failed.", "action_taken_str": "N/A", "observation": final_error_msg})
                break

            parsed_action_block = self._parse_llm_response_for_action(llm_response_str)

            if not parsed_action_block:
                final_error_msg = f"Failed to parse LLM action in iteration {i+1}. LLM Raw: {llm_response_str}"
                history.append({"thought": "Failed to parse own action.", "action_taken_str": "N/A", "observation": final_error_msg})
                break # Critical failure

            thought = parsed_action_block.get("thought", "No thought provided.")
            action = parsed_action_block.get("action", {})
            action_taken_str = json.dumps(action) # For history/logging

            observation_text, action_result_data = self._execute_action(action, current_context)
            history.append({"thought": thought, "action_taken_str": action_taken_str, "observation": observation_text})

            if action_result_data:
                if action_result_data.get("type") == "execution_result" and action_result_data.get("dataframe") is not None:
                    iteration_results.append(action_result_data["dataframe"]) # Store the dataframe
                    current_executed_code = action_result_data.get("executed_code", current_executed_code)
                elif action_result_data.get("type") == "execution_error":
                    # If an execution error occurs, it's part of the observation. The LLM should decide what to do next.
                    # We don't break the loop here unless the LLM decides to give up.
                    current_executed_code = action_result_data.get("executed_code", current_executed_code) # Log code that failed
                elif action_result_data.get("type") == "final_answer":
                    # The observation_text already contains "Final Answer: {summary}"
                    final_summary_from_llm = observation_text
                    final_error_msg = None # Clear any previous transient errors
                    break # Terminate loop

            if i == self.max_iterations - 1 and action.get("tool_name") != "provide_final_answer":
                final_error_msg = "Agent reached maximum iterations without providing a final answer."
                # final_summary_from_llm will remain its default or last value if not explicitly set by provide_final_answer
                # If it's still the default "No final answer reached...", the error message is better.
                if final_summary_from_llm == "No final answer reached by the agent.":
                    final_summary_from_llm = final_error_msg


        # Determine final results_df: usually the last one from iteration_results if any, or empty
        final_results_df = iteration_results[-1] if iteration_results and isinstance(iteration_results[-1], pd.DataFrame) else pd.DataFrame()

        # Consolidate the final natural language response
        # If an error message exists at the end, it usually takes precedence unless a final_answer action occurred.
        # final_summary_from_llm should hold the correct response based on the loop's outcome.
        # If the loop ended due to an error that wasn't overridden by a "provide_final_answer", make sure it's reflected.
        effective_nl_response = final_summary_from_llm
        if final_error_msg and not (action and action.get("tool_name") == "provide_final_answer"):
            # If there's an error and the last action wasn't provide_final_answer, the error is the response.
             effective_nl_response = final_error_msg


        return {
            'executed_query_text': current_executed_code,
            'results': final_results_df.to_dict(orient='records'),
            'error': final_error_msg,
            'natural_language_response': effective_nl_response,
            'intermediate_steps': history
        }
