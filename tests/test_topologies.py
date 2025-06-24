import pytest
from abc import ABC, abstractmethod
from typing import Dict, Any
from app.backend.topologies.base import Topology
from app.backend.llm_providers.base import LLMProvider # Required for Topology init

# Dummy LLMProvider for testing Topology interface
class DummyLLMProvider(LLMProvider):
    def generate_text(self, prompt: str, model_name: str, temperature: float = 0.1, max_tokens: int = 150, **kwargs) -> str: return "dummy text"
    def generate_code(self, prompt: str, model_name: str, temperature: float = 0.1, max_tokens: int = 200, **kwargs) -> str: return "dummy code"
    def generate_summary(self, prompt: str, model_name: str, temperature: float = 0.3, max_tokens: int = 200, **kwargs) -> str: return "dummy summary"

@pytest.fixture
def dummy_llm_provider():
    return DummyLLMProvider()

# Test that Topology is an Abstract Base Class
def test_topology_is_abc():
    assert issubclass(Topology, ABC)

# Test that the 'execute' method is abstract
def test_topology_abstract_methods():
    expected_abstract_methods = {"execute"}
    actual_abstract_methods = Topology.__abstractmethods__
    assert actual_abstract_methods == expected_abstract_methods

# Test instantiation with an LLMProvider
def test_topology_instantiation(dummy_llm_provider):
    class ConcreteTopology(Topology):
        def execute(self, natural_language_query: str, metadata: Dict[str, Any], agent_type: str, file_path: str, table_name: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "executed"}

    topology_instance = ConcreteTopology(llm_provider=dummy_llm_provider)
    assert topology_instance.llm_provider is dummy_llm_provider
    assert topology_instance.topology_config == {} # Default empty config

    config = {"retries": 3}
    topology_instance_with_config = ConcreteTopology(llm_provider=dummy_llm_provider, topology_config=config)
    assert topology_instance_with_config.topology_config == config


def test_topology_get_config_value(dummy_llm_provider):
    class ConcreteTopology(Topology):
        def execute(self, natural_language_query: str, metadata: Dict[str, Any], agent_type: str, file_path: str, table_name: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "executed"}

    my_config = {"key1": "value1", "key2": 100}
    topology = ConcreteTopology(llm_provider=dummy_llm_provider, topology_config=my_config)

    assert topology._get_config_value("key1") == "value1"
    assert topology._get_config_value("key2") == 100
    assert topology._get_config_value("non_existent_key") is None
    assert topology._get_config_value("non_existent_key", "default_val") == "default_val"

    empty_config_topology = ConcreteTopology(llm_provider=dummy_llm_provider)
    assert empty_config_topology._get_config_value("any_key") is None
    assert empty_config_topology._get_config_value("any_key", "default") == "default"


# Test that instantiating Topology directly raises TypeError
def test_cannot_instantiate_topology_directly(dummy_llm_provider):
    with pytest.raises(TypeError) as excinfo:
        Topology(llm_provider=dummy_llm_provider) # type: ignore
    assert "Can't instantiate abstract class Topology" in str(excinfo.value)
    assert "execute" in str(excinfo.value) # Check if the abstract method is mentioned

# Placeholder for future concrete topology tests
def test_placeholder_for_concrete_topology_tests():
    assert True

# Need __init__.py in app/backend/topologies
# Will be created by a separate tool call.


# --- Tests for SequentialReflectTopology ---
from unittest.mock import MagicMock, call, patch
import pandas as pd
from app.backend.topologies.sequential_reflect import SequentialReflectTopology

@pytest.fixture
def mock_llm_provider():
    provider = MagicMock(spec=LLMProvider)
    # Configure specific mock return values as needed per test
    return provider

@pytest.fixture
def sample_metadata():
    return {
        "table_name": "test_table", # This is the R object name / SQL table alias / Python df name
        "columns": [
            {"name": "col_a", "type": "INTEGER"},
            {"name": "col_b", "type": "TEXT"}
        ],
        "file_type": "csv" # Example, could be rdata, parquet, sqlite
    }

@pytest.fixture
def sample_r_metadata(): # For R, table_name in metadata usually means the R object name
     return {
        "table_name": "my_r_object",
        "columns": [
            {"name": "factor_col", "type": "factor"},
            {"name": "value_col", "type": "numeric"}
        ],
        "file_type": "rdata"
    }


# Test successful execution path (no errors)
@patch('app.backend.topologies.sequential_reflect.execute_duckdb_query')
def test_sequential_reflect_success_sql(mock_execute_query, mock_llm_provider, sample_metadata):
    # Configure LLM mocks
    mock_llm_provider.generate_code.return_value = "SELECT * FROM test_table"
    mock_llm_provider.generate_summary.return_value = "This is a summary of the results."

    # Configure execution mock
    mock_results_df = pd.DataFrame({"col_a": [1, 2], "col_b": ["a", "b"]})
    mock_execute_query.return_value = (mock_results_df, None) # (df, error_message)

    topology = SequentialReflectTopology(llm_provider=mock_llm_provider)
    result = topology.execute(
        natural_language_query="Get all data",
        metadata=sample_metadata,
        agent_type="sql",
        file_path="/path/to/data.csv",
        table_name="test_table", # This is the table name for DuckDB context
        original_uploaded_filename="data.csv"
    )

    assert result['error'] is None
    assert result['executed_query_text'] == "SELECT * FROM test_table"
    assert result['natural_language_response'] == "This is a summary of the results."
    assert result['results'] == mock_results_df.to_dict(orient='records')

    mock_llm_provider.generate_code.assert_called_once() # Initial generation
    # Correction prompt should not be called
    assert any("Corrected SQL Query:" in c.kwargs['prompt'] for c in mock_llm_provider.generate_code.call_args_list if "Corrected SQL Query:" in c.kwargs.get('prompt','')) is False
    mock_llm_provider.generate_summary.assert_called_once()
    mock_execute_query.assert_called_once_with("SELECT * FROM test_table", "/path/to/data.csv", "test_table")

# Test execution path with one error and successful correction
@patch('app.backend.topologies.sequential_reflect.execute_duckdb_query')
def test_sequential_reflect_correction_sql(mock_execute_query, mock_llm_provider, sample_metadata):
    # LLM generates initial bad code, then good code after correction
    mock_llm_provider.generate_code.side_effect = [
        "SELECT FRO test_table", # Initial bad code
        "SELECT * FROM test_table"  # Corrected code
    ]
    mock_llm_provider.generate_summary.return_value = "Summary of corrected results."

    # Execution fails first, then succeeds
    mock_results_df = pd.DataFrame({"col_a": [1], "col_b": ["x"]})
    mock_execute_query.side_effect = [
        (None, "Syntax error near FRO"), # Initial execution fails
        (mock_results_df, None)          # Corrected execution succeeds
    ]

    topology = SequentialReflectTopology(llm_provider=mock_llm_provider)
    result = topology.execute(
        natural_language_query="Get all data",
        metadata=sample_metadata,
        agent_type="sql",
        file_path="/path/to/data.csv",
        table_name="test_table",
        original_uploaded_filename="data.csv"
    )

    assert result['error'] is None # Final error should be None
    assert result['executed_query_text'] == "SELECT * FROM test_table" # Should be the corrected query
    assert result['natural_language_response'] == "Summary of corrected results."
    assert result['results'] == mock_results_df.to_dict(orient='records')

    assert mock_llm_provider.generate_code.call_count == 2 # Initial + Correction
    # Check that the correction prompt was actually constructed and called
    correction_call_args = mock_llm_provider.generate_code.call_args_list[1]
    assert "Corrected SQL Query:" in correction_call_args.kwargs['prompt']
    assert "Syntax error near FRO" in correction_call_args.kwargs['prompt']

    mock_llm_provider.generate_summary.assert_called_once()
    assert mock_execute_query.call_count == 2
    mock_execute_query.assert_any_call("SELECT FRO test_table", "/path/to/data.csv", "test_table")
    mock_execute_query.assert_any_call("SELECT * FROM test_table", "/path/to/data.csv", "test_table")

# Test execution path where correction also fails
@patch('app.backend.topologies.sequential_reflect.execute_duckdb_query')
def test_sequential_reflect_correction_fails_sql(mock_execute_query, mock_llm_provider, sample_metadata):
    mock_llm_provider.generate_code.side_effect = [
        "SELECT FRO test_table",      # Initial bad code
        "SELECT STILL_BAD FROM table" # Corrected but still bad code
    ]
    # No summary should be generated if correction fails
    mock_llm_provider.generate_summary.return_value = "This should not be called."

    mock_execute_query.side_effect = [
        (None, "Syntax error near FRO"),
        (None, "Syntax error near STILL_BAD")
    ]

    topology = SequentialReflectTopology(llm_provider=mock_llm_provider)
    result = topology.execute(
        natural_language_query="Get all data",
        metadata=sample_metadata,
        agent_type="sql",
        file_path="/path/to/data.csv",
        table_name="test_table",
        original_uploaded_filename="data.csv"
    )

    assert result['error'] == "Syntax error near STILL_BAD" # Final error after correction attempt
    assert result['executed_query_text'] == "SELECT STILL_BAD FROM table"
    assert "An error occurred: Syntax error near STILL_BAD" in result['natural_language_response']
    assert not result['results'] # No results

    assert mock_llm_provider.generate_code.call_count == 2
    mock_llm_provider.generate_summary.assert_not_called() # IMPORTANT: No summary on persistent error
    assert mock_execute_query.call_count == 2


# Test R datatable agent path (successful)
@patch('app.backend.topologies.sequential_reflect.execute_r_script')
def test_sequential_reflect_success_r(mock_execute_r, mock_llm_provider, sample_r_metadata):
    r_code = "active_df <- active_df[value_col > 10, .N, by = factor_col]"
    mock_llm_provider.generate_code.return_value = r_code
    mock_llm_provider.generate_summary.return_value = "R script summary."

    mock_r_results_df = pd.DataFrame({"factor_col": ["A"], "N": [5]})
    mock_execute_r.return_value = (mock_r_results_df, None)

    topology = SequentialReflectTopology(llm_provider=mock_llm_provider)
    result = topology.execute(
        natural_language_query="Count by factor_col for value_col > 10",
        metadata=sample_r_metadata, # R specific metadata
        agent_type="r_datatable",
        file_path="/path/to/data.Rdata",
        table_name="my_r_object", # This is the R object name
        original_uploaded_filename="data.Rdata"
    )

    assert result['error'] is None
    assert result['executed_query_text'] == r_code
    assert result['natural_language_response'] == "R script summary."
    mock_execute_r.assert_called_once_with(r_code, "/path/to/data.Rdata", "my_r_object")
    # Check that the prompt for R code generation was correct
    initial_prompt_args = mock_llm_provider.generate_code.call_args_list[0]
    assert "R data.table Code (assign to active_df):" in initial_prompt_args.kwargs['prompt']
    assert f"object named `my_r_object`" in initial_prompt_args.kwargs['prompt']


# Test Python Pandas agent path (successful)
@patch('app.backend.topologies.sequential_reflect.execute_python_pandas_code')
def test_sequential_reflect_success_python(mock_execute_python, mock_llm_provider, sample_metadata):
    # sample_metadata table_name is 'test_table', this will be the df name
    python_code = "test_table = test_table[test_table['col_a'] > 0]"
    mock_llm_provider.generate_code.return_value = python_code
    mock_llm_provider.generate_summary.return_value = "Python script summary."

    mock_python_results_df = pd.DataFrame({"col_a": [1,2], "col_b": ["a","b"]})
    mock_execute_python.return_value = (mock_python_results_df, None)

    topology = SequentialReflectTopology(llm_provider=mock_llm_provider)
    result = topology.execute(
        natural_language_query="Filter col_a > 0",
        metadata=sample_metadata, # Reusing sample_metadata, table_name is 'test_table'
        agent_type="python_pandas",
        file_path="/path/to/data.csv", # Source data file
        table_name="test_table", # This is the df name for pandas_code_execution
        original_uploaded_filename="data.csv"
    )
    assert result['error'] is None
    assert result['executed_query_text'] == python_code
    mock_execute_python.assert_called_once_with(python_code, "/path/to/data.csv", dataframe_name="test_table")
    initial_prompt_args = mock_llm_provider.generate_code.call_args_list[0]
    assert f"DataFrame variable named `test_table`" in initial_prompt_args.kwargs['prompt']
    assert f"assigns the result back to the *same* variable `test_table`" in initial_prompt_args.kwargs['prompt']


# Test critical error during LLM call (e.g., API failure)
def test_sequential_reflect_llm_critical_failure(mock_llm_provider, sample_metadata):
    mock_llm_provider.generate_code.side_effect = Exception("LLM API is down")

    topology = SequentialReflectTopology(llm_provider=mock_llm_provider)
    result = topology.execute(
        natural_language_query="Get all data",
        metadata=sample_metadata,
        agent_type="sql",
        file_path="/path/to/data.csv",
        table_name="test_table",
        original_uploaded_filename="data.csv"
    )

    assert "Critical error in SequentialReflectTopology: LLM API is down" in result['error']
    assert "Critical error in SequentialReflectTopology: LLM API is down" in result['natural_language_response']
    assert not result['results']
    assert result['executed_query_text'] == "" # No code was successfully generated/chosen


# Test that intermediate steps are recorded
@patch('app.backend.topologies.sequential_reflect.execute_duckdb_query')
def test_sequential_reflect_intermediate_steps(mock_execute_query, mock_llm_provider, sample_metadata):
    mock_llm_provider.generate_code.side_effect = ["BAD SQL", "GOOD SQL"]
    mock_llm_provider.generate_summary.return_value = "Summary."
    mock_execute_query.side_effect = [(None, "Error"), (pd.DataFrame({'a':[1]}), None)]

    topology = SequentialReflectTopology(llm_provider=mock_llm_provider)
    result = topology.execute("query", sample_metadata, "sql", "file.csv", "test_table", "file.csv")

    assert len(result['intermediate_steps']) > 0
    steps = [s['step'] for s in result['intermediate_steps']]
    assert "Generate Initial Code" in steps
    assert "Execute Initial Code" in steps
    assert "Attempt Correction" in steps
    assert "Execute Corrected Code" in steps
    assert "Summarize Results" in steps

    initial_code_step = next(s for s in result['intermediate_steps'] if s['step'] == "Generate Initial Code")
    assert initial_code_step['generated_code'] == "BAD SQL"

    correction_attempt_step = next(s for s in result['intermediate_steps'] if s['step'] == "Attempt Correction")
    assert correction_attempt_step['failed_code'] == "BAD SQL"
    assert correction_attempt_step['error_message'] == "Error"
    assert correction_attempt_step['corrected_code'] == "GOOD SQL"

    final_summary_step = next(s for s in result['intermediate_steps'] if s['step'] == "Summarize Results")
    assert final_summary_step['summary'] == "Summary."


# --- Tests for ParallelEnsembleTopology ---
from app.backend.topologies.parallel_ensemble import ParallelEnsembleTopology

@pytest.fixture
def parallel_config():
    return {
        "code_gen_models": ["mock_model_1", "mock_model_2", "mock_model_3"],
        "summary_model": "summary_mock_model",
        "max_workers": 3
    }

# Test successful path: first model generates valid code that executes
@patch('app.backend.topologies.sequential_reflect.execute_duckdb_query') # Target where it's *used* by the helper
def test_parallel_ensemble_first_model_succeeds(mock_execute_query, mock_llm_provider, sample_metadata, parallel_config):
    # LLM Provider mock setup
    # Model 1 generates good code, Model 2 bad, Model 3 good but won't be used if Model 1's code executes first.
    mock_llm_provider.generate_code.side_effect = lambda prompt, model_name: {
        "mock_model_1": "SELECT * FROM good_table_1",
        "mock_model_2": "SELECT BAD SYNTAX 2",
        "mock_model_3": "SELECT * FROM good_table_3"
    }.get(model_name, "default_code")

    mock_llm_provider.generate_summary.return_value = "Summary from parallel."

    # Execution mock setup
    # Code from mock_model_1 executes successfully.
    # Code from mock_model_2 would fail (but might not be called if model_1 is tried first and succeeds)
    # Code from mock_model_3 would succeed
    mock_df_model1 = pd.DataFrame({"col1": [1,2]})

    # We need to ensure execute_duckdb_query is called for "SELECT * FROM good_table_1"
    # and returns success. Other calls might not happen or might fail.
    def selective_execute(*args, **kwargs):
        code_being_executed = args[0]
        if code_being_executed == "SELECT * FROM good_table_1":
            return mock_df_model1, None
        elif code_being_executed == "SELECT BAD SYNTAX 2":
            return None, "Syntax Error for model 2"
        elif code_being_executed == "SELECT * FROM good_table_3":
            # This shouldn't be reached if model 1's output is tried first and succeeds
            return pd.DataFrame({"col3": [7,8]}), None
        return None, "Unknown code execution"

    mock_execute_query.side_effect = selective_execute

    topology = ParallelEnsembleTopology(llm_provider=mock_llm_provider, topology_config=parallel_config)
    result = topology.execute(
        natural_language_query="Get data",
        metadata=sample_metadata,
        agent_type="sql",
        file_path="/test/file.csv",
        table_name="test_table",
        original_uploaded_filename="file.csv"
    )

    assert result['error'] is None
    assert result['executed_query_text'] == "SELECT * FROM good_table_1"
    assert result['natural_language_response'] == "Summary from parallel."
    assert result['results'] == mock_df_model1.to_dict(orient='records')

    # Check LLM calls
    # generate_code should be called for all models in parallel_config
    assert mock_llm_provider.generate_code.call_count == len(parallel_config["code_gen_models"])
    called_models_for_code_gen = {call_args.kwargs['model_name'] for call_args in mock_llm_provider.generate_code.call_args_list}
    assert called_models_for_code_gen == set(parallel_config["code_gen_models"])

    mock_llm_provider.generate_summary.assert_called_once() # Only one summary for the successful code

    # Check execution calls - only the first successful one matters for the final result
    # The topology tries them in order of preference (which matches model list here)
    mock_execute_query.assert_any_call("SELECT * FROM good_table_1", "/test/file.csv", "test_table")
    # It's possible Model 2's code was generated and *could* have been tried if Model 1 failed.
    # The key is that Model 1's code was tried and succeeded.

    # Verify that the summary was generated using the correct model specified in topology_config
    summary_call_args = mock_llm_provider.generate_summary.call_args_list[0]
    assert summary_call_args.kwargs['model_name'] == parallel_config["summary_model"]

    # Check intermediate steps for clarity
    gen_outputs_step = next(s for s in result['intermediate_steps'] if s['step'] == "Parallel Code Generation")
    assert len(gen_outputs_step['generation_outputs']) == 3

    exec_attempt_step = next(s for s in result['intermediate_steps'] if s['step'] == "Attempting Execution of Generated Codes")
    assert exec_attempt_step['order'] == ["mock_model_1", "mock_model_2", "mock_model_3"] # Based on config order

    chosen_exec_step = next(s for s in result['intermediate_steps'] if s.get("chosen_for_summary"))
    assert chosen_exec_step['sub_step'] == "Execute code from mock_model_1"


# Test path: first model fails to generate code, second succeeds and executes
@patch('app.backend.topologies.sequential_reflect.execute_duckdb_query') # Target where it's *used* by the helper
def test_parallel_ensemble_second_model_succeeds(mock_execute_query, mock_llm_provider, sample_metadata, parallel_config):
    mock_llm_provider.generate_code.side_effect = lambda prompt, model_name: {
        "mock_model_1": None, # Model 1 fails to generate
        "mock_model_2": "SELECT * FROM good_table_2",
        "mock_model_3": "SELECT * FROM good_table_3"
    }.get(model_name, "default_code")
    mock_llm_provider.generate_summary.return_value = "Summary from model 2."

    mock_df_model2 = pd.DataFrame({"col2": [3,4]})
    def selective_execute(*args, **kwargs):
        code = args[0]
        if code == "SELECT * FROM good_table_2": return mock_df_model2, None
        return None, "Execution error for other codes"
    mock_execute_query.side_effect = selective_execute

    topology = ParallelEnsembleTopology(llm_provider=mock_llm_provider, topology_config=parallel_config)
    result = topology.execute("Get data", sample_metadata, "sql", "/file.csv", "test_table", "file.csv")

    assert result['error'] is None
    assert result['executed_query_text'] == "SELECT * FROM good_table_2"
    assert result['results'] == mock_df_model2.to_dict(orient='records')
    mock_execute_query.assert_called_once_with("SELECT * FROM good_table_2", "/file.csv", "test_table")

    gen_outputs_step = next(s for s in result['intermediate_steps'] if s['step'] == "Parallel Code Generation")
    model1_output = next(o for o in gen_outputs_step['generation_outputs'] if o['model'] == "mock_model_1")
    assert model1_output['code_generated'] is False

    chosen_exec_step = next(s for s in result['intermediate_steps'] if s.get("chosen_for_summary"))
    assert chosen_exec_step['sub_step'] == "Execute code from mock_model_2"


# Test path: all models generate code, but all executions fail
@patch('app.backend.topologies.sequential_reflect.execute_duckdb_query') # Target where it's *used* by the helper
def test_parallel_ensemble_all_executions_fail(mock_execute_query, mock_llm_provider, sample_metadata, parallel_config):
    mock_llm_provider.generate_code.side_effect = lambda prompt, model_name: f"CODE FROM {model_name}"
    # Summary should not be called
    mock_llm_provider.generate_summary.return_value = "This should not appear."

    # All executions fail
    mock_execute_query.return_value = (None, "Generic Execution Error")

    topology = ParallelEnsembleTopology(llm_provider=mock_llm_provider, topology_config=parallel_config)
    result = topology.execute("Get data", sample_metadata, "sql", "/file.csv", "test_table", "file.csv")

    assert result['error'] is not None
    assert "Generic Execution Error" in result['error']
    # The error message will be from the last attempted model in the preferred list
    assert f"Error from {parallel_config['code_gen_models'][-1]}" in result['error']
    assert "An error occurred: " in result['natural_language_response']
    assert not result['results']
    mock_llm_provider.generate_summary.assert_not_called()
    assert mock_execute_query.call_count == len(parallel_config["code_gen_models"])


# Test path: all models fail to generate any code
def test_parallel_ensemble_all_code_generation_fails(mock_llm_provider, sample_metadata, parallel_config):
    mock_llm_provider.generate_code.return_value = None # All models return None
    mock_llm_provider.generate_summary.return_value = "This should not appear."

    topology = ParallelEnsembleTopology(llm_provider=mock_llm_provider, topology_config=parallel_config)
    result = topology.execute("Get data", sample_metadata, "sql", "/file.csv", "test_table", "file.csv")

    assert result['error'] == "All LLM configurations failed to generate code."
    assert result['natural_language_response'] == "All LLM configurations failed to generate code."
    assert not result['results']
    mock_llm_provider.generate_summary.assert_not_called()

    # Ensure execute_duckdb_query (or other exec helpers) are not called
    # This requires patching them if they were globally patched for other tests,
    # or ensuring no mock_execute_query fixture is passed here if it implies calls.
    # For this test, we don't need to mock/patch execution helpers as they shouldn't be reached.


import json # <--- Import added here

# --- Tests for IterativeReasonActTopology ---
from app.backend.topologies.iterative_reason_act import IterativeReasonActTopology

@pytest.fixture
def react_config():
    return {
        "max_iterations": 3,
        "reason_act_model": "react_model",
        "code_gen_model": "code_gen_model_for_react", # Can be same as react_model
        "summary_model": "summary_model_for_react" # Can be same as react_model
    }

# Helper to create LLM response for ReAct
def create_react_llm_response(thought: str, tool_name: str, tool_input: Dict[str, Any]) -> str:
    action_block = {
        "thought": thought,
        "action": {
            "tool_name": tool_name,
            "tool_input": tool_input
        }
    }
    return f"Some preamble text perhaps...\n```json\n{json.dumps(action_block)}\n```\nSome trailing text."

# Test a simple successful ReAct flow: generate_code -> execute_code -> provide_final_answer
@patch('app.backend.topologies.sequential_reflect.execute_duckdb_query') # Corrected Patch Target
def test_react_topology_simple_success_sql(mock_execute_query, mock_llm_provider, sample_metadata, react_config):
    # --- Mock LLM Responses for ReAct loop ---
    # Iteration 1: Decide to generate SQL
    action1_input = {"language": "sql", "query_goal": "Get count of test_table"}
    llm_response_1 = create_react_llm_response(
        thought="I need to count rows in test_table.",
        tool_name="generate_code",
        tool_input=action1_input
    )
    # Iteration 2: LLM sees generated SQL, decides to execute it
    generated_sql_code = "SELECT COUNT(*) FROM test_table;"
    action2_input = {"language": "sql", "code": generated_sql_code}
    llm_response_2 = create_react_llm_response(
        thought="SQL code is generated, I should execute it.",
        tool_name="execute_code",
        tool_input=action2_input
    )
    # Iteration 3: LLM sees execution result, decides to provide final answer
    action3_input = {"summary_text": "The table test_table has 10 rows."}
    llm_response_3 = create_react_llm_response(
        thought="Execution successful, I have the count.",
        tool_name="provide_final_answer",
        tool_input=action3_input
    )

    mock_llm_provider.generate_text.side_effect = [llm_response_1, llm_response_2, llm_response_3]

    # LLM response for the generate_code action itself (called by _execute_action)
    mock_llm_provider.generate_code.return_value = generated_sql_code

    # Mock for execute_code action
    mock_execution_df = pd.DataFrame([{"COUNT(*)": 10}])
    mock_execute_query.return_value = (mock_execution_df, None)

    topology = IterativeReasonActTopology(llm_provider=mock_llm_provider, topology_config=react_config)
    result = topology.execute(
        natural_language_query="How many rows in test_table?",
        metadata=sample_metadata,
        agent_type="sql",
        file_path="/path/to/data.csv",
        table_name="test_table",
        original_uploaded_filename="data.csv"
    )

    assert result['error'] is None
    assert result['natural_language_response'] == "Final Answer: The table test_table has 10 rows."
    assert result['executed_query_text'] == generated_sql_code
    assert result['results'] == mock_execution_df.to_dict(orient='records')

    assert mock_llm_provider.generate_text.call_count == 3 # For 3 ReAct iterations
    mock_llm_provider.generate_code.assert_called_once() # For the generate_code action
    mock_execute_query.assert_called_once_with(generated_sql_code, "/path/to/data.csv", "test_table")

    # Check history (intermediate_steps)
    history = result['intermediate_steps']
    assert len(history) == 3
    assert history[0]['thought'] == "I need to count rows in test_table."
    assert json.loads(history[0]['action_taken_str'])['tool_name'] == "generate_code"
    assert f"Generated sql code:\n{generated_sql_code}" in history[0]['observation']

    assert history[1]['thought'] == "SQL code is generated, I should execute it."
    assert json.loads(history[1]['action_taken_str'])['tool_name'] == "execute_code"
    assert "Result preview:\n   COUNT(*)\n0        10" in history[1]['observation']

    assert history[2]['thought'] == "Execution successful, I have the count."
    assert json.loads(history[2]['action_taken_str'])['tool_name'] == "provide_final_answer"
    assert "Final Answer: The table test_table has 10 rows." in history[2]['observation']


# --- Tests for DefaultFallbackTopology ---
from app.backend.topologies.default_fallback import DefaultFallbackTopology

@pytest.fixture
def mock_parallel_topology_init(mocker):
    mock_instance = MagicMock(spec=ParallelEnsembleTopology)
    return mocker.patch('app.backend.topologies.default_fallback.ParallelEnsembleTopology', return_value=mock_instance)

@pytest.fixture
def mock_sequential_topology_init(mocker):
    mock_instance = MagicMock(spec=SequentialReflectTopology)
    # Mock the _summarize_results method separately if needed for refinement tests
    mock_instance._summarize_results = MagicMock(return_value="Refined summary from sequential helper.")
    return mocker.patch('app.backend.topologies.default_fallback.SequentialReflectTopology', return_value=mock_instance)


def test_default_fallback_parallel_succeeds_no_refine(
    mock_parallel_topology_init, mock_sequential_topology_init,
    mock_llm_provider, sample_metadata
):
    parallel_success_result = {
        'executed_query_text': "PARALLEL_SQL", 'results': [{"data": 1}], 'error': None,
        'natural_language_response': "Parallel success summary.", 'intermediate_steps': [{"par_step":1}]
    }
    mock_parallel_topology_init.return_value.execute.return_value = parallel_success_result

    config = {"refine_summary_after_parallel": False}
    topology = DefaultFallbackTopology(llm_provider=mock_llm_provider, topology_config=config)

    result = topology.execute("query", sample_metadata, "sql", "file.csv", "tbl", "file.csv")

    assert result['executed_query_text'] == "PARALLEL_SQL"
    assert result['natural_language_response'] == "Parallel success summary."
    assert result['error'] is None
    mock_parallel_topology_init.return_value.execute.assert_called_once()
    mock_sequential_topology_init.return_value.execute.assert_not_called()
    # Ensure sequential's _summarize_results was not called for refinement
    mock_sequential_topology_init.return_value._summarize_results.assert_not_called()
    assert len(result['intermediate_steps']) == 4 # DefaultInit, AttemptParallel, ParallelResult, plus one from parallel_result itself


def test_default_fallback_parallel_succeeds_with_refine(
    mock_parallel_topology_init, mock_sequential_topology_init,
    mock_llm_provider, sample_metadata
):
    parallel_success_result = {
        'executed_query_text': "PARALLEL_SQL_REFINE", 'results': [{"data_refine": 1}], 'error': None,
        'natural_language_response': "Original parallel summary.", 'intermediate_steps': [{"par_step_refine":1}]
    }
    mock_parallel_topology_init.return_value.execute.return_value = parallel_success_result

    # Sequential topology's _summarize_results method will be called by DefaultFallback
    expected_refined_summary = "Refined summary from sequential helper."
    mock_sequential_topology_init.return_value._summarize_results.return_value = expected_refined_summary

    config = {"refine_summary_after_parallel": True} # Enable refinement
    topology = DefaultFallbackTopology(llm_provider=mock_llm_provider, topology_config=config)

    result = topology.execute("query_refine", sample_metadata, "sql", "file_r.csv", "tbl_r", "file_r.csv")

    assert result['executed_query_text'] == "PARALLEL_SQL_REFINE"
    assert result['natural_language_response'] == expected_refined_summary # Summary is refined
    assert result['error'] is None
    mock_parallel_topology_init.return_value.execute.assert_called_once()
    mock_sequential_topology_init.return_value.execute.assert_not_called() # Full sequential execute not called
    mock_sequential_topology_init.return_value._summarize_results.assert_called_once()

    refine_step_found = any(s['step'] == "Refining Summary (Post-Parallel Success)" for s in result['intermediate_steps'])
    assert refine_step_found


def test_default_fallback_parallel_fails_sequential_succeeds(
    mock_parallel_topology_init, mock_sequential_topology_init,
    mock_llm_provider, sample_metadata
):
    parallel_fail_result = {
        'executed_query_text': "FAIL_SQL", 'results': [], 'error': "Parallel error.",
        'natural_language_response': "Parallel failed.", 'intermediate_steps': [{"par_fail_step":1}]
    }
    mock_parallel_topology_init.return_value.execute.return_value = parallel_fail_result

    sequential_success_result = {
        'executed_query_text': "SEQ_SQL", 'results': [{"seq_data": 2}], 'error': None,
        'natural_language_response': "Sequential success summary.", 'intermediate_steps': [{"seq_step":1}]
    }
    mock_sequential_topology_init.return_value.execute.return_value = sequential_success_result

    topology = DefaultFallbackTopology(llm_provider=mock_llm_provider) # No special config needed
    result = topology.execute("query2", sample_metadata, "sql", "file2.csv", "tbl2", "file2.csv")

    assert result['executed_query_text'] == "SEQ_SQL"
    assert result['natural_language_response'] == "Sequential success summary."
    assert result['error'] is None
    mock_parallel_topology_init.return_value.execute.assert_called_once()
    mock_sequential_topology_init.return_value.execute.assert_called_once()

    fallback_step_found = any(s['step'] == "Parallel Ensemble Failed, Proceeding to Sequential Fallback" for s in result['intermediate_steps'])
    assert fallback_step_found


def test_default_fallback_both_fail(
    mock_parallel_topology_init, mock_sequential_topology_init,
    mock_llm_provider, sample_metadata
):
    parallel_fail_result = {'error': "Parallel error.", 'intermediate_steps': []} # Simplified
    mock_parallel_topology_init.return_value.execute.return_value = parallel_fail_result

    sequential_fail_result = {'error': "Sequential error.", 'natural_language_response': "Sequential failed.", 'intermediate_steps': []} # Simplified
    mock_sequential_topology_init.return_value.execute.return_value = sequential_fail_result

    topology = DefaultFallbackTopology(llm_provider=mock_llm_provider)
    result = topology.execute("query3", sample_metadata, "sql", "file3.csv", "tbl3", "file3.csv")

    assert result['error'] == "Sequential error." # Error from sequential should be final
    assert result['natural_language_response'] == "Sequential failed."
    mock_parallel_topology_init.return_value.execute.assert_called_once()
    mock_sequential_topology_init.return_value.execute.assert_called_once()

def test_default_fallback_parallel_critical_error_sequential_succeeds(
    mock_parallel_topology_init, mock_sequential_topology_init,
    mock_llm_provider, sample_metadata
):
    # Parallel topology's execute method itself raises an exception
    mock_parallel_topology_init.return_value.execute.side_effect = Exception("Parallel critical failure")

    sequential_success_result = {
        'executed_query_text': "SEQ_SQL_CRIT", 'results': [{"seq_crit_data": 3}], 'error': None,
        'natural_language_response': "Sequential success after parallel critical.", 'intermediate_steps': [{"seq_crit_step":1}]
    }
    mock_sequential_topology_init.return_value.execute.return_value = sequential_success_result

    topology = DefaultFallbackTopology(llm_provider=mock_llm_provider)
    result = topology.execute("query_crit", sample_metadata, "sql", "file_crit.csv", "tbl_crit", "file_crit.csv")

    assert result['error'] is None # Sequential succeeded
    assert result['natural_language_response'] == "Sequential success after parallel critical."
    mock_parallel_topology_init.return_value.execute.assert_called_once()
    mock_sequential_topology_init.return_value.execute.assert_called_once()

    crit_error_step = next(s for s in result['intermediate_steps'] if s['step'] == "Critical Error during Parallel Ensemble Execution")
    assert "Parallel critical failure" in crit_error_step['error_message']


# --- Tests for TopologyFactory ---
from app.backend.topologies.factory import TopologyFactory, TopologyFactoryError

def test_topology_factory_get_sequential_reflect(mock_llm_provider):
    config = {"some_specific_config": "value"}
    topology = TopologyFactory.get_topology("sequential_reflect", mock_llm_provider, config)
    assert isinstance(topology, SequentialReflectTopology)
    assert topology.llm_provider == mock_llm_provider
    assert topology.topology_config["some_specific_config"] == "value"

def test_topology_factory_get_parallel_ensemble(mock_llm_provider):
    config = {"code_gen_models": ["test_m1", "test_m2"]}
    topology = TopologyFactory.get_topology("parallel_ensemble", mock_llm_provider, config)
    assert isinstance(topology, ParallelEnsembleTopology)
    assert topology.llm_provider == mock_llm_provider
    assert topology.code_gen_models == ["test_m1", "test_m2"]

def test_topology_factory_get_iterative_reason_act(mock_llm_provider):
    config = {"max_iterations": 10}
    topology = TopologyFactory.get_topology("iterative_reason_act", mock_llm_provider, config)
    assert isinstance(topology, IterativeReasonActTopology)
    assert topology.llm_provider == mock_llm_provider
    assert topology.max_iterations == 10

def test_topology_factory_get_default_fallback(mock_llm_provider):
    config = {"refine_summary_after_parallel": True}
    topology = TopologyFactory.get_topology("default_fallback", mock_llm_provider, config)
    assert isinstance(topology, DefaultFallbackTopology)
    assert topology.llm_provider == mock_llm_provider
    assert topology.refine_summary_after_parallel is True

def test_topology_factory_unknown_topology(mock_llm_provider):
    with pytest.raises(TopologyFactoryError) as excinfo:
        TopologyFactory.get_topology("non_existent_topology", mock_llm_provider)
    assert "Unknown topology name: non_existent_topology" in str(excinfo.value)

def test_topology_factory_case_insensitivity(mock_llm_provider):
    topology = TopologyFactory.get_topology("SeQuEnTiAl_ReFlEcT", mock_llm_provider)
    assert isinstance(topology, SequentialReflectTopology)

def test_topology_factory_no_config_provided(mock_llm_provider):
    # Ensure it works when topology_specific_config is None (should default to {})
    topology = TopologyFactory.get_topology("sequential_reflect", mock_llm_provider)
    assert isinstance(topology, SequentialReflectTopology)
    assert topology.topology_config == {} # Check it defaulted to empty dict


# Test ReAct reaching max iterations
@patch('app.backend.code_execution.execute_duckdb_query')
def test_react_topology_max_iterations(mock_execute_query, mock_llm_provider, sample_metadata, react_config):
    # LLM always decides to generate code (simulating a loop)
    action_input = {"language": "sql", "query_goal": "Get data"}
    llm_response_loop = create_react_llm_response(
        thought="I still need more data, let me try generating some code again.",
        tool_name="generate_code",
        tool_input=action_input
    )
    mock_llm_provider.generate_text.return_value = llm_response_loop # Same response for all ReAct steps
    mock_llm_provider.generate_code.return_value = "SELECT 1;" # Code gen action is successful

    max_iters = react_config["max_iterations"]
    topology = IterativeReasonActTopology(llm_provider=mock_llm_provider, topology_config=react_config)
    result = topology.execute("Query", sample_metadata, "sql", "/file.csv", "test_table", "data.csv")

    assert "Agent reached maximum iterations" in result['error']
    assert "Agent reached maximum iterations" in result['natural_language_response']
    assert mock_llm_provider.generate_text.call_count == max_iters
    assert mock_llm_provider.generate_code.call_count == max_iters # generate_code called each iteration

    history = result['intermediate_steps']
    assert len(history) == max_iters
    assert "I still need more data" in history[-1]['thought']


# Test ReAct when LLM fails to parse its own action JSON
def test_react_topology_json_parse_failure(mock_llm_provider, sample_metadata, react_config):
    mock_llm_provider.generate_text.return_value = "This is not valid JSON for an action."

    topology = IterativeReasonActTopology(llm_provider=mock_llm_provider, topology_config=react_config)
    result = topology.execute("Query", sample_metadata, "sql", "/file.csv", "test_table", "data.csv")

    assert "Failed to parse LLM action" in result['error']
    assert "Failed to parse LLM action" in result['natural_language_response']
    assert mock_llm_provider.generate_text.call_count == 1
    history = result['intermediate_steps']
    assert len(history) == 1
    assert "Failed to parse own action" in history[0]['thought']


# Test ReAct when an execute_code action results in an error from code_execution
@patch('app.backend.topologies.sequential_reflect.execute_duckdb_query') # Corrected Patch Target
def test_react_topology_execute_code_error(mock_execute_query, mock_llm_provider, sample_metadata, react_config):
    # Iteration 1: Generate code
    llm_response_1 = create_react_llm_response("Thought 1", "generate_code", {"language": "sql", "query_goal": "Test"})
    # Iteration 2: Execute (this will fail)
    generated_code_to_fail = "SELECT BAD;"
    llm_response_2 = create_react_llm_response("Thought 2: Executing", "execute_code", {"language": "sql", "code": generated_code_to_fail})
    # Iteration 3: LLM sees error, decides to stop (hypothetically)
    llm_response_3 = create_react_llm_response("Thought 3: Saw error, giving up.", "provide_final_answer", {"summary_text": "Failed due to execution error."})

    mock_llm_provider.generate_text.side_effect = [llm_response_1, llm_response_2, llm_response_3]
    mock_llm_provider.generate_code.return_value = generated_code_to_fail

    # Mock execution to return an error
    mock_execute_query.return_value = (None, "DuckDB Execution Error")

    topology = IterativeReasonActTopology(llm_provider=mock_llm_provider, topology_config=react_config)
    result = topology.execute("Query", sample_metadata, "sql", "/file.csv", "test_table", "data.csv")

    assert result['error'] is None # Final error is None because LLM chose to provide_final_answer
    assert result['natural_language_response'] == "Final Answer: Failed due to execution error."
    assert result['executed_query_text'] == generated_code_to_fail # Last executed code

    history = result['intermediate_steps']
    assert len(history) == 3
    assert "Error executing code: DuckDB Execution Error" in history[1]['observation']
    assert json.loads(history[2]['action_taken_str'])['tool_name'] == "provide_final_answer"
