import pytest
import os
import json
import pandas as pd
import io
import subprocess # For subprocess.CompletedProcess
from app.backend.app import app, ALLOWED_EXTENSIONS, UPLOAD_FOLDER
# Import globals that might be modified or read by tests
from app.backend import app as backend_app
from app.backend.app import execute_r_script # Import the function to be tested
from openai import OpenAIError # For testing LLM failure scenarios


@pytest.fixture
def client():
    app.config['TESTING'] = True

    # Reset relevant global states before each test
    backend_app.current_metadata = None
    backend_app.current_uploaded_filename = None
    backend_app.current_uploaded_filepath = None

    # Ensure UPLOAD_FOLDER exists for testing and is empty
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    for f in os.listdir(UPLOAD_FOLDER): # Clean up any previous test files
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    with app.test_client() as client:
        yield client

    # Cleanup UPLOAD_FOLDER after test
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        except OSError as e:
            print(f"Error removing file {f} from {UPLOAD_FOLDER}: {e}")


def test_upload_rdata_metadata_extraction(client, mocker):
    """
    Tests the /upload endpoint for Rdata files and subsequent metadata extraction,
    mocking the R script execution for metadata.
    """
    # 1. Mock subprocess.run for the metadata extraction R script
    mock_r_metadata_json = {
        "object_name": "sample_df",
        "columns": [
            {"name": "id", "type": "integer"},
            {"name": "name", "type": "character"},
            {"name": "value", "type": "numeric"},
            {"name": "is_active", "type": "logical"}
        ]
    }
    # The path to subprocess.run as used in app.backend.app
    mock_subprocess_run = mocker.patch('app.backend.app.subprocess.run')
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=['Rscript', '-'],
        returncode=0,
        stdout=json.dumps(mock_r_metadata_json),
        stderr=''
    )

    # 2. Prepare the Rdata file for upload
    # The file 'test_r_data.Rdata' was created in the root by a previous bash step.
    # For the test, we need to place it where the app expects to save it, or simulate that.
    # The app saves to UPLOAD_FOLDER. The test fixture ensures UPLOAD_FOLDER is clean.
    # We'll make the client.post() write it there.

    rdata_file_name = "test_r_data.Rdata"
    # Path to the source Rdata file (created by Rscript in the root directory)
    source_rdata_file_path = os.path.join(os.getcwd(), rdata_file_name)

    # Ensure the source Rdata file exists
    if not os.path.exists(source_rdata_file_path):
        # Fallback: create a dummy file if the Rscript didn't run or file is missing
        # This allows the test to proceed with mocking, though it indicates a setup issue.
        print(f"Warning: Source Rdata file {source_rdata_file_path} not found. Creating dummy file.")
        with open(source_rdata_file_path, "wb") as f:
            # A real Rdata file has a specific structure. This is just a placeholder.
            # The content doesn't matter as much since subprocess.run is mocked for metadata.
            f.write(b"RDX2\nX\n\x00\x00\x00\x02\x00\x00\x00\x03\x00\x03\x05\x00\x00\x00\x00\x05sample_df\x00\x00\x00\x01\x00\x00\x00\x00")

    # 3. Simulate File Upload
    with open(source_rdata_file_path, 'rb') as rdata_file_content:
        data = {'file': (rdata_file_content, rdata_file_name)}
        response = client.post('/upload', data=data, content_type='multipart/form-data')

    # 4. Assert Response
    assert response.status_code == 200
    response_json = response.get_json()
    assert response_json['message'] == 'RData processed successfully. Metadata extracted.'

    # Define expected metadata after Python's type mapping
    # Type mapping in app.py:
    # 'integer': 'INTEGER', 'numeric': 'REAL', 'character': 'TEXT',
    # 'factor': 'TEXT', 'logical': 'BOOLEAN' (or TEXT), 'Date': 'DATE', 'POSIXct': 'TIMESTAMP'
    # Current mapping for 'logical' is BOOLEAN. If it was TEXT, this should be 'TEXT'.
    # Let's check the app.py: 'logical': 'BOOLEAN',
    expected_metadata_in_response = {
        'table_name': 'sample_df',
        'columns': [
            {'name': 'id', 'type': 'INTEGER'},
            {'name': 'name', 'type': 'TEXT'},
            {'name': 'value', 'type': 'REAL'},
            {'name': 'is_active', 'type': 'BOOLEAN'} # Based on 'logical': 'BOOLEAN' mapping
        ]
    }

    assert response_json['metadata']['table_name'] == expected_metadata_in_response['table_name']
    # Sort columns by name for comparison to avoid order issues
    sorted_actual_columns = sorted(response_json['metadata']['columns'], key=lambda x: x['name'])
    sorted_expected_columns = sorted(expected_metadata_in_response['columns'], key=lambda x: x['name'])
    assert sorted_actual_columns == sorted_expected_columns

    # 5. Assert Global State (current_metadata in the app)
    assert backend_app.current_metadata is not None
    assert backend_app.current_metadata['table_name'] == expected_metadata_in_response['table_name']
    # Sort columns in global state as well
    sorted_current_meta_cols = sorted(backend_app.current_metadata['columns'], key=lambda x: x['name'])
    assert sorted_current_meta_cols == sorted_expected_columns

    # Check that the file was actually saved to UPLOAD_FOLDER by the /upload endpoint
    expected_saved_path = os.path.join(UPLOAD_FOLDER, rdata_file_name)
    assert os.path.exists(expected_saved_path)

    # subprocess.run should have been called by the upload logic for Rdata
    mock_subprocess_run.assert_called_once()
    # Check some args of the call if necessary, e.g., that Rscript was used.
    # The first arg to subprocess.run is a list:
    assert mock_subprocess_run.call_args[0][0][0] == 'Rscript'
    # The R script itself is passed via stdin (input=...), and filepath_r as an argument
    assert mock_subprocess_run.call_args[0][0][2] == expected_saved_path.replace('\\', '/')


# It's good practice to clean up the generated Rdata file from the root directory
# This can be done in a session-level fixture or here if it's the only test using it.
def test_cleanup_generated_rdata_file():
    rdata_file_path = "test_r_data.Rdata"
    if os.path.exists(rdata_file_path):
        os.remove(rdata_file_path)

    r_script_path = "create_sample_rdata.R"
    if os.path.exists(r_script_path):
        os.remove(r_script_path)

    assert not os.path.exists(rdata_file_path), "Cleanup failed to remove test_r_data.Rdata"
    assert not os.path.exists(r_script_path), "Cleanup failed to remove create_sample_rdata.R"


# --- Tests for execute_r_script ---

def test_execute_r_script_success(mocker):
    """
    Tests successful execution of an R script via execute_r_script.
    Mocks subprocess.run to simulate R creating a CSV output.
    """
    # 1. Patch subprocess.run used by execute_r_script
    mock_subproc_run = mocker.patch('app.backend.app.subprocess.run')

    # 2. Define a side_effect function for the mock
    def mock_r_execution_success(*args, **kwargs):
        r_script_content = kwargs.get('input', '')
        import re
        # Extract the temporary CSV path from the R script content
        # The R script writes: fwrite(active_df, file="{temp_csv_path_r}", row.names=FALSE)
        match = re.search(r"fwrite\(active_df, file\s*=\s*['\"](.*?)['\"]", r_script_content)
        if match:
            temp_csv_path = match.group(1)
            # Simulate R script creating the CSV with expected data
            # For r_code_string = "active_df <- active_df[id > 1]"
            csv_content = "id,name,value,is_active\n2,Bob,20.3,FALSE\n3,Charlie,30.8,TRUE\n"
            with open(temp_csv_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            return subprocess.CompletedProcess(args[0], returncode=0, stdout="Mock R success", stderr="")
        else:
            # This indicates a problem with the test's understanding of the R script format
            return subprocess.CompletedProcess(args[0], returncode=1, stdout="", stderr="Mock Error: Could not find temp CSV path in R script input.")

    mock_subproc_run.side_effect = mock_r_execution_success

    rdata_file_path = "test_r_data.Rdata" # Assumed to be in root from previous setup
    # Ensure it exists for the test, or create a dummy if it doesn't
    if not os.path.exists(rdata_file_path):
        print(f"Warning: {rdata_file_path} not found for test_execute_r_script_success. Creating dummy.")
        with open(rdata_file_path, "wb") as f:
            f.write(b"dummy Rdata content for execute_r_script test")


    # 3. Call execute_r_script
    r_code_string = "active_df <- active_df[id > 1]"
    target_object_name = "sample_df"

    df_result, error = execute_r_script(r_code_string, rdata_file_path, target_object_name)

    # 4. Assert Results
    assert error is None, f"execute_r_script returned an error: {error}"
    assert df_result is not None, "execute_r_script did not return a DataFrame"

    expected_data = {
        'id': [2, 3],
        'name': ['Bob', 'Charlie'],
        'value': [20.3, 30.8],
        'is_active': [False, True]
    }
    expected_df = pd.DataFrame(expected_data)

    # Convert specific columns if necessary, pandas might read numbers as float by default from CSV
    if 'id' in df_result.columns:
         df_result['id'] = df_result['id'].astype(int) # Match type of expected_df
    if 'is_active' in df_result.columns and isinstance(df_result['is_active'].iloc[0], str):
        # Pandas might read TRUE/FALSE from CSV as strings if not perfectly formatted.
        # Convert to bool if they are strings 'TRUE'/'FALSE'
        df_result['is_active'] = df_result['is_active'].apply(lambda x: x.upper() == 'TRUE' if isinstance(x, str) else x).astype(bool)


    pd.testing.assert_frame_equal(
        df_result.reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_dtype=False, # Be lenient with types initially, can be tightened
        check_like=True # For columns order if not reset_index
    )

    # 5. Verify mock calls
    mock_subproc_run.assert_called_once()


def test_execute_r_script_r_error(mocker):
    """
    Tests execute_r_script when the R script itself encounters an error.
    """
    mock_subproc_run = mocker.patch('app.backend.app.subprocess.run')
    r_error_message = "Error in data.table: column 'non_existent_column' not found"
    mock_subproc_run.return_value = subprocess.CompletedProcess(
        args=['Rscript', 'dummy_path.R'], # The actual path is a temp file
        returncode=1,
        stdout="",
        stderr=r_error_message
    )

    rdata_file_path = "test_r_data.Rdata" # Assumed available
    if not os.path.exists(rdata_file_path):
        print(f"Warning: {rdata_file_path} not found for test_execute_r_script_r_error. Creating dummy.")
        with open(rdata_file_path, "wb") as f:
            f.write(b"dummy Rdata content")

    r_code_string = "active_df <- active_df[, non_existent_column]" # This code would cause an error
    target_object_name = "sample_df"

    df_result, error = execute_r_script(r_code_string, rdata_file_path, target_object_name)

    assert df_result is None
    assert error is not None
    assert r_error_message in error
    mock_subproc_run.assert_called_once()


def test_execute_r_script_rscript_not_found(mocker):
    """
    Tests execute_r_script when Rscript command is not found (FileNotFoundError).
    """
    mock_subproc_run = mocker.patch('app.backend.app.subprocess.run')
    mock_subproc_run.side_effect = FileNotFoundError("Mock: Rscript not found")

    rdata_file_path = "test_r_data.Rdata" # Assumed available
    if not os.path.exists(rdata_file_path):
        print(f"Warning: {rdata_file_path} not found for test_execute_r_script_rscript_not_found. Creating dummy.")
        with open(rdata_file_path, "wb") as f:
            f.write(b"dummy Rdata content")

    r_code_string = "active_df <- active_df[id > 1]"
    target_object_name = "sample_df"

    df_result, error = execute_r_script(r_code_string, rdata_file_path, target_object_name)

    assert df_result is None
    assert error is not None
    # Check against the specific error message returned by execute_r_script for FileNotFoundError
    assert "Rscript command not found" in error
    mock_subproc_run.assert_called_once()


# --- Tests for /query with agent_type: 'r_datatable' ---

def test_query_r_agent_success(client, mocker):
    """
    Tests the /query endpoint with agent_type='r_datatable' for a successful query.
    Mocks LLM calls and execute_r_script.
    """
    # 1. Setup global state (simulating a file has been uploaded and metadata extracted)
    # UPLOAD_FOLDER is managed by the client fixture
    rdata_filename = "test_r_data.Rdata"
    backend_app.current_uploaded_filepath = os.path.join(UPLOAD_FOLDER, rdata_filename)
    # Create a dummy file in UPLOAD_FOLDER as current_uploaded_filepath would point there
    if not os.path.exists(backend_app.current_uploaded_filepath):
        with open(backend_app.current_uploaded_filepath, "wb") as f:
            f.write(b"dummy Rdata for query test") # Content doesn't matter due to execute_r_script mock

    backend_app.current_metadata = {
        'table_name': 'sample_df',  # This is the R object name
        'columns': [
            {'name': 'id', 'type': 'INTEGER'},
            {'name': 'name', 'type': 'TEXT'},
            {'name': 'value', 'type': 'REAL'},
            {'name': 'is_active', 'type': 'BOOLEAN'}
        ]
    }
    backend_app.current_uploaded_filename = rdata_filename


    # 2. Mock LLM calls (openai.Completion.create)
    mock_openai_completion = mocker.patch('app.backend.app.openai.Completion.create')

    r_code_to_execute = "active_df <- active_df[id == 1]"
    summary_text = "Found one entry with id 1."

    r_code_response = mocker.Mock()
    r_code_response.choices = [mocker.Mock(text=r_code_to_execute)]

    summary_response_mock = mocker.Mock()
    summary_response_mock.choices = [mocker.Mock(text=summary_text)]

    mock_openai_completion.side_effect = [r_code_response, summary_response_mock]

    # 3. Mock execute_r_script
    mock_exec_r = mocker.patch('app.backend.app.execute_r_script')
    expected_df_data = {
        'id': [1], 'name': ['Alice'], 'value': [10.5], 'is_active': [True]
    }
    expected_exec_r_df = pd.DataFrame(expected_df_data)
    mock_exec_r.return_value = (expected_exec_r_df.copy(), None)

    # 4. Make Request to /query
    query_payload = {
        'naturalLanguageQuery': 'Get data for Alice where id is 1',
        'agent_type': 'r_datatable',
        'metadata': backend_app.current_metadata # Pass metadata as frontend would
    }
    response = client.post('/query', json=query_payload)

    # 5. Assert Response
    assert response.status_code == 200
    data = response.get_json()

    assert data['executed_query_text'] == r_code_to_execute
    assert data['natural_language_response'] == summary_text
    assert data['error'] is None

    # Verify results structure (Pandas to_dict('records') format)
    pd.testing.assert_frame_equal(
        pd.DataFrame(data['results']),
        expected_exec_r_df,
        check_dtype=False # API response might have types like int64 vs int
    )

    # 6. Verify Mock Calls
    assert mock_openai_completion.call_count == 2 # R code gen + Summary
    mock_exec_r.assert_called_once_with(r_code_to_execute, backend_app.current_uploaded_filepath, backend_app.current_metadata['table_name'])


def test_query_r_agent_self_correction(client, mocker):
    # Setup global state (similar to success test)
    rdata_filename = "test_r_data.Rdata"
    backend_app.current_uploaded_filepath = os.path.join(UPLOAD_FOLDER, rdata_filename)
    if not os.path.exists(backend_app.current_uploaded_filepath):
        with open(backend_app.current_uploaded_filepath, "wb") as f:
            f.write(b"dummy Rdata for query test")

    backend_app.current_metadata = {
        'table_name': 'sample_df',
        'columns': [ {'name': 'id', 'type': 'INTEGER'}, {'name': 'name', 'type': 'TEXT'} ]
    }
    backend_app.current_uploaded_filename = rdata_filename

    # Mock LLM Calls
    mock_openai_completion = mocker.patch('app.backend.app.openai.Completion.create')

    bad_r_code = "active_df <- active_df[bad_col == 1]" # Will fail
    good_r_code = "active_df <- active_df[id == 1]"   # Corrected
    summary_text = "Summary after correction."

    call1_r_gen = mocker.Mock(); call1_r_gen.choices = [mocker.Mock(text=bad_r_code)]
    call2_r_correct = mocker.Mock(); call2_r_correct.choices = [mocker.Mock(text=good_r_code)]
    call3_summary = mocker.Mock(); call3_summary.choices = [mocker.Mock(text=summary_text)]

    mock_openai_completion.side_effect = [call1_r_gen, call2_r_correct, call3_summary]

    # Mock execute_r_script (fail first, then succeed)
    mock_exec_r = mocker.patch('app.backend.app.execute_r_script')

    def exec_r_side_effect(r_code, filepath, obj_name):
        if r_code == bad_r_code:
            return (None, "Mock R Error: column 'bad_col' not found")
        elif r_code == good_r_code:
            df_data = {'id': [1], 'name': ['Alice']} # Simplified for this test
            return (pd.DataFrame(df_data), None)
        return (None, "Unexpected R code in mock_exec_r side_effect")

    mock_exec_r.side_effect = exec_r_side_effect

    # Make Request
    response = client.post('/query', json={
        'naturalLanguageQuery': 'Get data for id 1',
        'agent_type': 'r_datatable',
        'metadata': backend_app.current_metadata
    })

    # Assert Response
    assert response.status_code == 200
    data = response.get_json()
    assert data['executed_query_text'] == good_r_code
    assert data['natural_language_response'] == summary_text
    assert data['error'] is None
    expected_results_df = pd.DataFrame({'id': [1], 'name': ['Alice']})
    pd.testing.assert_frame_equal(pd.DataFrame(data['results']), expected_results_df, check_dtype=False)

    # Verify Mock Calls
    assert mock_openai_completion.call_count == 3 # R gen, R correct, Summary
    assert mock_exec_r.call_count == 2


def test_query_r_agent_llm_fails_generation(client, mocker):
    # Setup global state
    backend_app.current_uploaded_filepath = os.path.join(UPLOAD_FOLDER, "test_r_data.Rdata")
    if not os.path.exists(backend_app.current_uploaded_filepath):
        with open(backend_app.current_uploaded_filepath, "wb") as f: f.write(b"dummy")
    backend_app.current_metadata = {'table_name': 'sample_df', 'columns': [{'name': 'id', 'type': 'INTEGER'}]}
    backend_app.current_uploaded_filename = "test_r_data.Rdata"

    # Mock LLM to fail on first call (R code generation)
    mock_openai_completion = mocker.patch('app.backend.app.openai.Completion.create')
    mock_openai_completion.side_effect = OpenAIError("LLM simulation error during R code generation.")

    response = client.post('/query', json={
        'naturalLanguageQuery': 'Any query',
        'agent_type': 'r_datatable',
        'metadata': backend_app.current_metadata
    })

    assert response.status_code == 500 # OpenAIError should lead to 500
    data = response.get_json()
    assert "OpenAI API error" in data['error']
    assert "LLM simulation error" in data['error']
    assert data['executed_query_text'] == "" # Since generation failed

def test_query_r_agent_llm_returns_empty_r_code(client, mocker):
    # Setup global state
    backend_app.current_uploaded_filepath = os.path.join(UPLOAD_FOLDER, "test_r_data.Rdata")
    if not os.path.exists(backend_app.current_uploaded_filepath):
        with open(backend_app.current_uploaded_filepath, "wb") as f: f.write(b"dummy")
    backend_app.current_metadata = {'table_name': 'sample_df', 'columns': [{'name': 'id', 'type': 'INTEGER'}]}
    backend_app.current_uploaded_filename = "test_r_data.Rdata"

    mock_openai_completion = mocker.patch('app.backend.app.openai.Completion.create')
    empty_r_code_response = mocker.Mock()
    empty_r_code_response.choices = [mocker.Mock(text="")] # LLM returns empty string
    mock_openai_completion.return_value = empty_r_code_response

    response = client.post('/query', json={
        'naturalLanguageQuery': 'Any query',
        'agent_type': 'r_datatable',
        'metadata': backend_app.current_metadata
    })

    assert response.status_code == 500 # Or 400 depending on desired behavior for empty code
    data = response.get_json()
    assert "LLM did not return R code" in data['error']
    assert data['executed_query_text'] == ""


def test_query_r_agent_persistent_r_error(client, mocker):
    # Setup global state
    backend_app.current_uploaded_filepath = os.path.join(UPLOAD_FOLDER, "test_r_data.Rdata")
    if not os.path.exists(backend_app.current_uploaded_filepath):
        with open(backend_app.current_uploaded_filepath, "wb") as f: f.write(b"dummy")
    backend_app.current_metadata = {'table_name': 'sample_df', 'columns': [{'name': 'id', 'type': 'INTEGER'}]}
    backend_app.current_uploaded_filename = "test_r_data.Rdata"

    # Mock LLM calls
    mock_openai_completion = mocker.patch('app.backend.app.openai.Completion.create')
    initial_bad_r_code = "active_df <- active_df[, non_existent_col1]"
    corrected_bad_r_code = "active_df <- active_df[, non_existent_col2]" # Still bad

    call1_r_gen = mocker.Mock(); call1_r_gen.choices = [mocker.Mock(text=initial_bad_r_code)]
    call2_r_correct = mocker.Mock(); call2_r_correct.choices = [mocker.Mock(text=corrected_bad_r_code)]
    # No summary call needed as it errors out before that
    mock_openai_completion.side_effect = [call1_r_gen, call2_r_correct]

    # Mock execute_r_script to fail both times
    mock_exec_r = mocker.patch('app.backend.app.execute_r_script')
    error_msg1 = "R Error: non_existent_col1 not found"
    error_msg2 = "R Error: non_existent_col2 not found"
    mock_exec_r.side_effect = [
        (None, error_msg1), # Fails on initial_bad_r_code
        (None, error_msg2)  # Fails on corrected_bad_r_code
    ]

    response = client.post('/query', json={
        'naturalLanguageQuery': 'A query that leads to persistent errors',
        'agent_type': 'r_datatable',
        'metadata': backend_app.current_metadata
    })

    assert response.status_code == 400 # Should be a client-type error after correction fails
    data = response.get_json()
    assert data['executed_query_text'] == corrected_bad_r_code
    assert error_msg2 in data['error'] # Error from the second R execution attempt
    assert data['results'] is None
    assert data['natural_language_response'] is None

    assert mock_openai_completion.call_count == 2 # R gen + R correct
    assert mock_exec_r.call_count == 2
