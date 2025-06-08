import pytest
import os
import pandas as pd
from app.backend.app import execute_pandas_code, UPLOAD_FOLDER
# Note: UPLOAD_FOLDER might not be directly used by these unit tests if tmp_path is used for all file ops,
# but for endpoint tests, the app will use its configured UPLOAD_FOLDER.
from app.backend.app import app # For client fixture and setting globals
from app.backend import app as backend_app # For setting globals directly
from openai import OpenAIError # For testing LLM failure scenarios

@pytest.fixture
def client(tmp_path):
    # Configure UPLOAD_FOLDER to use tmp_path for this test session
    # This ensures that file operations during tests are isolated.
    # Important: app.config is modified here.
    original_upload_folder = app.config['UPLOAD_FOLDER']
    temp_upload_folder = tmp_path / "test_uploads"
    os.makedirs(temp_upload_folder, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = str(temp_upload_folder)

    # Update the UPLOAD_FOLDER imported directly in the test module if it was top-level
    # For this structure, modifying app.config['UPLOAD_FOLDER'] should be enough
    # as the app internally uses app.config.

    app.config['TESTING'] = True

    # Reset relevant global states before each test
    backend_app.current_metadata = None
    backend_app.current_uploaded_filename = None
    backend_app.current_uploaded_filepath = None

    with app.test_client() as client:
        yield client

    # Teardown: Clean up temp_upload_folder contents (though tmp_path itself is auto-cleaned)
    # and restore original UPLOAD_FOLDER if it matters for other test modules (it might).
    for item in temp_upload_folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            # Add rmtree if you expect subdirectories, for now just basic cleanup
            pass
    app.config['UPLOAD_FOLDER'] = original_upload_folder


def test_execute_pandas_success(tmp_path):
    """
    Tests successful execution of Pandas code via execute_pandas_code.
    """
    file_path = tmp_path / "test_data.csv"
    csv_content = "id,name,value,category\n1,Alice,100,A\n2,Bob,200,B\n3,Charlie,150,A\n4,David,250,C"
    file_path.write_text(csv_content)

    pandas_code_string = "df = df[df['value'] > 150]"

    df_result, error = execute_pandas_code(pandas_code_string, str(file_path))

    assert error is None, f"execute_pandas_code returned an error: {error}"
    assert df_result is not None, "execute_pandas_code did not return a DataFrame"

    expected_data = {
        'id': [2, 4],
        'name': ['Bob', 'David'],
        'value': [200, 250],
        'category': ['B', 'C']
    }
    expected_df = pd.DataFrame(expected_data)

    # Ensure 'id' column is int for comparison, as read_csv might make it float if there were NaNs (not in this case)
    # or if all values could be coerced. Here, it should be fine, but good to be explicit.
    if 'id' in df_result.columns:
         df_result['id'] = df_result['id'].astype(int)
    if 'value' in df_result.columns:
         df_result['value'] = df_result['value'].astype(int)


    pd.testing.assert_frame_equal(
        df_result.reset_index(drop=True),
        expected_df.reset_index(drop=True)
    )


def test_execute_pandas_code_execution_error(tmp_path):
    """
    Tests execute_pandas_code when the provided Pandas code string causes an execution error.
    """
    file_path = tmp_path / "test_data.csv"
    csv_content = "id,name,value,category\n1,Alice,100,A\n2,Bob,200,B"
    file_path.write_text(csv_content)

    pandas_code_string = "df = df[df['non_existent_column'] == 'A']" # This will cause a KeyError

    df_result, error = execute_pandas_code(pandas_code_string, str(file_path))

    assert df_result is None
    assert error is not None
    # Check for specific error message components from execute_pandas_code
    assert "Pandas code execution error (KeyError)" in error or "non_existent_column" in error


def test_execute_pandas_unsupported_file_type(tmp_path):
    """
    Tests execute_pandas_code with an unsupported file type (e.g., .sqlite).
    """
    file_path = tmp_path / "test_data.sqlite"
    file_path.write_text("dummy sqlite content") # Content doesn't matter

    pandas_code_string = "df = df.head(1)" # Simple code, won't be reached

    df_result, error = execute_pandas_code(pandas_code_string, str(file_path))

    assert df_result is None
    assert error is not None
    assert "Unsupported file type for Pandas execution" in error


def test_execute_pandas_file_not_found():
    """
    Tests execute_pandas_code when the specified data file does not exist.
    """
    non_existent_file_path = "non_existent_file.csv"
    pandas_code_string = "df = df.head(1)"

    df_result, error = execute_pandas_code(pandas_code_string, non_existent_file_path)

    assert df_result is None
    assert error is not None
    assert "Data file not found" in error
    assert non_existent_file_path in error


def test_execute_pandas_code_syntax_error(tmp_path):
    """
    Tests execute_pandas_code with a Python syntax error in the pandas_code_string.
    """
    file_path = tmp_path / "test_data.csv"
    csv_content = "id,name,value,category\n1,Alice,100,A"
    file_path.write_text(csv_content)

    pandas_code_string = "df = df[df['value'] > 150] This is not valid Python"

    df_result, error = execute_pandas_code(pandas_code_string, str(file_path))

    assert df_result is None
    assert error is not None
    assert "Pandas code execution error (SyntaxError)" in error


def test_execute_pandas_code_name_error_in_df_manipulation(tmp_path):
    """
    Tests execute_pandas_code where the pandas code references an undefined variable
    during DataFrame manipulation (leading to NameError within exec).
    """
    file_path = tmp_path / "test_data.csv"
    csv_content = "id,name,value,category\n1,Alice,100,A"
    file_path.write_text(csv_content)

    # some_undefined_variable is not defined in the exec scope
    pandas_code_string = "df = df[df['value'] > some_undefined_variable]"

    df_result, error = execute_pandas_code(pandas_code_string, str(file_path))

    assert df_result is None
    assert error is not None
    assert "Pandas code execution error (NameError)" in error
    assert "some_undefined_variable" in error

def test_execute_pandas_code_result_not_dataframe(tmp_path):
    """
    Tests execute_pandas_code where the executed code does not result in a DataFrame
    assigned to the expected variable name.
    """
    file_path = tmp_path / "test_data.csv"
    csv_content = "id,name,value,category\n1,Alice,100,A"
    file_path.write_text(csv_content)

    pandas_code_string = "df = 123" # Assigns an int, not a DataFrame

    df_result, error = execute_pandas_code(pandas_code_string, str(file_path))

    assert df_result is None
    assert error is not None
    assert "did not result in a DataFrame" in error
    assert "or the result is not a DataFrame" in error

def test_execute_pandas_code_custom_df_name_success(tmp_path):
    """
    Tests successful execution with a custom df_name.
    """
    file_path = tmp_path / "test_data.csv"
    csv_content = "id,name,value,category\n1,Alice,100,A\n2,Bob,200,B"
    file_path.write_text(csv_content)

    custom_name = "my_custom_df"
    pandas_code_string = f"{custom_name} = {custom_name}[{custom_name}['value'] == 100]"

    df_result, error = execute_pandas_code(pandas_code_string, str(file_path), df_name=custom_name)

    assert error is None
    assert df_result is not None
    assert len(df_result) == 1
    assert df_result['name'].iloc[0] == 'Alice'

def test_execute_pandas_code_custom_df_name_result_mismatch(tmp_path):
    """
    Tests when custom df_name is used but the code assigns to a different variable.
    """
    file_path = tmp_path / "test_data.csv"
    csv_content = "id,name,value,category\n1,Alice,100,A"
    file_path.write_text(csv_content)

    custom_name = "my_custom_df"
    # Code assigns to 'df', but execute_pandas_code will look for 'my_custom_df' in local_scope
    pandas_code_string = f"df = {custom_name}[{custom_name}['value'] == 100]"

    df_result, error = execute_pandas_code(pandas_code_string, str(file_path), df_name=custom_name)

    assert df_result is None
    assert error is not None
    assert f"did not result in a DataFrame named '{custom_name}'" in error


# --- Tests for /query with agent_type: 'python_pandas' ---

def test_query_pandas_agent_success(client, mocker, tmp_path):
    """
    Tests the /query endpoint with agent_type='python_pandas' for a successful query.
    """
    # 1. Prepare a test CSV file in the test-specific UPLOAD_FOLDER
    test_csv_filename = "test_data_for_query.csv"
    # client fixture has updated app.config['UPLOAD_FOLDER'] to be under tmp_path
    csv_path_in_upload_folder = tmp_path / "test_uploads" / test_csv_filename

    csv_content = "id,name,value,category\n1,Alice,100,A\n2,Bob,200,B\n3,Charlie,150,A\n4,David,250,C"
    csv_path_in_upload_folder.write_text(csv_content)

    # 2. Setup global state in the app context
    backend_app.current_uploaded_filepath = str(csv_path_in_upload_folder)
    backend_app.current_uploaded_filename = test_csv_filename
    backend_app.current_metadata = {
        'table_name': 'test_data_for_query',
        'columns': [
            {'name': 'id', 'type': 'INTEGER'}, {'name': 'name', 'type': 'TEXT'},
            {'name': 'value', 'type': 'INTEGER'}, {'name': 'category', 'type': 'TEXT'}
        ]
    }

    # 3. Mock LLM calls
    mock_openai_completion = mocker.patch('app.backend.app.openai.Completion.create')
    pandas_code_to_execute = "df = df[df['value'] > 150]"
    summary_text = "Filtered data where value is greater than 150."

    pandas_code_response = mocker.Mock()
    pandas_code_response.choices = [mocker.Mock(text=pandas_code_to_execute)]

    summary_response_mock = mocker.Mock()
    summary_response_mock.choices = [mocker.Mock(text=summary_text)]

    mock_openai_completion.side_effect = [pandas_code_response, summary_response_mock]

    # 4. Mock execute_pandas_code (to unit test endpoint logic)
    mock_exec_pandas = mocker.patch('app.backend.app.execute_pandas_code')
    expected_df_data = {'id': [2, 4], 'name': ['Bob', 'David'], 'value': [200, 250], 'category': ['B', 'C']}
    expected_exec_pandas_df = pd.DataFrame(expected_df_data)
    mock_exec_pandas.return_value = (expected_exec_pandas_df.copy(), None)

    # 5. Make Request to /query
    query_payload = {
        'naturalLanguageQuery': 'Get data where value > 150',
        'agent_type': 'python_pandas',
        'metadata': backend_app.current_metadata
    }
    response = client.post('/query', json=query_payload)

    # 6. Assert Response
    assert response.status_code == 200
    data = response.get_json()

    assert data['executed_query_text'] == pandas_code_to_execute
    assert data['natural_language_response'] == summary_text
    assert data['error'] is None

    pd.testing.assert_frame_equal(
        pd.DataFrame(data['results']),
        expected_exec_pandas_df,
        check_dtype=False
    )

    # 7. Verify Mock Calls
    assert mock_openai_completion.call_count == 2
    mock_exec_pandas.assert_called_once_with(pandas_code_to_execute, str(csv_path_in_upload_folder), "df")


def test_query_pandas_agent_self_correction(client, mocker, tmp_path):
    test_csv_filename = "test_data_for_correction.csv"
    csv_path_in_upload_folder = tmp_path / "test_uploads" / test_csv_filename
    csv_path_in_upload_folder.write_text("id,name\n1,Alice\n2,Bob")

    backend_app.current_uploaded_filepath = str(csv_path_in_upload_folder)
    backend_app.current_uploaded_filename = test_csv_filename
    backend_app.current_metadata = {
        'table_name': 'test_data_for_correction',
        'columns': [{'name': 'id', 'type': 'INTEGER'}, {'name': 'name', 'type': 'TEXT'}]
    }

    mock_openai_completion = mocker.patch('app.backend.app.openai.Completion.create')
    bad_pandas_code = "df = df[df['bad_col'] == 1]"
    good_pandas_code = "df = df[df['id'] == 1]"
    summary_text = "Summary after correction."

    call1_pandas_gen = mocker.Mock(); call1_pandas_gen.choices = [mocker.Mock(text=bad_pandas_code)]
    call2_pandas_correct = mocker.Mock(); call2_pandas_correct.choices = [mocker.Mock(text=good_pandas_code)]
    call3_summary = mocker.Mock(); call3_summary.choices = [mocker.Mock(text=summary_text)]
    mock_openai_completion.side_effect = [call1_pandas_gen, call2_pandas_correct, call3_summary]

    mock_exec_pandas = mocker.patch('app.backend.app.execute_pandas_code')
    def exec_pandas_side_effect(code, filepath, df_name):
        if code == bad_pandas_code:
            return (None, "Mock Pandas Error: column 'bad_col' not found")
        elif code == good_pandas_code:
            df_data = {'id': [1], 'name': ['Alice']}
            return (pd.DataFrame(df_data), None)
        return (None, "Unexpected Pandas code in mock_exec_pandas side_effect")
    mock_exec_pandas.side_effect = exec_pandas_side_effect

    response = client.post('/query', json={
        'naturalLanguageQuery': 'Get id 1',
        'agent_type': 'python_pandas',
        'metadata': backend_app.current_metadata
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data['executed_query_text'] == good_pandas_code
    assert data['natural_language_response'] == summary_text
    assert data['error'] is None
    expected_results_df = pd.DataFrame({'id': [1], 'name': ['Alice']})
    pd.testing.assert_frame_equal(pd.DataFrame(data['results']), expected_results_df, check_dtype=False)

    assert mock_openai_completion.call_count == 3
    assert mock_exec_pandas.call_count == 2


def test_query_pandas_agent_unsupported_file(client, mocker, tmp_path):
    # Setup global state with an unsupported file type for Pandas agent
    unsupported_filename = "test_data.sqlite"
    # The client fixture sets UPLOAD_FOLDER to a temp path
    sqlite_path_in_upload_folder = tmp_path / "test_uploads" / unsupported_filename
    sqlite_path_in_upload_folder.write_text("dummy sqlite content")


    backend_app.current_uploaded_filepath = str(sqlite_path_in_upload_folder)
    backend_app.current_uploaded_filename = unsupported_filename # Crucial for the check
    backend_app.current_metadata = {
        'table_name': 'some_table',
        'columns': [{'name': 'id', 'type': 'INTEGER'}]
    }

    # LLM and execute_pandas_code mocks might not be strictly needed if file check is early
    # but good for robustness if that logic changes.
    mocker.patch('app.backend.app.openai.Completion.create')
    mocker.patch('app.backend.app.execute_pandas_code')


    response = client.post('/query', json={
        'naturalLanguageQuery': 'Any query',
        'agent_type': 'python_pandas',
        'metadata': backend_app.current_metadata
    })

    assert response.status_code == 400
    data = response.get_json()
    assert "Python Pandas agent currently only supports CSV and Parquet files" in data['error']
    assert data['results'] is None
    assert data['executed_query_text'] is None # or "" depending on exact error path
    assert data['natural_language_response'] is None
