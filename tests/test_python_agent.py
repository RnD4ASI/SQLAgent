import unittest
import pandas as pd
import os
import tempfile
import shutil
import sys

# Adjust path to import from app.backend.app
# This assumes the tests are run from the project root directory (e.g., using `python -m unittest discover tests`)
# Or that PYTHONPATH is set up appropriately.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.backend.app import execute_python_pandas_code
from pandas.testing import assert_frame_equal
from unittest.mock import patch, Mock
import json

# Flask app related imports for integration tests
from app.backend.app import app # The Flask app instance
# To control global state during tests if necessary:
import app.backend.app as backend_app_module


class TestExecutePythonPandasCode(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_csv_path = os.path.join(self.test_dir, "sample.csv")
        self.sample_parquet_path = os.path.join(self.test_dir, "sample.parquet")
        self.unsupported_txt_path = os.path.join(self.test_dir, "sample.txt")

        self.data = {'name': ['Alice', 'Bob', 'Charlie'],
                     'age': [30, 24, 35],
                     'city': ['New York', 'Los Angeles', 'Chicago']}
        self.sample_df = pd.DataFrame(self.data)

        self.sample_df.to_csv(self.sample_csv_path, index=False)
        self.sample_df.to_parquet(self.sample_parquet_path, index=False)
        with open(self.unsupported_txt_path, 'w') as f:
            f.write("This is not a dataframe file.")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_successful_execution_csv(self):
        code = "df['age_plus_ten'] = df['age'] + 10\ndf = df[df['age'] > 25]"
        expected_data = {'name': ['Alice', 'Charlie'],
                         'age': [30, 35],
                         'city': ['New York', 'Chicago'],
                         'age_plus_ten': [40, 45]}
        expected_df = pd.DataFrame(expected_data)

        result_df, error = execute_python_pandas_code(code, self.sample_csv_path, dataframe_name='df')

        self.assertIsNone(error)
        self.assertIsNotNone(result_df)
        assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    def test_successful_execution_parquet(self):
        code = "df['is_young'] = df['age'] < 30\ndf = df[df['city'] == 'Los Angeles']"
        # Bob is 24, < 30, Los Angeles
        expected_data = {'name': ['Bob'],
                         'age': [24],
                         'city': ['Los Angeles'],
                         'is_young': [True]}
        expected_df = pd.DataFrame(expected_data)

        result_df, error = execute_python_pandas_code(code, self.sample_parquet_path, dataframe_name='df')

        self.assertIsNone(error)
        self.assertIsNotNone(result_df)
        assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    def test_empty_result_dataframe(self):
        code = "df = df[df['age'] > 100]" # No one is older than 100
        expected_df = pd.DataFrame(columns=['name', 'age', 'city']) # Empty DF with original columns

        result_df, error = execute_python_pandas_code(code, self.sample_csv_path, dataframe_name='df')

        self.assertIsNone(error)
        self.assertIsNotNone(result_df)
        self.assertTrue(result_df.empty)
        # Check if columns are preserved as pandas behavior
        self.assertListEqual(list(result_df.columns), list(expected_df.columns))


    def test_dataframe_renaming_or_assignment(self):
        # Test with 'df1' as the dataframe name passed to the function
        # and the code internally uses 'df1'
        code = "df1 = df1[df1['age'] > 30]\ndf1['city_upper'] = df1['city'].str.upper()"
        # Charlie is 35 > 30
        expected_data = {'name': ['Charlie'],
                         'age': [35],
                         'city': ['Chicago'],
                         'city_upper': ['CHICAGO']}
        expected_df = pd.DataFrame(expected_data)

        result_df, error = execute_python_pandas_code(code, self.sample_csv_path, dataframe_name='df1')

        self.assertIsNone(error)
        self.assertIsNotNone(result_df)
        assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


    def test_execution_error_syntax(self):
        code = "df['age'] = df['age' + 10" # Syntax error: missing closing bracket

        result_df, error = execute_python_pandas_code(code, self.sample_csv_path, dataframe_name='df')

        self.assertIsNone(result_df)
        self.assertIsNotNone(error)
        # Check for the specific error message pattern from the script
        self.assertIn("Error during Python code execution:", error)
        self.assertIn("never closed", error) # Specific part of a syntax error message

    def test_execution_error_runtime_pandas(self):
        code = "df['new_col'] = df['non_existent_col'] * 2"

        result_df, error = execute_python_pandas_code(code, self.sample_csv_path, dataframe_name='df')

        self.assertIsNone(result_df)
        self.assertIsNotNone(error)
        # The error from pandas might be KeyError or similar, wrapped in the execution message.
        self.assertTrue("KeyError" in error or "non_existent_col" in error or "column not found" in error.lower())


    def test_input_file_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "does_not_exist.csv")
        code = "df['age_plus_one'] = df['age'] + 1"

        result_df, error = execute_python_pandas_code(code, non_existent_path, dataframe_name='df')

        self.assertIsNone(result_df)
        self.assertIsNotNone(error)
        self.assertIn("Error loading data:", error)
        self.assertIn("No such file or directory", error) # More general part of FileNotFoundError message

    def test_code_removes_dataframe(self):
        code = "del df"

        result_df, error = execute_python_pandas_code(code, self.sample_csv_path, dataframe_name='df')

        self.assertIsNone(result_df)
        self.assertIsNotNone(error)
        self.assertIn("DataFrame 'df' not found after code execution", error)

    def test_code_changes_df_to_non_dataframe(self):
        code = "df = 'I am not a dataframe anymore'"
        result_df, error = execute_python_pandas_code(code, self.sample_csv_path, dataframe_name='df')

        self.assertIsNone(result_df)
        self.assertIsNotNone(error)
        self.assertIn("Resulting object 'df' is not a Pandas DataFrame", error)


    def test_unsupported_file_type(self):
        code = "df['name_lower'] = df['name'].str.lower()"
        result_df, error = execute_python_pandas_code(code, self.unsupported_txt_path, dataframe_name='df')

        self.assertIsNone(result_df)
        self.assertIsNotNone(error)
        self.assertIn("Unsupported file type", error)
        self.assertIn("Only CSV and Parquet are supported", error)

if __name__ == '__main__':
    unittest.main()


class TestPythonPandasAgentIntegration(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

        self.test_upload_dir = tempfile.mkdtemp()
        self.original_upload_folder = app.config['UPLOAD_FOLDER']
        app.config['UPLOAD_FOLDER'] = self.test_upload_dir

        # Store and reset global state from app.backend.app
        self.original_openai_api_key = backend_app_module.OPENAI_API_KEY
        self.original_current_uploaded_filepath = backend_app_module.current_uploaded_filepath
        self.original_current_uploaded_filename = backend_app_module.current_uploaded_filename
        self.original_current_metadata = backend_app_module.current_metadata
        self.original_last_successful_df = backend_app_module.last_successful_df

        backend_app_module.OPENAI_API_KEY = "test_api_key" # Ensure API key check passes
        backend_app_module.current_uploaded_filepath = None
        backend_app_module.current_uploaded_filename = None
        backend_app_module.current_metadata = None
        backend_app_module.last_successful_df = None

        # Sample data
        self.sample_data = {'name': ['Alice', 'Bob', 'Charlie'],
                            'age': [30, 24, 35],
                            'city': ['New York', 'Los Angeles', 'Chicago']}
        self.sample_df = pd.DataFrame(self.sample_data)
        self.sample_csv_filename = "sample_test.csv"
        self.sample_parquet_filename = "sample_test.parquet"

        self.sample_csv_filepath = os.path.join(self.test_upload_dir, self.sample_csv_filename)
        self.sample_parquet_filepath = os.path.join(self.test_upload_dir, self.sample_parquet_filename)

        self.sample_df.to_csv(self.sample_csv_filepath, index=False)
        self.sample_df.to_parquet(self.sample_parquet_filepath, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_upload_dir)
        app.config['UPLOAD_FOLDER'] = self.original_upload_folder

        # Restore global state
        backend_app_module.OPENAI_API_KEY = self.original_openai_api_key
        backend_app_module.current_uploaded_filepath = self.original_current_uploaded_filepath
        backend_app_module.current_uploaded_filename = self.original_current_uploaded_filename
        backend_app_module.current_metadata = self.original_current_metadata
        backend_app_module.last_successful_df = self.original_last_successful_df
        app.config['TESTING'] = False


    def _simulate_upload_and_metadata(self, filename, file_type='csv'):
        backend_app_module.current_uploaded_filepath = os.path.join(self.test_upload_dir, filename)
        backend_app_module.current_uploaded_filename = filename

        table_name = filename.split('.')[0]
        columns_data = []
        # Use the known full schema from self.sample_df for robustness in tests
        for col_name_iter in self.sample_df.columns: # Iterate in defined order
            col_series = self.sample_df[col_name_iter]
            col_type = 'TEXT' # Default
            if pd.api.types.is_integer_dtype(col_series): col_type = 'INTEGER'
            elif pd.api.types.is_float_dtype(col_series): col_type = 'REAL'
            elif pd.api.types.is_bool_dtype(col_series): col_type = 'BOOLEAN'
            elif pd.api.types.is_datetime64_any_dtype(col_series): col_type = 'DATETIME'
            columns_data.append({'name': col_name_iter, 'type': col_type})

        backend_app_module.current_metadata = {
            'table_name': table_name,
            'columns': columns_data
        }

    @patch('app.backend.app.openai.Completion.create')
    def test_successful_pandas_query_csv(self, mock_openai_completion):
        self._simulate_upload_and_metadata(self.sample_csv_filename, file_type='csv')

        mock_response = Mock()
        mock_response.choices = [Mock()]
        # The table_name for sample_csv_filename will be 'sample_test'
        mock_code = "sample_test = sample_test[sample_test['age'] > 25]"
        mock_response.choices[0].text = mock_code

        # Mock for the summarization call
        mock_summary_response = Mock()
        mock_summary_response.choices = [Mock()]
        mock_summary_response.choices[0].text = "Mocked summary: 2 people older than 25."

        mock_openai_completion.side_effect = [mock_response, mock_summary_response]


        response = self.client.post('/query', json={
            'naturalLanguageQuery': 'show people older than 25',
            'agent_type': 'python_pandas',
            'metadata': backend_app_module.current_metadata
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['executed_query_text'], mock_code)
        self.assertIsNotNone(data['results'])
        self.assertEqual(len(data['results']), 2) # Alice (30) and Charlie (35)
        self.assertIn("Alice", [r['name'] for r in data['results']])
        self.assertIsNotNone(data['natural_language_response']) # Mocked LLM summary also happens

    @patch('app.backend.app.openai.Completion.create')
    def test_successful_pandas_query_parquet(self, mock_openai_completion):
        self._simulate_upload_and_metadata(self.sample_parquet_filename, file_type='parquet')

        mock_response = Mock()
        mock_response.choices = [Mock()]
        # The table_name for sample_parquet_filename will be 'sample_test'
        mock_code = "sample_test = sample_test[sample_test['city'] == 'Chicago']"
        mock_response.choices[0].text = mock_code

        # Mock for the summarization call
        mock_summary_response = Mock()
        mock_summary_response.choices = [Mock()]
        mock_summary_response.choices[0].text = "Mocked summary: 1 person from Chicago."

        mock_openai_completion.side_effect = [mock_response, mock_summary_response]

        response = self.client.post('/query', json={
            'naturalLanguageQuery': "show people from Chicago",
            'agent_type': 'python_pandas',
            'metadata': backend_app_module.current_metadata
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['executed_query_text'], mock_code)
        self.assertEqual(len(data['results']), 1)
        self.assertEqual(data['results'][0]['name'], 'Charlie')

    @patch('app.backend.app.openai.Completion.create')
    @patch('app.backend.app.execute_python_pandas_code')
    def test_pandas_query_llm_error_correction_flow(self, mock_execute_code, mock_openai_completion):
        self._simulate_upload_and_metadata(self.sample_csv_filename, file_type='csv')

        mock_faulty_code = "df = df[df['age'] > 25"
        mock_good_code = "df = df[df['age'] > 30]"

        mock_response_faulty = Mock()
        mock_response_faulty.choices = [Mock()]
        mock_response_faulty.choices[0].text = mock_faulty_code

        mock_response_good = Mock()
        mock_response_good.choices = [Mock()]
        mock_response_good.choices[0].text = mock_good_code

        mock_summary_response = Mock()
        mock_summary_response.choices = [Mock()]
        mock_summary_response.choices[0].text = "Mocked summary for correction flow."

        mock_openai_completion.side_effect = [mock_response_faulty, mock_response_good, mock_summary_response]

        successful_result_df = self.sample_df[self.sample_df['age'] > 30].copy()
        mock_execute_code.side_effect = [
            (None, "Syntax error in submitted code."),
            (successful_result_df, None)
        ]

        response = self.client.post('/query', json={
            'naturalLanguageQuery': 'show people older than 30',
            'agent_type': 'python_pandas',
            'metadata': backend_app_module.current_metadata
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['executed_query_text'], mock_good_code)
        self.assertEqual(len(data['results']), 1)
        self.assertEqual(data['results'][0]['name'], 'Charlie')
        self.assertEqual(mock_openai_completion.call_count, 3) # Gen + Correction + Summary
        self.assertEqual(mock_execute_code.call_count, 2)

    @patch('app.backend.app.openai.Completion.create')
    @patch('app.backend.app.execute_python_pandas_code')
    def test_pandas_query_execution_error_after_correction(self, mock_execute_code, mock_openai_completion):
        self._simulate_upload_and_metadata(self.sample_csv_filename, file_type='csv')

        mock_code_attempt1 = "df = df[df['age'] / 0]"
        mock_code_attempt2 = "df = df['non_existent_column']"

        mock_response1 = Mock()
        mock_response1.choices = [Mock()]
        mock_response1.choices[0].text = mock_code_attempt1

        mock_response2 = Mock()
        mock_response2.choices = [Mock()]
        mock_response2.choices[0].text = mock_code_attempt2

        mock_openai_completion.side_effect = [mock_response1, mock_response2]
        mock_execute_code.return_value = (None, "Persistent runtime error")

        response = self.client.post('/query', json={
            'naturalLanguageQuery': 'try something that fails twice',
            'agent_type': 'python_pandas',
            'metadata': backend_app_module.current_metadata
        })

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIsNotNone(data['error'])
        self.assertEqual(data['executed_query_text'], mock_code_attempt2)

    def test_pandas_query_no_file_uploaded(self):
        backend_app_module.current_uploaded_filepath = None
        backend_app_module.current_metadata = None

        response = self.client.post('/query', json={
            'naturalLanguageQuery': 'any query',
            'agent_type': 'python_pandas'
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('No file has been uploaded', data['error'])

    def test_pandas_query_missing_metadata_empty_cols(self):
        self._simulate_upload_and_metadata(self.sample_csv_filename, file_type='csv')
        backend_app_module.current_metadata['columns'] = [] # Empty columns

        response = self.client.post('/query', json={
            'naturalLanguageQuery': 'any query',
            'agent_type': 'python_pandas',
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('Column metadata is missing', data['error'])

    def test_pandas_query_missing_metadata_no_cols_key(self):
        self._simulate_upload_and_metadata(self.sample_csv_filename, file_type='csv')
        del backend_app_module.current_metadata['columns'] # No 'columns' key

        response = self.client.post('/query', json={
            'naturalLanguageQuery': 'any query',
            'agent_type': 'python_pandas',
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('Column metadata is missing', data['error'])

    @patch('app.backend.app.openai.Completion.create')
    def test_pandas_query_llm_fails_to_generate_code(self, mock_openai_completion):
        self._simulate_upload_and_metadata(self.sample_csv_filename, file_type='csv')

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].text = ""
        mock_openai_completion.return_value = mock_response

        response = self.client.post('/query', json={
            'naturalLanguageQuery': 'any query',
            'agent_type': 'python_pandas',
            'metadata': backend_app_module.current_metadata
        })
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('LLM did not return Python Pandas code', data['error'])

    @patch('app.backend.app.openai.Completion.create') # Keep it simple first
    def test_pandas_query_llm_api_error(self, mock_openai_completion):
        # Simulate an APIError being raised by the OpenAI call
        # Ensure openai.APIError is a valid import in app.backend.app or this test will need adjustment
        # For testing, we can assume app.backend.app imports it if it tries to catch it.
        # The mock will raise it. If app.backend.app cannot import it, then the except block itself is flawed.
        from openai import APIError as OpenAI_APIError # Should be available
        import httpx # For creating a dummy request object

        dummy_request = httpx.Request("GET", "https://api.openai.com/v1/completions") # A realistic looking dummy request

        # Instantiate APIError with the 'request' argument
        # The message can be anything for the test.
        # The body argument can be omitted if it defaults to None or is not strictly needed by str(APIError).
        mock_openai_completion.side_effect = OpenAI_APIError(
            message="Simulated OpenAI API Error",
            request=dummy_request,
            body=None # Or provide a dummy response body if needed by error formatting
        )

        self._simulate_upload_and_metadata(self.sample_csv_filename, file_type='csv')

        response = self.client.post('/query', json={
            'naturalLanguageQuery': 'any query',
            'agent_type': 'python_pandas',
            'metadata': backend_app_module.current_metadata
        })
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('OpenAI API error: Simulated OpenAI API Error', data['error'])
