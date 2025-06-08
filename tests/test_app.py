import unittest
import os
import sys
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory (project root) to the Python path
# to allow imports from the 'app' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io # Import the io module
import sqlite3 # Import sqlite3
import openai # Import openai for OpenAIError
from app.backend.app import app, execute_duckdb_query # Import Flask app and helper
import app.backend.app as backend_app # Import the app module itself for accessing globals

# Global variable to store the path to the test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
# Ensure TEST_DATA_DIR exists
os.makedirs(TEST_DATA_DIR, exist_ok=True)


class TestApp(unittest.TestCase):
    def setUp(self):
        """Set up test client and other resources."""
        self.app = app.test_client()
        self.app.testing = True
        
        # Ensure the app.config['UPLOAD_FOLDER'] is set to a test-specific directory
        # This is important if any tests interact with file uploads directly
        # For this example, we'll mock file system interactions where possible
        app.config['UPLOAD_FOLDER'] = TEST_DATA_DIR

        # Create dummy data files for testing execute_duckdb_query
        self.csv_file_path = os.path.join(TEST_DATA_DIR, 'sample.csv')
        self.parquet_file_path = os.path.join(TEST_DATA_DIR, 'sample.parquet')
        self.sqlite_file_path = os.path.join(TEST_DATA_DIR, 'sample.sqlite')

        # Sample DataFrame
        self.sample_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.0, 20.5, 30.1]
        })
        self.sample_df.to_csv(self.csv_file_path, index=False)
        self.sample_df.to_parquet(self.parquet_file_path, index=False)
        
        # Create a dummy SQLite file with a table
        import sqlite3
        conn = sqlite3.connect(self.sqlite_file_path)
        self.sample_df.to_sql('sample_table', conn, index=False, if_exists='replace')
        conn.close()


    def tearDown(self):
        """Clean up resources after tests."""
        # Remove dummy files
        if os.path.exists(self.csv_file_path):
            os.remove(self.csv_file_path)
        if os.path.exists(self.parquet_file_path):
            os.remove(self.parquet_file_path)
        if os.path.exists(self.sqlite_file_path):
            os.remove(self.sqlite_file_path)
        # Potentially remove TEST_DATA_DIR if it's empty and created by tests,
        # but for this setup, it's fine to leave it.

    # --- Test execute_duckdb_query ---
    def test_execute_duckdb_csv_successful(self):
        df, error = execute_duckdb_query("SELECT * FROM sample_table", self.csv_file_path, "sample_table")
        self.assertIsNone(error)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertTrue('name' in df.columns)

    def test_execute_duckdb_parquet_successful(self):
        df, error = execute_duckdb_query("SELECT * FROM sample_table", self.parquet_file_path, "sample_table")
        self.assertIsNone(error)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)

    def test_execute_duckdb_sqlite_successful(self):
        # For SQLite, table name in SQL query must match the table in the .sqlite file
        df, error = execute_duckdb_query("SELECT * FROM sample_table", self.sqlite_file_path, "sample_table")
        self.assertIsNone(error)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)

    def test_execute_duckdb_invalid_sql(self):
        df, error = execute_duckdb_query("SELECT FROM table", self.csv_file_path, "sample_table")
        self.assertIsNotNone(error)
        self.assertIsNone(df)
        self.assertIn("DuckDB SQL execution error", error)
    
    def test_execute_duckdb_non_existent_file(self):
        df, error = execute_duckdb_query("SELECT * FROM sample_table", "non_existent.csv", "sample_table")
        self.assertIsNotNone(error)
        self.assertIsNone(df)
        self.assertIn("No files found that match the pattern", error) # Updated assertion

    def test_execute_duckdb_non_existent_table_in_sqlite(self):
        # Querying a table that does not exist within the SQLite file
        df, error = execute_duckdb_query("SELECT * FROM non_existent_table", self.sqlite_file_path, "non_existent_table")
        self.assertIsNotNone(error)
        self.assertIsNone(df)
        # DuckDB's error might vary, but it should indicate a catalog error or similar
        self.assertTrue("Catalog Error" in error or "no such table" in error.lower())


    # --- Test File Upload (/upload) ---
    @patch('app.backend.app.pd.read_csv')
    def test_upload_csv_successful(self, mock_read_csv):
        # Mock pd.read_csv to return a sample DataFrame for metadata inference
        mock_df = pd.DataFrame({
            'col1': [1, 2],
            'col2': ['a', 'b'],
            'col3': [1.0, 2.0] # Mixed types to test inference
        })
        mock_read_csv.return_value = mock_df

        # Mock file object
        mock_file_content = "col1,col2,col3\n1,a,1.0\n2,b,2.0"
        mock_file = MagicMock()
        mock_file.filename = 'test.csv'
        mock_file.stream = io.BytesIO(mock_file_content.encode('utf-8'))
        
        # Patch 'open' used by file.save() if it's called, though for test_client it might not be
        # For this test, we primarily care about the metadata part.
        # If file.save() is problematic, mock it within app.backend.app if it's directly called there.
        # Here, we assume file.save() is handled by Flask/Werkzeug and we focus on read_csv mock.
        
        with patch('builtins.open', mock_open()) as mock_open_file: # Mocks open for file.save
            with patch('os.path.join', return_value=os.path.join(TEST_DATA_DIR, 'test.csv')): # Mocks path join
                response = self.app.post('/upload',
                                         content_type='multipart/form-data',
                                         data={'file': (io.BytesIO(mock_file_content.encode('utf-8')), 'test.csv')})

        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data['message'], 'File uploaded successfully')
        self.assertIsNotNone(json_data['metadata'])
        self.assertEqual(json_data['metadata']['table_name'], 'test')
        self.assertEqual(len(json_data['metadata']['columns']), 3)
        self.assertEqual(json_data['metadata']['columns'][0]['name'], 'col1')
        self.assertEqual(json_data['metadata']['columns'][0]['type'], 'INTEGER')
        self.assertEqual(json_data['metadata']['columns'][1]['name'], 'col2')
        self.assertEqual(json_data['metadata']['columns'][1]['type'], 'TEXT')
        self.assertEqual(json_data['metadata']['columns'][2]['name'], 'col3')
        self.assertEqual(json_data['metadata']['columns'][2]['type'], 'REAL')


    def test_upload_unsupported_file_type(self):
        mock_file_content = "some content"
        response = self.app.post('/upload',
                                 content_type='multipart/form-data',
                                 data={'file': (io.BytesIO(mock_file_content.encode('utf-8')), 'test.txt')})
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertEqual(json_data['error'], 'File type not allowed')

    @patch('app.backend.app.pd.read_csv')
    def test_upload_malformed_csv(self, mock_read_csv):
        # Simulate an error during pd.read_csv (e.g., empty file, no header for inference)
        mock_read_csv.side_effect = Exception("Error parsing CSV")

        mock_file_content = "" # Empty content
        with patch('builtins.open', mock_open()):
             with patch('os.path.join', return_value=os.path.join(TEST_DATA_DIR, 'malformed.csv')):
                response = self.app.post('/upload',
                                     content_type='multipart/form-data',
                                     data={'file': (io.BytesIO(mock_file_content.encode('utf-8')), 'malformed.csv')})
        
        self.assertEqual(response.status_code, 500)
        json_data = response.get_json()
        self.assertIn('Error processing CSV file', json_data['error'])

    # --- Test LLM Prompt Construction (mocking OpenAI API) ---
    @patch('app.backend.app.openai.Completion.create')
    def test_prompt_construction_sql_generation(self, mock_openai_completion):
        # Set up necessary global variables in app.backend.app that handle_query uses
        backend_app.current_metadata = {
            'table_name': 'test_table',
            'columns': [{'name': 'id', 'type': 'INTEGER'}, {'name': 'data', 'type': 'TEXT'}]
        }
        backend_app.current_uploaded_filepath = self.csv_file_path # Needs a valid path for execute_duckdb_query
        backend_app.current_uploaded_filename = "sample.csv"
        backend_app.OPENAI_API_KEY = "fake_key" # Ensure API is "configured"

        # Mock the LLM response for SQL generation
        mock_openai_completion.return_value = MagicMock(choices=[MagicMock(text="SELECT * FROM test_table WHERE id = 1;")])
        
        # Call the /query endpoint
        self.app.post('/query', json={
            'naturalLanguageQuery': 'Find data for ID 1',
            'metadata': backend_app.current_metadata, # Send metadata as UI would
            'agent_type': 'sql'
        })

        # Check that openai.Completion.create was called
        mock_openai_completion.assert_called()
        
        # Get the actual prompt passed to the LLM
        # The first call to LLM is for SQL generation.
        args, kwargs = mock_openai_completion.call_args_list[0]
        actual_prompt = kwargs['prompt']
        
        self.assertIn("Table Name: test_table", actual_prompt)
        self.assertIn("- id (INTEGER)", actual_prompt)
        self.assertIn("- data (TEXT)", actual_prompt)
        self.assertIn("User Question: Find data for ID 1", actual_prompt)
        self.assertIn("SQL Query:", actual_prompt)
        
        # Reset global OPENAI_API_KEY if it was set for this test
        backend_app.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


    @patch('app.backend.app.openai.Completion.create')
    @patch('app.backend.app.execute_duckdb_query') # Mock duckdb execution to simulate failure
    def test_prompt_construction_reflection(self, mock_execute_duckdb, mock_openai_completion):
        backend_app.current_metadata = {
            'table_name': 'reflect_table',
            'columns': [{'name': 'col', 'type': 'TEXT'}]
        }
        backend_app.current_uploaded_filepath = self.csv_file_path
        backend_app.current_uploaded_filename = "sample.csv"
        backend_app.OPENAI_API_KEY = "fake_key"

        # Simulate initial SQL generation
        mock_openai_completion.side_effect = [
            MagicMock(choices=[MagicMock(text="SELECT fail_col FROM reflect_table;")]), # 1st call: initial SQL
            MagicMock(choices=[MagicMock(text="SELECT col FROM reflect_table;")])        # 2nd call: corrected SQL
        ]
        # Simulate DuckDB error on the first SQL query
        mock_execute_duckdb.side_effect = [
            (None, "DuckDB SQL execution error: no such column: fail_col"), # 1st execute fails
            (pd.DataFrame({'col': ['data']}), None)                         # 2nd execute succeeds
        ]

        self.app.post('/query', json={
            'naturalLanguageQuery': 'Get the column',
            'metadata': backend_app.current_metadata,
            'agent_type': 'sql'
        })

        self.assertEqual(mock_openai_completion.call_count, 3) # SQL gen + Reflection + Summary
        
        # Get the prompt for the reflection call (second call to LLM)
        args, kwargs = mock_openai_completion.call_args_list[1] # Index 1 for reflection
        reflection_prompt = kwargs['prompt']

        self.assertIn("The following SQL query resulted in an error. Please correct it.", reflection_prompt)
        self.assertIn("Original Question: Get the column", reflection_prompt)
        self.assertIn("Table Name: reflect_table", reflection_prompt)
        self.assertIn("- col (TEXT)", reflection_prompt)
        self.assertIn("Failed SQL: SELECT fail_col FROM reflect_table;", reflection_prompt)
        self.assertIn("Error Message: DuckDB SQL execution error: no such column: fail_col", reflection_prompt)
        self.assertIn("Corrected SQL Query:", reflection_prompt)

        # Optionally, check the summary prompt (third call)
        args_summary, kwargs_summary = mock_openai_completion.call_args_list[2] # Index 2 for summary
        summary_prompt = kwargs_summary['prompt']
        self.assertIn("Based on the user's question 'Get the column'", summary_prompt)
        self.assertIn("the SQL query 'SELECT col FROM reflect_table;'", summary_prompt)
        
        backend_app.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


    @patch('app.backend.app.openai.Completion.create')
    @patch('app.backend.app.execute_duckdb_query')
    def test_prompt_construction_summary(self, mock_execute_duckdb, mock_openai_completion):
        backend_app.current_metadata = {
            'table_name': 'summary_table',
            'columns': [{'name': 'data', 'type': 'TEXT'}]
        }
        backend_app.current_uploaded_filepath = self.csv_file_path
        backend_app.current_uploaded_filename = "sample.csv"
        backend_app.OPENAI_API_KEY = "fake_key"

        # For this test, we only want two LLM calls: SQL gen + Summary
        # The execute_duckdb_query should succeed on the first try.
        mock_llm_sql_gen = MagicMock(choices=[MagicMock(text="SELECT data FROM summary_table;")])
        mock_llm_summary = MagicMock(choices=[MagicMock(text="This is the summary.")])

        mock_openai_completion.side_effect = [
            mock_llm_sql_gen,
            mock_llm_summary
        ]
        sample_results_df = pd.DataFrame({'data': ['result1', 'result2']})
        # Ensure execute_duckdb_query is only called once and succeeds
        mock_execute_duckdb.return_value = (sample_results_df, None)

        self.app.post('/query', json={
            'naturalLanguageQuery': 'Summarize the data',
            'metadata': backend_app.current_metadata,
            'agent_type': 'sql'
        })

        self.assertEqual(mock_openai_completion.call_count, 2)
        mock_execute_duckdb.assert_called_once() # Ensure it was only called once
        
        # Get the prompt for the summary call (second call to LLM in this flow)
        args, kwargs = mock_openai_completion.call_args_list[1] # Summary is the second call
        summary_prompt = kwargs['prompt']

        self.assertIn("Based on the user's question 'Summarize the data'", summary_prompt)
        self.assertIn("the SQL query 'SELECT data FROM summary_table;'", summary_prompt)
        self.assertIn("The query returned the following results:", summary_prompt)
        self.assertIn("result1", summary_prompt) # Check for result data in prompt
        self.assertIn("result2", summary_prompt)
        self.assertIn("Natural Language Answer:", summary_prompt)

        backend_app.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


    # --- Test Plotting (/plot_data) ---
    def test_plot_data_successful(self):
        # Pre-populate the last_successful_df in the app's context
        backend_app.last_successful_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        
        response = self.app.post('/plot_data')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('plot_image', json_data)
        self.assertTrue(json_data['plot_image'].startswith('data:image/png;base64,'))

    def test_plot_data_no_data(self):
        backend_app.last_successful_df = None # Reset for this test
        response = self.app.post('/plot_data')
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertEqual(json_data['error'], 'No data available to plot. Please execute a query first.')
        # Restore if it was modified, though setUp should handle fresh state for each test method
        # For safety, can reset to a known state or ensure tests don't interfere.

    def test_plot_data_unsuitable_data(self):
        # E.g. DataFrame with only non-numeric data for some plot types
        backend_app.last_successful_df = pd.DataFrame({'text_col': ['a', 'b', 'c']})
        response = self.app.post('/plot_data')
        # This might return 400 or 500 depending on how robust the plotting error handling is.
        # Based on current app.py, it should be a 400 with "No numeric columns".
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn('Plotting logic could not determine a suitable plot for this data structure.', json_data['error'])

# Metadata for pandas tests
pandas_test_metadata = {
    'table_name': 'test_df', # Placeholder, as pandas operates on a df variable
    'columns': [
        {'name': 'col_a', 'type': 'INTEGER'},
        {'name': 'col_b', 'type': 'TEXT'},
        {'name': 'col_c', 'type': 'REAL'}
    ]
}

class TestPandasAgent(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        app.config['UPLOAD_FOLDER'] = TEST_DATA_DIR
        backend_app.OPENAI_API_KEY = "fake_key"

        # Create a specific CSV for pandas tests that matches pandas_test_metadata
        self.pandas_test_csv_path = os.path.join(TEST_DATA_DIR, 'pandas_sample.csv')
        pd.DataFrame({
            'col_a': [1, 2, 3, 4, 5],
            'col_b': ['alpha', 'beta', 'gamma', 'delta', 'epsilon'],
            'col_c': [10.1, 20.2, 5.5, 15.9, 12.0]
        }).to_csv(self.pandas_test_csv_path, index=False)

        # Simulate file upload for pandas tests
        backend_app.current_uploaded_filepath = self.pandas_test_csv_path
        backend_app.current_metadata = pandas_test_metadata # This metadata matches the CSV above
        backend_app.current_uploaded_filename = "pandas_sample.csv"


    def tearDown(self):
        # Reset any global state changed during tests
        backend_app.current_uploaded_filepath = None
        backend_app.current_metadata = None
        backend_app.current_uploaded_filename = None
        backend_app.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if os.path.exists(self.pandas_test_csv_path):
            os.remove(self.pandas_test_csv_path)

    @patch('app.backend.app.openai.Completion.create')
    def test_pandas_code_generation_selection(self, mock_openai_completion):
        natural_language_query = "select all data from col_b"
        expected_pandas_code = "result_df = df[['col_b']]"
        expected_summary = "Mocked summary for selection."

        # Mock LLM calls: first for code gen, second for summary
        mock_openai_completion.side_effect = [
            MagicMock(choices=[MagicMock(text=expected_pandas_code)]), # Code gen
            MagicMock(choices=[MagicMock(text=expected_summary)])      # Summary
        ]

        response = self.app.post('/query', json={
            'naturalLanguageQuery': natural_language_query,
            'metadata': pandas_test_metadata,
            'agent_type': 'pandas'
        })

        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()

        # Assert that LLM was called twice (once for code, once for summary)
        self.assertEqual(mock_openai_completion.call_count, 2)

        # Assert code generation prompt (first call)
        args_code, kwargs_code = mock_openai_completion.call_args_list[0]
        actual_code_gen_prompt = kwargs_code['prompt']
        self.assertIn("You are a helpful assistant that generates Python pandas code.", actual_code_gen_prompt)
        self.assertIn("Given a pandas DataFrame named 'df' with the following columns:", actual_code_gen_prompt)
        self.assertIn(f"User Question: \"{natural_language_query}\"", actual_code_gen_prompt)
        self.assertIn("Pandas Code:", actual_code_gen_prompt)

        # Assert summary prompt (second call)
        args_summary, kwargs_summary = mock_openai_completion.call_args_list[1]
        actual_summary_prompt = kwargs_summary['prompt']
        self.assertIn(f"Based on the user's question '{natural_language_query}',", actual_summary_prompt) # Adjusted assertion
        self.assertIn(expected_pandas_code, actual_summary_prompt) # Check if generated code is in summary prompt
        self.assertIn("The pandas code returned the following DataFrame:", actual_summary_prompt) # For small DFs
        self.assertIn("alpha", actual_summary_prompt) # Check for some data in prompt

        # Assert endpoint response structure
        self.assertEqual(json_data.get('code_type'), 'pandas')
        self.assertEqual(json_data.get('generated_code'), expected_pandas_code)
        self.assertIsNotNone(json_data.get('natural_language_response'))
        self.assertIsNone(json_data.get('error'))

        expected_results = [{'col_b': 'alpha'}, {'col_b': 'beta'}, {'col_b': 'gamma'}, {'col_b': 'delta'}, {'col_b': 'epsilon'}]
        self.assertEqual(json_data.get('results'), expected_results)
        # Assert that the NL response is the mocked summary
        self.assertEqual(json_data.get('natural_language_response'), expected_summary)


    @patch('app.backend.app.openai.Completion.create')
    def test_pandas_code_generation_filter(self, mock_openai_completion):
        natural_language_query = "show col_a and col_b where col_c is greater than 10.5"
        expected_pandas_code = "result_df = df[df['col_c'] > 10.5][['col_a', 'col_b']]"
        expected_summary = "Filtered data based on col_c."

        # Mock LLM calls: first for code gen, second for summary
        mock_openai_completion.side_effect = [
            MagicMock(choices=[MagicMock(text=expected_pandas_code)]), # Code gen
            MagicMock(choices=[MagicMock(text=expected_summary)])      # Summary
        ]

        response = self.app.post('/query', json={
            'naturalLanguageQuery': natural_language_query,
            'metadata': pandas_test_metadata,
            'agent_type': 'pandas'
        })

        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()

        self.assertEqual(mock_openai_completion.call_count, 2)

        # Assert code generation prompt (first call)
        args_code, kwargs_code = mock_openai_completion.call_args_list[0]
        self.assertIn(f"User Question: \"{natural_language_query}\"", kwargs_code['prompt'])

        # Assert summary prompt (second call)
        args_summary, kwargs_summary = mock_openai_completion.call_args_list[1]
        self.assertIn(f"Based on the user's question '{natural_language_query}',", kwargs_summary['prompt']) # Adjusted assertion
        self.assertIn(expected_pandas_code, kwargs_summary['prompt'])
        self.assertIn("The pandas code returned the following DataFrame:", kwargs_summary['prompt']) # For small DFs


        self.assertEqual(json_data.get('code_type'), 'pandas')
        self.assertEqual(json_data.get('generated_code'), expected_pandas_code)
        self.assertIsNone(json_data.get('error'))

        actual_results = json_data.get('results')
        self.assertEqual(len(actual_results), 3)
        self.assertIn({'col_a': 2, 'col_b': 'beta'}, actual_results)
        self.assertIn({'col_a': 4, 'col_b': 'delta'}, actual_results)
        self.assertIn({'col_a': 5, 'col_b': 'epsilon'}, actual_results)
        self.assertEqual(json_data.get('natural_language_response'), expected_summary)


    @patch('app.backend.app.openai.Completion.create')
    def test_pandas_code_generation_groupby(self, mock_openai_completion):
        natural_language_query = "group by col_b and sum col_a"
        expected_pandas_code = "result_df = df.groupby('col_b')['col_a'].sum().reset_index()"
        expected_summary = "Grouped data and summed col_a."

        mock_openai_completion.side_effect = [
            MagicMock(choices=[MagicMock(text=expected_pandas_code)]), # Code gen
            MagicMock(choices=[MagicMock(text=expected_summary)])      # Summary
        ]

        response = self.app.post('/query', json={
            'naturalLanguageQuery': natural_language_query,
            'metadata': pandas_test_metadata,
            'agent_type': 'pandas'
        })
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()

        self.assertEqual(mock_openai_completion.call_count, 2)
        # Prompts can be checked similarly if needed for code gen (call_args_list[0])
        # and summary (call_args_list[1])

        self.assertEqual(json_data.get('code_type'), 'pandas')
        self.assertEqual(json_data.get('generated_code'), expected_pandas_code)
        self.assertIsNone(json_data.get('error'))

        expected_results = [
            {'col_b': 'alpha', 'col_a': 1}, {'col_b': 'beta', 'col_a': 2},
            {'col_b': 'gamma', 'col_a': 3}, {'col_b': 'delta', 'col_a': 4},
            {'col_b': 'epsilon', 'col_a': 5}
        ]
        actual_results = json_data.get('results')
        self.assertEqual(len(actual_results), 5)
        for res in expected_results:
            self.assertIn(res, actual_results)
        self.assertEqual(json_data.get('natural_language_response'), expected_summary)

    @patch('app.backend.app.openai.Completion.create')
    def test_pandas_summary_generation_fails(self, mock_openai_completion):
        natural_language_query = "select col_a"
        expected_pandas_code = "result_df = df[['col_a']]"

        # Mock LLM: success for code gen, failure for summary
        mock_openai_completion.side_effect = [
            MagicMock(choices=[MagicMock(text=expected_pandas_code)]), # Code gen
            openai.OpenAIError("Simulated API error for summary")      # Summary fails
        ]

        response = self.app.post('/query', json={
            'naturalLanguageQuery': natural_language_query,
            'metadata': pandas_test_metadata,
            'agent_type': 'pandas'
        })
        self.assertEqual(response.status_code, 200) # Endpoint itself should still be 200
        json_data = response.get_json()

        self.assertEqual(mock_openai_completion.call_count, 2)
        self.assertEqual(json_data.get('generated_code'), expected_pandas_code)
        self.assertIsNone(json_data.get('error')) # No error from code execution
        self.assertIsNotNone(json_data.get('results'))
        self.assertIn("error generating summary", json_data.get('natural_language_response').lower())


    # Tests for the execute_pandas_code function (to be created in app.py)
    def test_execute_pandas_code_successful_selection(self):
        from app.backend.app import execute_pandas_code # Import locally to use after it's defined
        sample_df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
        code_string = "result_df = df[['A']]"
        result_df, error = execute_pandas_code(code_string, sample_df.copy())
        self.assertIsNone(error)
        self.assertIsNotNone(result_df)
        self.assertTrue(isinstance(result_df, pd.DataFrame))
        self.assertEqual(list(result_df.columns), ['A'])
        self.assertEqual(len(result_df), 2)

    def test_execute_pandas_code_successful_filter(self):
        from app.backend.app import execute_pandas_code
        sample_df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 5]})
        code_string = "result_df = df[df['B'] > 10]"
        result_df, error = execute_pandas_code(code_string, sample_df.copy())
        self.assertIsNone(error)
        self.assertIsNotNone(result_df)
        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]['A'], 2)

    def test_execute_pandas_code_successful_groupby(self):
        from app.backend.app import execute_pandas_code
        sample_df = pd.DataFrame({'Category': ['X', 'Y', 'X'], 'Value': [10, 20, 30]})
        code_string = "result_df = df.groupby('Category')['Value'].sum().reset_index()"
        result_df, error = execute_pandas_code(code_string, sample_df.copy())
        self.assertIsNone(error)
        self.assertIsNotNone(result_df)
        self.assertEqual(len(result_df), 2)
        # Check if X sums to 40
        self.assertEqual(result_df[result_df['Category'] == 'X']['Value'].iloc[0], 40)


    def test_execute_pandas_code_syntax_error(self):
        from app.backend.app import execute_pandas_code
        sample_df = pd.DataFrame({'A': [1, 2]})
        code_string = "result_df = df['A] # Missing quote"
        result_df, error = execute_pandas_code(code_string, sample_df.copy())
        self.assertIsNotNone(error)
        self.assertIsNone(result_df)
        self.assertIn("SyntaxError", error)

    def test_execute_pandas_code_runtime_error(self):
        from app.backend.app import execute_pandas_code
        sample_df = pd.DataFrame({'A': [1, 2]})
        code_string = "result_df = df['NonExistentCol']" # NameError or KeyError depending on pandas version/usage
        result_df, error = execute_pandas_code(code_string, sample_df.copy())
        self.assertIsNotNone(error)
        self.assertIsNone(result_df)
        self.assertTrue("KeyError" in error or "NameError" in error or "column not found" in error.lower())


    def test_execute_pandas_code_no_result_df(self):
        from app.backend.app import execute_pandas_code
        sample_df = pd.DataFrame({'A': [1, 2]})
        code_string = "temp_df = df.copy()" # Does not assign to result_df
        result_df, error = execute_pandas_code(code_string, sample_df.copy())
        self.assertIsNotNone(error)
        self.assertIsNone(result_df)
        self.assertIn("Pandas code did not assign its result to 'result_df'", error)

    # Tests for data loading in handle_pandas_agent_query
    @patch('app.backend.app.execute_pandas_code') # Mock the execution part
    @patch('app.backend.app.openai.Completion.create') # Mock LLM call
    def test_pandas_agent_loads_csv(self, mock_openai_completion, mock_execute_pandas):
        # Setup: Create a dummy CSV file for this test
        test_csv_path = os.path.join(TEST_DATA_DIR, 'test_load.csv')
        pd.DataFrame({'x': [1,2], 'y': ['a','b']}).to_csv(test_csv_path, index=False)

        backend_app.current_uploaded_filepath = test_csv_path
        backend_app.current_uploaded_filename = "test_load.csv"
        backend_app.current_metadata = {'table_name': 'test_load', 'columns': [{'name': 'x', 'type': 'INTEGER'}, {'name': 'y', 'type': 'TEXT'}]}

        # Mock LLM to return simple valid code
        mock_openai_completion.return_value = MagicMock(choices=[MagicMock(text="result_df = df.copy()")])
        # Mock execute_pandas_code to just return a dummy result
        mock_execute_pandas.return_value = (pd.DataFrame({'dummy': [1]}), None)

        self.app.post('/query', json={
            'naturalLanguageQuery': 'load the csv',
            'agent_type': 'pandas',
            'metadata': backend_app.current_metadata
        })

        mock_execute_pandas.assert_called_once()
        args, _ = mock_execute_pandas.call_args
        called_with_code, called_with_df = args[0], args[1]

        self.assertEqual(called_with_code, "result_df = df.copy()")
        self.assertTrue(isinstance(called_with_df, pd.DataFrame))
        self.assertEqual(list(called_with_df.columns), ['x', 'y'])
        self.assertEqual(len(called_with_df), 2)

        if os.path.exists(test_csv_path): os.remove(test_csv_path)

    @patch('app.backend.app.execute_pandas_code')
    @patch('app.backend.app.openai.Completion.create')
    def test_pandas_agent_loads_parquet(self, mock_openai_completion, mock_execute_pandas):
        test_parquet_path = os.path.join(TEST_DATA_DIR, 'test_load.parquet')
        pd.DataFrame({'p': [10,20], 'q': ['c','d']}).to_parquet(test_parquet_path, index=False)

        backend_app.current_uploaded_filepath = test_parquet_path
        backend_app.current_uploaded_filename = "test_load.parquet"
        backend_app.current_metadata = {'table_name': 'test_load', 'columns': [{'name': 'p', 'type': 'INTEGER'}, {'name': 'q', 'type': 'TEXT'}]}

        mock_openai_completion.return_value = MagicMock(choices=[MagicMock(text="result_df = df.head(1)")])
        mock_execute_pandas.return_value = (pd.DataFrame({'dummy': [1]}), None)

        self.app.post('/query', json={
            'naturalLanguageQuery': 'load the parquet',
            'agent_type': 'pandas',
            'metadata': backend_app.current_metadata
        })

        mock_execute_pandas.assert_called_once()
        args, _ = mock_execute_pandas.call_args
        called_with_df = args[1]
        self.assertTrue(isinstance(called_with_df, pd.DataFrame))
        self.assertEqual(list(called_with_df.columns), ['p', 'q'])
        self.assertEqual(len(called_with_df), 2)

        if os.path.exists(test_parquet_path): os.remove(test_parquet_path)

    @patch('app.backend.app.execute_pandas_code')
    @patch('app.backend.app.openai.Completion.create')
    def test_pandas_agent_loads_sqlite(self, mock_openai_completion, mock_execute_pandas):
        test_sqlite_path = os.path.join(TEST_DATA_DIR, 'test_load.sqlite')
        conn = sqlite3.connect(test_sqlite_path)
        pd.DataFrame({'s_col1': [100,200], 's_col2': ['sqlite_a','sqlite_b']}).to_sql('my_sqlite_table', conn, index=False, if_exists='replace')
        conn.close()

        backend_app.current_uploaded_filepath = test_sqlite_path
        backend_app.current_uploaded_filename = "test_load.sqlite"
        # For SQLite, table_name in metadata is crucial
        backend_app.current_metadata = {'table_name': 'my_sqlite_table', 'columns': [{'name': 's_col1', 'type': 'INTEGER'}, {'name': 's_col2', 'type': 'TEXT'}]}

        mock_openai_completion.return_value = MagicMock(choices=[MagicMock(text="result_df = df.describe()")])
        mock_execute_pandas.return_value = (pd.DataFrame({'dummy': [1]}), None)

        self.app.post('/query', json={
            'naturalLanguageQuery': 'load sqlite data',
            'agent_type': 'pandas',
            'metadata': backend_app.current_metadata
        })

        mock_execute_pandas.assert_called_once()
        args, _ = mock_execute_pandas.call_args
        called_with_df = args[1]
        self.assertTrue(isinstance(called_with_df, pd.DataFrame))
        self.assertEqual(list(called_with_df.columns), ['s_col1', 's_col2'])
        self.assertEqual(len(called_with_df), 2)
        self.assertEqual(called_with_df['s_col1'].iloc[0], 100)

        if os.path.exists(test_sqlite_path): os.remove(test_sqlite_path)


if __name__ == '__main__':
    unittest.main()
