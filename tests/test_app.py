import unittest
import os
import sys
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory (project root) to the Python path
# to allow imports from the 'app' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.backend.app import app, execute_duckdb_query # Import Flask app and helper

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
        self.assertIn("Data file not found", error)

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
        app.backend.app.current_metadata = {
            'table_name': 'test_table',
            'columns': [{'name': 'id', 'type': 'INTEGER'}, {'name': 'data', 'type': 'TEXT'}]
        }
        app.backend.app.current_uploaded_filepath = self.csv_file_path # Needs a valid path for execute_duckdb_query
        app.backend.app.current_uploaded_filename = "sample.csv"
        app.backend.app.OPENAI_API_KEY = "fake_key" # Ensure API is "configured"

        # Mock the LLM response for SQL generation
        mock_openai_completion.return_value = MagicMock(choices=[MagicMock(text="SELECT * FROM test_table WHERE id = 1;")])
        
        # Call the /query endpoint
        self.app.post('/query', json={
            'naturalLanguageQuery': 'Find data for ID 1',
            'metadata': app.backend.app.current_metadata # Send metadata as UI would
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
        app.backend.app.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


    @patch('app.backend.app.openai.Completion.create')
    @patch('app.backend.app.execute_duckdb_query') # Mock duckdb execution to simulate failure
    def test_prompt_construction_reflection(self, mock_execute_duckdb, mock_openai_completion):
        app.backend.app.current_metadata = {
            'table_name': 'reflect_table',
            'columns': [{'name': 'col', 'type': 'TEXT'}]
        }
        app.backend.app.current_uploaded_filepath = self.csv_file_path 
        app.backend.app.current_uploaded_filename = "sample.csv"
        app.backend.app.OPENAI_API_KEY = "fake_key"

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
            'metadata': app.backend.app.current_metadata
        })

        self.assertEqual(mock_openai_completion.call_count, 2) # SQL gen + Reflection
        
        # Get the prompt for the reflection call (second call to LLM)
        args, kwargs = mock_openai_completion.call_args_list[1]
        reflection_prompt = kwargs['prompt']

        self.assertIn("The following SQL query resulted in an error. Please correct it.", reflection_prompt)
        self.assertIn("Original Question: Get the column", reflection_prompt)
        self.assertIn("Table Name: reflect_table", reflection_prompt)
        self.assertIn("- col (TEXT)", reflection_prompt)
        self.assertIn("Failed SQL: SELECT fail_col FROM reflect_table;", reflection_prompt)
        self.assertIn("Error Message: DuckDB SQL execution error: no such column: fail_col", reflection_prompt)
        self.assertIn("Corrected SQL Query:", reflection_prompt)
        
        app.backend.app.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


    @patch('app.backend.app.openai.Completion.create')
    @patch('app.backend.app.execute_duckdb_query')
    def test_prompt_construction_summary(self, mock_execute_duckdb, mock_openai_completion):
        app.backend.app.current_metadata = {
            'table_name': 'summary_table',
            'columns': [{'name': 'data', 'type': 'TEXT'}]
        }
        app.backend.app.current_uploaded_filepath = self.csv_file_path
        app.backend.app.current_uploaded_filename = "sample.csv"
        app.backend.app.OPENAI_API_KEY = "fake_key"

        # Simulate SQL generation and successful execution
        mock_openai_completion.side_effect = [
            MagicMock(choices=[MagicMock(text="SELECT data FROM summary_table;")]), # SQL gen
            MagicMock(choices=[MagicMock(text="This is the summary.")])             # NL Summary
        ]
        sample_results_df = pd.DataFrame({'data': ['result1', 'result2']})
        mock_execute_duckdb.return_value = (sample_results_df, None) # Simulate successful execution

        self.app.post('/query', json={
            'naturalLanguageQuery': 'Summarize the data',
            'metadata': app.backend.app.current_metadata
        })

        self.assertEqual(mock_openai_completion.call_count, 2) # SQL gen + Summary
        
        # Get the prompt for the summary call (second call to LLM in this flow)
        args, kwargs = mock_openai_completion.call_args_list[1]
        summary_prompt = kwargs['prompt']

        self.assertIn("Based on the user's question 'Summarize the data'", summary_prompt)
        self.assertIn("the SQL query 'SELECT data FROM summary_table;'", summary_prompt)
        self.assertIn("The query returned the following results:", summary_prompt)
        self.assertIn("result1", summary_prompt) # Check for result data in prompt
        self.assertIn("result2", summary_prompt)
        self.assertIn("Natural Language Answer:", summary_prompt)

        app.backend.app.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


    # --- Test Plotting (/plot_data) ---
    def test_plot_data_successful(self):
        # Pre-populate the last_successful_df in the app's context
        app.backend.app.last_successful_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        
        response = self.app.post('/plot_data')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('plot_image', json_data)
        self.assertTrue(json_data['plot_image'].startswith('data:image/png;base64,'))

    def test_plot_data_no_data(self):
        app.backend.app.last_successful_df = None
        response = self.app.post('/plot_data')
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertEqual(json_data['error'], 'No data available to plot. Please execute a query first.')

    def test_plot_data_unsuitable_data(self):
        # E.g. DataFrame with only non-numeric data for some plot types, or empty after filtering
        app.backend.app.last_successful_df = pd.DataFrame({'text_col': ['a', 'b', 'c']})
        response = self.app.post('/plot_data')
        self.assertEqual(response.status_code, 400) # Should fail as no numeric data
        json_data = response.get_json()
        self.assertIn('No numeric columns found for plotting', json_data['error'])


if __name__ == '__main__':
    unittest.main()
