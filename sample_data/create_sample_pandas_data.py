import subprocess
import sys
import os

def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        print(f"{package_name} is already installed.")
        return True
    except ImportError:
        print(f"{package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"{package_name} installed successfully.")
            # After installation, the script needs to be re-run for the import to work reliably
            # Or, we can try to add the installation path to sys.path, but re-running is cleaner.
            return False # Indicate that installation happened and re-run might be needed
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package_name}: {e}")
            sys.exit(1)

# Flag to check if we are in the re-executed context
# We set an environment variable to signal that packages should now be installed.
PACKAGES_INSTALLED_FLAG = "PACKAGES_INSTALLED"

if not os.environ.get(PACKAGES_INSTALLED_FLAG):
    pandas_installed = install_and_import("pandas")
    pyarrow_installed = install_and_import("pyarrow")

    if not pandas_installed or not pyarrow_installed:
        print("Packages were installed. Re-running the script...")
        # Set the flag and re-execute the script
        env = os.environ.copy()
        env[PACKAGES_INSTALLED_FLAG] = "1"
        process = subprocess.Popen([sys.executable] + sys.argv, env=env)
        process.wait() # Wait for the re-executed script to finish
        sys.exit(process.returncode) # Exit with the same code as the child process

# If we reach here, it means either packages were already installed,
# or this is the re-execution after installation.
import pandas as pd
import pyarrow # pyarrow is used by pandas for parquet, so just importing is enough

# Define the output directory and file names
output_dir = "sample_data"
csv_file = os.path.join(output_dir, "sample_data.csv")
parquet_file = os.path.join(output_dir, "sample_data.parquet")

# Create a sample DataFrame
data = {
    'Date': pd.to_datetime(['2023-01-01'] * 5 + ['2023-01-02'] * 5 + ['2023-01-03'] * 5),
    'Category': ['A', 'B', 'A', 'C', 'B'] * 3,
    'Quantity': [10, 15, 12, 18, 20, 11, 16, 13, 19, 21, 10, 14, 12, 17, 22],
    'Revenue': [100.0, 150.0, 120.0, 180.0, 200.0, 110.0, 160.0, 130.0, 190.0, 210.0, 100.0, 140.0, 120.0, 170.0, 220.0]
}
sample_df = pd.DataFrame(data)

# Ensure the output directory exists (it should, from previous steps)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the DataFrame to a CSV file
sample_df.to_csv(csv_file, index=False)
print(f"DataFrame saved to '{csv_file}'")

# Save the DataFrame to a Parquet file
sample_df.to_parquet(parquet_file, index=False)
print(f"DataFrame saved to '{parquet_file}'")

print("Sample pandas DataFrame created and saved to CSV and Parquet formats successfully.")
