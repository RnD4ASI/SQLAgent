import sqlite3
import os

# Define the database file path
db_file = os.path.join("sample_data", "sample_data.db")

# Connect to SQLite database (this will create the file if it doesn't exist)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create sample_table
cursor.execute("""
CREATE TABLE IF NOT EXISTS sample_table (
    ID INTEGER PRIMARY KEY,
    Product TEXT,
    Price REAL
)
""")

# Sample data to insert
sample_data = [
    (1, 'Apple', 0.50),
    (2, 'Banana', 0.30),
    (3, 'Orange', 0.40),
    (4, 'Milk', 2.50),
    (5, 'Bread', 1.80),
    (6, 'Eggs', 3.20),
    (7, 'Chicken', 5.00),
    (8, 'Rice', 1.20),
    (9, 'Tomatoes', 0.60),
    (10, 'Potatoes', 0.75)
]

# Insert data into sample_table
cursor.executemany("INSERT INTO sample_table (ID, Product, Price) VALUES (?, ?, ?)", sample_data)

# Commit the changes
conn.commit()

# Close the connection
conn.close()

print(f"Database '{db_file}' created and populated successfully.")
