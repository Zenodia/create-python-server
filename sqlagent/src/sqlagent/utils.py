import sqlite3

def get_db_schema(db_path):
    """
    Retrieves the schema of a SQLite database.

    Args:
        db_path (str): The path to the SQLite database.

    Returns:
        dict: A dictionary where the keys are the table names and the values are lists of tuples.
              Each tuple contains the column name, data type, whether it's nullable, and the default value.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get the list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            # Get the schema of the current table
            cursor.execute(f"PRAGMA table_info({table})")
            schema[table] = cursor.fetchall()

        conn.close()
        return schema

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None

def sample_rows(db_path, num_samples):
    """
    Connects to a SQLite database, retrieves the first 'num_samples' rows of each table,
    and returns a list of tuples containing the table name and sample rows as a dictionary.

    Args:
        db_path (str): Path to the SQLite database file.
        num_samples (int): Number of rows to sample from each table.

    Returns:
        list[tuple]: A list of tuples, where each tuple contains the table name and sample rows as a dictionary.
    """

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the list of tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    # Initialize an empty list to store the results
    results = []

    # Iterate over each table
    for table in tables:
        # Fetch the first 'num_samples' rows of the table
        cursor.execute(f"SELECT * FROM {table} LIMIT {num_samples};")
        rows = cursor.fetchall()

        # Get the column names
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]

        # Initialize an empty dictionary to store the sample rows
        sample_rows_dict = {}

        # Iterate over each column
        for i, column in enumerate(columns):
            # Get the values for this column
            values = [row[i] for row in rows]

            # Add the column and values to the dictionary
            sample_rows_dict[column] = values

        # Append the table name and sample rows dictionary to the results list
        results.append((table, sample_rows_dict))

    # Close the database connection
    conn.close()

    return results

import sqlite3

def execute_sql_query(db_path, sql_command):
    """
    Execute an SQL command on a SQLite database.

    Args:
    db_path (str): The path to the SQLite database.
    sql_command (str): The SQL command to execute.

    Returns:
    dict: A dictionary where the keys are the column names and the values are lists of column values.

    Raises:
    sqlite3.Error: If an error occurs while executing the SQL command.
    """

    try:
        # Establish a connection to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute the SQL command
        cursor.execute(sql_command)

        # Fetch all the rows from the last executed statement
        rows = cursor.fetchall()

        # Get the column names from the cursor description
        column_names = [description[0] for description in cursor.description]

        # Create a dictionary where the keys are the column names and the values are lists of column values
        result_dict = dict(zip(column_names, zip(*rows)))

        # Convert the tuples to lists
        result_dict = {key: list(value) for key, value in result_dict.items()}

        # Commit the transaction
        conn.commit()

        # Close the connection
        conn.close()

        return result_dict

    except sqlite3.Error as e:
        msg = f"An error occurred: {e}"
        print(msg)
        return msg

def extract_sql(sql):
    sql = sql.split('[SQL]')[1].split('[/SQL]')[0].strip()
    sql = sql.replace("```sql",'').replace("```",'')
    return sql


import os

def load_dotenv(filepath: str) -> None:
    """
    Load environment variables from a file.

    :param filepath: Path to the file containing environment variables
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip()
    except FileNotFoundError:
        logging.error(f"File {filepath} not found.")
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")

def invalid(res):
    if len(res) == 0:
        return True
    for k,v in res.items():
        if len(v)>100:
            return True
    return False

if __name__ == '__main__':
    db_path = "/datasets/text2sql_datasets/geforce/database/geforce/geforce.sqlite"
    sql_command = "SELECT name FROM sqlite_master WHERE type='table';"
    results = execute_sql_query(db_path, sql_command)

    if results is not None:
        for row in results:
            print(row)
    