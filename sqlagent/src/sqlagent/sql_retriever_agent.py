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

def invalid(res):
    if len(res) == 0:
        return True
    for k,v in res.items():
        if len(v)>100:
            return True
    return False

system_prompt = """You are a SQL expert who converts a user question to a sqlite SQL query.
The schema of the database is listed below:

## Schema
{schema}

The sample values of each table and column is provided below:

## Sample values
{samples}

## For reference, below are similar queries which have been successful in the past:
{reference_queries}

## Instructions
- always use the 'AS' keyword to assign a name or names to the results. the name should be from the user's query.

## The input format is below:
[Question] user's question in natural language form [/Question]

## The output format is blow:
[Thought] your thoughts on how to approach this problem step by step [/Thought]
[Draft SQL] output draft sql [/Draft SQL]
[Reflect] your reflections on if the generated Draft SQL has any obvious error, especially wrong table or column name error [/Reflect]
[SQL] output sql [/SQL]

## Example
Input:
[Question] show me all the table names [/Question]

Output:
[Thought] To show all the table names, I can use the SQLITE_MASTER table, which contains a catalog of all tables in the current database. The table name is stored in the 'name' column of the SQLITE_MASTER table. I will use a SELECT statement to retrieve the 'name' column from the SQLITE_MASTER table, but only for rows where the 'type' column is 'table'. [/Thought]
[Draft SQL] SELECT name FROM SQLITE_MASTER WHERE type='table' [/Draft SQL]
[Reflect] This SQL query is standard and should work as expected. However, the actual table names may vary depending on the specific database schema. [/Reflect]
[SQL] SELECT name FROM SQLITE_MASTER WHERE type='table' [/SQL]
"""

debug_prompt = """You are a SQL expert who examine a user question in natural language, a candidate sqlite SQL query and its execution result.
Your task is to determine if the candidate sql is correct.
The schema of the database is listed below:

## Schema
{schema}

The sample values of each table and column is provided below:

## Sample values
{samples}

The input format is below. All the fields are enclosed in square brackets:
[Question] user's question in natural language form [/Question]
[Candidate SQL] previously generated sql [/Candidate SQL]
[Result] the sql execution result [/Result]

The output format is blow:
[Thought] your thoughts on if the candidate sql is correct [/Thought]
[SQL] fixed sql if the candidate sql is not correct, otherwise repeat the candidate sql [/SQL]
"""

template = """[Question] {nlq} [/Question]
[Candidate SQL] {sql} [/Candidate SQL]
[Result] {result} [/Result]"""

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models import BaseLLM


class Text2SQL:

    def __init__(self, llm: BaseChatModel, schema, samples):
        # self.sys_prompt = system_prompt.format(schema=schema, samples=samples)
        # self.debug_prompt = debug_prompt.format(schema=schema, samples=samples)
        self.llm = llm
        self._schema = schema
        self._samples = samples

        self.sys_prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt),
                # MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("[Question] {query} [/Question]"),
            ], )

        self.debug_prompt_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(debug_prompt),
            HumanMessagePromptTemplate.from_template(template),
        ], )

        self.invoke_chain = self.sys_prompt_template | self.llm
        self.debug_chain = self.debug_prompt_template | self.llm

    async def ainvoke(self, query: str, reference_queries: list[tuple[str, str]] | None = None, history=None):

        completion = self.invoke_chain.astream(
            input={
                "query":
                    query,
                "chat_history":
                    history or [],
                "reference_queries":
                    "\n".join([
                        f"[Candidate SQL] {q_in} [/Candidate SQL]\n[Result] {q_out} [/Result]" for q_in,
                        q_out in reference_queries or []
                    ]),
                "schema":
                    self._schema,
                "samples":
                    self._samples
            })

        res = ''
        async for chunk in completion:
            if chunk.content is not None:
                result = str(chunk.content)
                print(result, end='')
                res += result
        return res

        # msgs = [{"role": "system", "content": self.sys_prompt}] + history
        # msgs.append({"role": "user", "content": f'[Question] {query} [/Question]'})
        # return await self.ainfer(msgs)

    # async def ainfer(self, msgs):
    #     completion = self.llm.astream(input=msgs, messages=msgs)

    #     res = ''
    #     async for chunk in completion:
    #         if chunk is not None:
    #             result = str(chunk)
    #             print(result, end='')
    #             res += result
    #     return res

    async def debug(self, nlq, sql, res):

        completion = self.debug_chain.astream(input={
            "query": nlq, "schema": self._schema, "samples": self._samples, "sql": sql, "result": res
        })

        res = ''
        async for chunk in completion:
            if chunk.content is not None:
                result = str(chunk.content)
                print(result, end='')
                res += result
        return res

        msgs = [{"role": "system", "content": self.sys_prompt}]
        query = template.format(nlq=nlq, sql=sql, result=res)
        msgs.append({"role": "user", "content": query})
        return await self.ainfer(msgs)
