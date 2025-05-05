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


if __name__ == '__main__':
    s = system_prompt.format(schema='hello', samples='haha')
    print(s)
