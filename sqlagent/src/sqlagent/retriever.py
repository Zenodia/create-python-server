from langchain_core.language_models import BaseChatModel
from langchain_core.language_models import BaseLLM
from .llm import Text2SQL
from .utils import execute_sql_query
from .utils import extract_sql
from .utils import get_db_schema
from .utils import invalid
from .utils import sample_rows


class Retriever:

    def __init__(self, llm: BaseChatModel, db_path, n_samples=5, history: list[str] | None = None):
        self.db_path = db_path
        self.history = history
        self.use_history = history is not None
        self.schema = get_db_schema(db_path)
        self.samples = sample_rows(db_path, n_samples)
        self.text2sql = Text2SQL(llm=llm, schema=self.schema, samples=self.samples)

    async def retrieve(self, query, reference_queries: list[tuple[str, str]] | None = None):
        history = self.history if self.use_history else []

        output = await self.text2sql.ainvoke(query=query, reference_queries=reference_queries, history=history)

        sql = extract_sql(output)
        res = execute_sql_query(self.db_path, sql)

        if isinstance(res, str) or invalid(res):
            print('Debug result ...')
            output = await self.text2sql.debug(query, sql, res)
            sql = extract_sql(output)
            res = execute_sql_query(self.db_path, sql)
        if self.use_history:
            self.history.append({"role": "user", "content": f'[Question] {query} [/Question]'})
            self.history.append({"role": "assistant", "content": output})
            self.history.append({"role": "tool", "content": str(res)})

        print(f'\n\nSQL Execution Results:\n{str(res)}\n\n')
        return res, sql

    def reset_history(self):
        self.history = []
