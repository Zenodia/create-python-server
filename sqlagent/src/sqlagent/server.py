import os
import sys
import sqlite3
import logging
from contextlib import closing
from pathlib import Path
from pydantic import AnyUrl
from typing import Any

from mcp.server import InitializationOptions
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from openai import OpenAI
import os 
from dotenv import load_dotenv
from colorama import Fore
# api_key='any_key'
# model_uri = 'http://10.117.21.52:8080'
# model = "llama-3.1-70b-instruct-int4"
from openai import OpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from .sql_retriever_agent import get_db_schema,sample_rows,execute_sql_query,Text2SQL,extract_sql
load_dotenv()

# reconfigure UnicodeEncodeError prone default (i.e. windows-1252) to utf-8
if sys.platform == "win32" and os.environ.get('PYTHONIOENCODING') is None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger('mcp_sqlite_server')
logger.info("Starting MCP SQLite Server")

api_key = os.environ["NVIDIA_API_KEY"]
print(Fore.LIGHTBLUE_EX , " entered API _KEY ", api_key[-4:])
model_uri = 'https://integrate.api.nvidia.com/v1'
model = "meta/llama-3.1-405b-instruct" 
db_path = "C:\\Users\\zcharpy\\Contacts\\create-python-server\\sqlagent\\src\\sqlagent\\sample_geforce.sqlite"
llm= ChatNVIDIA(model="meta/llama-3.1-70b-instruct" )
schema=get_db_schema(db_path)
print(Fore.GREEN +"schema=\n", schema)
samples = sample_rows(db_path, 3)
text2sql = Text2SQL(llm=llm, schema=schema, samples=samples)



async def main(db_path: str = db_path):
    logger.info(f"Starting SQLite MCP Server with DB path: {db_path}")
    
    server = Server("sqlagent")

    # Register handlers
    logger.debug("Registering handlers")

    @server.list_resources()
    async def handle_list_resources() -> list[types.Resource]:
        logger.debug("Handling list_resources request")
        return [
            types.Resource(
                uri=AnyUrl("memo://insights"),
                name="NVIDIA's geforce sample database",
                description="a SQL agent equipped with geforce database , can answer to gaming questions",
                mimeType="text/plain",
            )
        ]

    @server.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> str:
        logger.debug(f"Handling read_resource request for URI: {uri}")
        if uri.scheme != "memo":
            logger.error(f"Unsupported URI scheme: {uri.scheme}")
            raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

        path = str(uri).replace("memo://", "")
        if not path or path != "insights":
            logger.error(f"Unknown resource path: {path}")
            raise ValueError(f"Unknown resource path: {path}")

        return schema

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        logger.debug("Handling list_prompts request")
        return [
            types.Prompt(
                name="mcp-demo",
                description="A prompt to seed the database with initial data and demonstrate what you can do with an SQLite MCP Server + Claude",
                arguments=[
                    types.PromptArgument(
                        name="topic",
                        description="Topic to seed the database with initial data",
                        required=True,
                    )
                ],
            )
        ]

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        logger.debug(f"Handling get_prompt request for {name} with args {arguments}")
        if name != "mcp-demo":
            logger.error(f"Unknown prompt: {name}")
            raise ValueError(f"Unknown prompt: {name}")

        if not arguments or "topic" not in arguments:
            logger.error("Missing required argument: topic")
            raise ValueError("Missing required argument: topic")

        topic = arguments["topic"]
        prompt = PROMPT_TEMPLATE.format(topic=topic)

        logger.debug(f"Generated prompt template for topic: {topic}")
        return types.GetPromptResult(
            description=f"Demo template for {topic}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt.strip()),
                )
            ],
        )

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="sqlagent",
                description="natural text query to sql agent",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "user input query"},
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool execution requests"""
        try:
            print(Fore.GREEN + "calling tools using the input arguments :\n ", arguments)
            query=arguments["query"]
            output = await text2sql.ainvoke(query=query, reference_queries=[], history=[])
            sql = extract_sql(output)
            print(Fore.CYAN+"extracted sql query from sqlagent =\n", sql)
            print(Fore.CYAN+"executing sql query against the db sample_geforce.sqlite ..." )
            res = execute_sql_query(db_path, sql)
            print(Fore.CYAN+ "output from executing the sql query :\n", res)
            return [types.TextContent(type="text", text=f"{res}")]
        
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async with stdio_server() as (read_stream, write_stream):
        logger.info("Server running with stdio transport")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="sqlite",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

class ServerWrapper():
    """A wrapper to compat with mcp[cli]"""
    def run(self):
        import asyncio
        asyncio.run(main())


wrapper = ServerWrapper()