import os
import json
import logging

import genaihub_client
genaihub_client.set_environment_variables()

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langgraph.prebuilt import create_react_agent

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(ROOT_DIR, '..', 'data', 'output', 'output_formatted_mcp.json')
TUTORIAL_FILE = "cp-aibus-dox-booster-key-test.md"

llm = init_llm('anthropic--claude-4-sonnet')

def serialize_message(msg):
    if hasattr(msg, 'type') and hasattr(msg, 'content'):
        return {"type": msg.type, "content": msg.content}
    return str(msg)

def save_output(messages):
    serializable_messages = [serialize_message(m) for m in messages]
    output_path = os.path.join(ROOT_DIR, '..', 'data', 'output', 'output_formatted_mcp.json')
    with open(output_path, "w") as f:
        json.dump(serializable_messages, f, indent=4)
    return serializable_messages

async def main():
    logger.info("Starting website checker agent")
    logger.info("Importing LangChainAdapter and creating adapter")

    server_params = StdioServerParameters(
        command="npx",
        # Make sure to update to the full absolute path to your math_server.py file
        args=["@playwright/mcp@latest"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            logger.info("LangChainAdapter created. Creating tools...")
            tools = await load_mcp_tools(session)
            tool_names = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

            tutorial_path = os.path.join(ROOT_DIR, '..', 'data', 'tutorials', TUTORIAL_FILE)
            with open(tutorial_path, 'r') as f:
                markdown = f.read()

            logger.info(f"Tools created: {tools}")

            logger.info("LLM initialized and create_react_agent.")
            agent = create_react_agent(llm, tools)
            # Run a query that will leverage browser tools
            result = await agent.ainvoke({
                "messages": [
                    ("user", f"""You are a tutorial tester agent. Use the scratchpad for your reasoning and tool selection.
                     Read the following tutorial in markdown.
                     Extract the steps you need to follow to complete the tutorial.
                     On the way take notes on where the tutorial was not clear enough and provide feedback.
                     You have the following tools available to you: {tool_names}
                     Summarize what you did and say completed at the end.
                     If necessary due to browser loading times use the wait tool.
                     
                     Tutorial in Markdown: {markdown}"""
                    )]
                },
                {"recursion_limit": 50}
            )
            logger.info(f"LLM invocation result: {result}")         
            messages = result.get("messages", [])
            serializable_messages = save_output(messages)
            print(serializable_messages)


if __name__ == "__main__":
    import asyncio
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create handlers
    file_handler = logging.FileHandler('tutorial_tester_agent.log')
    stream_handler = logging.StreamHandler()
    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("Logger initialized. Starting main execution.")
    logger.info("Initializing MCP client and LLM.")
    try:
        logger.info("Running main async function.")
        asyncio.run(main())
        logger.info("Main execution finished successfully.")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise