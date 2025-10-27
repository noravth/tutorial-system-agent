import os
import json
import logging
from datetime import datetime

import genaihub_client
genaihub_client.set_environment_variables()

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langgraph.prebuilt import create_react_agent

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add these imports for better formatting
from rich.console import Console

# At the top of your script, after imports:
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(ROOT_DIR, '..', 'data', 'output', f'output_formatted_mcp_{timestamp}.json')
TUTORIAL_FILE = "ailaunchpad-orchestration.md"
RECURSION_LIMIT = 500

# Assign LLM
LLM = init_llm('anthropic--claude-4-sonnet')

# Initialize rich console for better printing
console = Console()

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
            agent = create_react_agent(LLM, tools)
            
            console.print("[bold magenta]🤖 Starting Tutorial Tester Agent[/bold magenta]")
            console.print(f"[bold]Available Tools:[/bold]\n{tool_names}")
            console.print(f"[bold]Tutorial File:[/bold] {TUTORIAL_FILE}")
            
            logger.info("Starting agent stream processing.")
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": f"""You are a tutorial tester agent. Use the scratchpad for your reasoning and tool selection.
                      Read the following tutorial in markdown.
                      Extract the steps you need to follow to complete the tutorial.
                      On the way take notes on where the tutorial was not clear enough and provide feedback.
                      You have the following tools available to you: {tool_names}
                      Summarize what you did and say completed at the end.
                      If necessary due to browser loading times use the wait tool.
                      IMPORTANT! ONLY run the booster or create services/instances/subscriptions 
                      when you could not find the respective instance or subscription in your 
                      trial subaccount under services -> instances and subscriptions. 
                      If you can find it there, do not run the booster, but continue 
                      with the next step of the tutorial after running the booster.
                     
                      When you create services/instances/subscriptions/dev spaces or run boosters
                      or anything else that requires initializing a system make sure you give it time
                      with your wait tool to not run out of memory before the system is fully initialized.

                      Always answer every question in the tutorial based on what you read in the 
                      tutorial or based on what you did in the system. Include the question and
                      the response in your final report in the scratchpad.

                      Tutorial in Markdown: {markdown}"""}]},
                stream_mode="updates",
                config={"recursion_limit": RECURSION_LIMIT}
            ):
                for step, data in chunk.items():
                    console.print(f"step: {step}")
                    content = data['messages'][-1].content
                    console.print(f"content: {data['messages'][-1].content}")

                    # Prepare the output dict
                    output_dict = {
                        "step": step,
                        "content": content
                    }
                    # Append to file as a JSON line
                    with open(OUTPUT_FILE, "a") as f:
                        f.write(json.dumps(output_dict, ensure_ascii=False))
                        f.write("\n")

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