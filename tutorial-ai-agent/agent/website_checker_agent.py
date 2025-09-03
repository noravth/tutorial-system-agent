import json
import asyncio
import os
import logging
import nest_asyncio
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('website_checker_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import genaihub_client
genaihub_client.set_environment_variables()
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langgraph.prebuilt import create_react_agent

from gen_ai_hub.proxy.gen_ai_hub_proxy.client import Deployment, GOOGLE_VERTEX
from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI, init_chat_model
from gen_ai_hub.proxy.langchain.init_models import catalog
Deployment.prediction_urls.register(
    {'gemini-2.5-flash': GOOGLE_VERTEX,
     'gemini-2.5-pro': GOOGLE_VERTEX}
)

catalog.register(
    "gen-ai-hub",
    ChatVertexAI,
    "gemini-2.5-flash",
    "gemini-2.5-pro",)(init_chat_model)

llm = init_llm('gemini-2.5-pro') #('gpt-4o-mini') 

async def main():
    logger.info("Starting website checker agent")
    
    async_browser = create_async_playwright_browser()
    logger.info("Created async playwright browser")
    
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()
    # Add the secret input tool
    logger.info(f"Initialized {len(tools)} tools for the agent")

    tool_names = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    system_prompt = f"""
    You are an expert software tutorial tester. 
    Your job is to:
    1. Extract a step-by-step action plan from a markdown tutorial.
    2. Execute each step in a real browser using the available tools.
    3. Use a scratchpad to reason step by step before taking any action.
    4. For each step, decide which tool to use and with which arguments.
    5. If you need to navigate, use the 'navigate_browser' tool. To click, use 'click_element'. To extract elements, use 'get_elements', etc.
    6. If you need to login or provide a password, use the 'get_secret_input' tool to prompt the user for input in the terminal.
    7. Always explain your reasoning in the scratchpad before using a tool.

    Available tools:
    {tool_names}
    """

    agent_chain = create_react_agent(
        model=llm,
        tools=tools
    )
    logger.info("Created react agent chain")

    markdown = """
    1. Go to https://python.langchain.com/docs/integrations/tools/playwright/
    2. return the title of the page
    3. Click on the API Reference tab on the top.
    4. And list base packages.
    """
    """markdown = 
    1. https://account.hanatrial.ondemand.com/trial/#/home/trial 
    2. log in to a trial account
    3. return a summary of the page content.
    """
    logger.info(f"Processing tutorial with {len(markdown.split())} words")
    
    print("Agent is starting...")
    logger.info("Invoking agent with tutorial instructions")
    result = await agent_chain.ainvoke({
        "messages": [
            ("user", f"{system_prompt}\n\nHere is a software tutorial in markdown:\n\n{markdown}\n\nPlease extract an action plan and execute it step by step in the browser. \
                Use the scratchpad for your reasoning and tool selection.")
        ]
    })
    logger.info("Agent execution completed")
    
    print("\n=== Agent Output ===")
    logger.info("Processing agent output for serialization")
    # Serialize all messages in result["messages"]
    def serialize_message(msg):
        # Try to extract type and content, fallback to str
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            return {"type": msg.type, "content": msg.content}
        return str(msg)

    serializable_messages = [serialize_message(m) for m in result.get("messages", [])]
    logger.info(f"Serialized {len(serializable_messages)} messages")
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_formatted.json")
    logger.info(f"Saving output to {output_path}")
    
    with open(output_path, "w") as f:
        json.dump(serializable_messages, f, indent=4)
    logger.info("Output saved successfully")
    
    print(serializable_messages)
    logger.info("Website checker agent completed successfully")

if __name__ == "__main__":
    logger.info("Starting main execution")
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise