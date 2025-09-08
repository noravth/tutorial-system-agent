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
from langchain_community.tools.shell import ShellTool
from langgraph.prebuilt import create_react_agent

from gen_ai_hub.proxy.gen_ai_hub_proxy.client import Deployment, GOOGLE_VERTEX
from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI, init_chat_model
from gen_ai_hub.proxy.langchain.init_models import catalog
'''Deployment.prediction_urls.register(
    {'gemini-2.5-flash': GOOGLE_VERTEX,
     'gemini-2.5-pro': GOOGLE_VERTEX}
)

catalog.register(
    "gen-ai-hub",
    ChatVertexAI,
    "gemini-2.5-flash",
    "gemini-2.5-pro",)(init_chat_model)'''

llm = init_llm('anthropic--claude-4-sonnet') #('gemini-2.5-pro') #('gpt-4o-mini') 

async def main():
    logger.info("Starting website checker agent")
    
    async_browser = create_async_playwright_browser(headless=False)
    logger.info("Created async playwright browser")
    
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()

    shell_tool = ShellTool()
    #tools.append(shell_tool)

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
    6. Always explain your reasoning in the scratchpad before using a tool.
    7. After completing ALL steps, provide a final summary and say 'TASK COMPLETED'.
    8. If a step fails after 2 attempts, note the failure and move to the next step.

    IMPORTANT: Be decisive and don't retry the same action more than twice. If something doesn't work, try an alternative approach or move on.

    Available tools:
    {tool_names}
    """

    agent_chain = create_react_agent(
        model=llm,
        tools=tools
    )
    logger.info("Created react agent chain")

    # Simpler test tutorial
    markdown = """### Go To Your Trial Account


1. In your web browser, open the [SAP BTP Trial cockpit](https://account.hanatrial.ondemand.com/trial/#/globalaccount/0be4add5-cc76-48a6-be37-ab6010d2ffea/accountModel&//?section=SubaccountsSection&view=TilesView).

2. Navigate to the trial global account by clicking **Go To Your Trial Account**.

    <!-- border -->![Trial global account](01_Foundation20Onboarding_Home.png)

    >If this is your first time accessing your trial account, you'll have to configure your account by choosing a region. Please select **US East (VA) - AWS**. Your user profile will be set up for you automatically.

    >Wait till your account is set up and ready to go. Your global account, your subaccount, your organization, and your space are launched. This may take a couple of minutes.

    >Choose **Continue**.

    ><!-- border -->![Account setup](02_Foundation20Onboarding_Processing.png)

    >For more details on how to configure entitlements, quotas, subaccounts and service plans on SAP BTP Trial, see [Manage Entitlements on SAP BTP Trial](cp-trial-entitlements).



### Run booster


SAP BTP creates interactive guided boosters to automate cockpit steps, so users can save time when trying out the services.

Now, you will use the **Set up account for SAP Document AI** booster to automatically assign entitlements, update your subaccount, create a service instance and the associated service key for SAP Document AI.

1. On the navigation side bar, click **Boosters**.

    <!-- border -->![Service Key](access-booster.png)

2. Search for **SAP Document AI** and click **Start**.
    
    <!-- border -->![Service Key](access-booster-tile.png)

    >If you have more than one subaccount, the booster will choose automatically the correct subaccount and space, but this will require that you click **Next** twice and **Finish** once, before being able to see the **Success** dialog box.

    <!-- border -->![Service Key](booster-success.png)
"""
    
    
    
    """
    1. Go to https://python.org
    2. Get the page title
    3. Click on the events tab
    4. Extract the first event name and location
    5. Say 'TASK COMPLETED'
    """
    
    """
    1. https://account.hanatrial.ondemand.com/trial/#/home/trial 
    2. log in to a trial account
    3. return a summary of the page content.
    """
    
    
    
    """Open the terminal and type python --version to check the installed Python version."""
    
    
    logger.info(f"Processing tutorial with {len(markdown.split())} words")
    
    print("Agent is starting...")
    logger.info("Invoking agent with tutorial instructions")
    # Add recursion limit configuration
    config = {
        "recursion_limit": 50,  # Increase from default 25
        "max_execution_time": 300  # 5 minutes timeout
    }
    
    result = await agent_chain.ainvoke({
        "messages": [
            ("user", f"{system_prompt}\n\nHere is a software tutorial in markdown:\n\n{markdown}\n\nPlease extract an action plan and execute it step by step in the browser. \
                Use the scratchpad for your reasoning and tool selection. After completing all steps, provide a summary and say 'TASK COMPLETED'.")
        ]
    }, config=config)
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