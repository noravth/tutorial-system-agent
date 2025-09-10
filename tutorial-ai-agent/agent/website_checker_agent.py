import json
import asyncio
import os
import logging
import nest_asyncio
nest_asyncio.apply()

from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.shell import ShellTool
from langgraph.prebuilt import create_react_agent

from gen_ai_hub.proxy.langchain.init_models import init_llm
import genaihub_client
genaihub_client.set_environment_variables()

# Constants
LOCALHOST_URL = "http://localhost:5173"
TUTORIAL_FILE = "cp-aibus-dox-booster-key.md" #"ui5-webcomponents-react-introduction.md" # 
OUTPUT_FILE = "output_formatted.json"
MAX_RECURSION_LIMIT = 50
MAX_EXECUTION_TIME = 300
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class AgentConfig:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.tool_names = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        self.system_prompt = f"""
            You are an expert software tutorial tester. 
            Your job is to:
            1. Extract a step-by-step action plan from a markdown tutorial. Be aware that there is a lot of irrelevant text in the tutorial that you should ignore. Also ignore the prerequisites and setup steps.
            2. Execute each step in a real browser or in the terminal using the available tools.
            3. Use a scratchpad to reason step by step before taking any action.
            4. For each step, decide which tool to use and with which arguments.
            5. If you need to navigate, use the 'navigate_browser' tool. To click, use 'click_element'. To extract elements, use 'get_elements', etc.
            6. Always explain your reasoning in the scratchpad before using a tool.
            7. After completing ALL steps, provide a final summary and say 'TASK COMPLETED'.

            IMPORTANT: 
            - When you run the app with "npm run dev", navigate to {LOCALHOST_URL} immediately after running the command. Do not wait for any output. It just opens the server and will have to stay open in order for you to open localhost.
            - After navigating to {LOCALHOST_URL}, get the page title to confirm the server is running.
            - Be decisive and don't retry the same action more than twice. If something doesn't work, try an alternative approach or move on.
            - When it says "SAP BTP Trial cockpit" use this url: https://account.hanatrial.ondemand.com/trial/#/globalaccount/0be4add5-cc76-48a6-be37-ab6010d2ffea/accountModel&//?section=SubaccountsSection&view=TilesView and if you did
              that you can also skip clicking on "go to your trial account."

            Available tools:
            {self.tool_names}
            """

class AgentOrchestrator:
    def __init__(self, config):
        self.config = config
        self.agent_chain = create_react_agent(
            model=config.llm,
            tools=config.tools
        )

    async def run(self, markdown):
        system_prompt = self.config.system_prompt
        config = {
            "recursion_limit": MAX_RECURSION_LIMIT,
            "max_execution_time": MAX_EXECUTION_TIME
        }
        result = await self.agent_chain.ainvoke({
            "messages": [
                ("user", f"{system_prompt}\n\nHere is a software tutorial in markdown:\n\n{markdown}\n\nPlease extract an action plan and execute it step by step in the browser. \
                    Use the scratchpad for your reasoning and tool selection. After completing all steps, provide a summary and say 'TASK COMPLETED'.")
            ]
        }, config=config)
        return result

class OutputHandler:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def serialize_message(self, msg):
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            return {"type": msg.type, "content": msg.content}
        return str(msg)

    def save_output(self, messages):
        serializable_messages = [self.serialize_message(m) for m in messages]
        output_path = os.path.join(self.output_dir, OUTPUT_FILE)
        with open(output_path, "w") as f:
            json.dump(serializable_messages, f, indent=4)
        return serializable_messages

async def main():
    logger.info("Starting website checker agent")
    
    # Enable JavaScript in the browser
    async_browser = create_async_playwright_browser(
        headless=False,
        args=['--enable-javascript']  # Add this argument
    )
    logger.info("Created async playwright browser with JavaScript enabled")
    
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()

    # Add ShellTool with a timeout
    shell_tool = ShellTool(timeout=60)  # Set a timeout of 60 seconds
    tools.append(shell_tool)

    logger.info(f"Initialized {len(tools)} tools for the agent")

    # Initialize configurations
    llm = init_llm('anthropic--claude-4-sonnet')
    agent_config = AgentConfig(llm, tools)
    agent_orchestrator = AgentOrchestrator(agent_config)
    output_handler = OutputHandler(os.path.join(ROOT_DIR, '..', 'data', 'output'))

    # Load the tutorial file
    tutorial_path = os.path.join(ROOT_DIR, '..', 'data', 'tutorials', TUTORIAL_FILE)
    with open(tutorial_path, 'r') as f:
        markdown = f.read()
    print(markdown)

    logger.info(f"Processing tutorial with {len(markdown.split())} words")
    
    print("Agent is starting...")
    logger.info("Invoking agent with tutorial instructions")

    # Run the agent
    result = await agent_orchestrator.run(markdown)

    logger.info("Agent execution completed")
    
    print("\n=== Agent Output ===")
    logger.info("Processing agent output for serialization")

    # Handle the output
    messages = result.get("messages", [])
    serializable_messages = output_handler.save_output(messages)
    
    print(serializable_messages)
    logger.info("Website checker agent completed successfully")

if __name__ == "__main__":
    nest_asyncio.apply()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler('website_checker_agent.log')
    stream_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logger.info("Starting main execution")
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise