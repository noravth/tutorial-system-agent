"""
Tutorial Tester Agent using LangGraph and MCP.

This module implements an AI agent that automatically tests and validates
tutorials by executing them step-by-step using browser automation through
the Playwright MCP Server. It leverages LangGraph's ReAct pattern with
LLM-powered decision making and tool orchestration.

Key Components:
- LLM: SAP GenAI Hub Claude 4 Sonnet
- Tools: Playwright MCP Server for browser automation
- Agent: LangGraph ReAct agent with streaming capabilities
- Output: Structured JSON logs of agent decisions and results
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import genaihub_client
from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.proxy.langchain.init_models import init_llm
from gen_ai_hub.prompt_registry.client import PromptTemplateClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console

# ============================================================================
# Configuration Constants
# ============================================================================

# Directory and path configuration
ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR.parent / "data"
OUTPUT_DIR = DATA_DIR / "output"
TUTORIALS_DIR = DATA_DIR / "tutorials"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique output file naming
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Agent configuration
TUTORIAL_FILE = "ailaunchpad-orchestration.md"
RECURSION_LIMIT = 500
MODEL_ID = "anthropic--claude-4-sonnet"

# Prompt registry template ID for system prompt
SYSTEM_PROMPT_TEMPLATE_ID = "c0b92dd9-cfd2-49d8-bfe0-7c65a9c2f9eb"

# MCP Server configuration
MCP_SERVER_COMMAND = "npx"
MCP_SERVER_ARGS = ["@playwright/mcp@latest"]

# ============================================================================
# Global Instances
# ============================================================================

# Initialize environment and LLM
genaihub_client.set_environment_variables()
LLM = init_llm(MODEL_ID)

# Rich console for formatted output
console = Console()



# ============================================================================
# Utility Functions
# ============================================================================


def get_output_file_path(use_timestamp: bool = True) -> Path:
    """
    Generate output file path for agent results.

    Args:
        use_timestamp: If True, includes timestamp in filename for uniqueness.

    Returns:
        Path object pointing to the output JSON file.
    """
    if use_timestamp:
        filename = f"output_formatted_mcp_{TIMESTAMP}.json"
    else:
        filename = "output_formatted_mcp.json"
    return OUTPUT_DIR / filename


def retrieve_system_prompt(tool_descriptions: str) -> str:
    """
    Retrieve the system prompt for the agent from SAP Prompt Registry.

    This function fetches a pre-configured prompt template from the SAP
    GenAI Hub Prompt Registry and fills it with current tool descriptions.

    Args:
        tool_descriptions: String of available tools and their descriptions.

    Returns:
        The filled system prompt content.

    Raises:
        Exception: If the API call fails or template is not found.
    """
    try:
        proxy_client = get_proxy_client(proxy_version="gen-ai-hub")
        prompt_client = PromptTemplateClient(proxy_client=proxy_client)
        response = prompt_client.fill_prompt_template_by_id(
            template_id=SYSTEM_PROMPT_TEMPLATE_ID,
            input_params={"tool_names": tool_descriptions},
        )
        return response.parsed_prompt[0].content
    except Exception as e:
        logging.error(f"Failed to retrieve system prompt: {e}")
        raise


def load_tutorial_file(filename: str) -> str:
    """
    Load tutorial content from a markdown file.

    Args:
        filename: Name of the tutorial file in the tutorials directory.

    Returns:
        The content of the tutorial file as a string.

    Raises:
        FileNotFoundError: If the tutorial file does not exist.
    """
    tutorial_path = TUTORIALS_DIR / filename
    if not tutorial_path.exists():
        raise FileNotFoundError(f"Tutorial file not found: {tutorial_path}")

    with open(tutorial_path, "r", encoding="utf-8") as f:
        return f.read()


def save_agent_result(step: str, content: str, output_path: Optional[Path] = None) -> None:
    """
    Save individual agent execution result to JSON file.

    Appends a single step's result as a JSON line to the output file.
    This streaming approach allows monitoring results in real-time.

    Args:
        step: The step identifier (e.g., 'agent', 'tool_call').
        content: The content/message from this step.
        output_path: Optional custom output path. Uses default if not specified.
    """
    if output_path is None:
        output_path = get_output_file_path(use_timestamp=True)

    output_dict = {"step": step, "content": content, "timestamp": datetime.now().isoformat()}
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
    except IOError as e:
        logging.error(f"Failed to save result to {output_path}: {e}")
        raise



# ============================================================================
# Main Agent Execution
# ============================================================================


async def main() -> None:
    """
    Main entry point for the Tutorial Tester Agent.

    This function orchestrates the entire agent workflow:
    1. Initializes MCP client for Playwright browser automation
    2. Loads available tools from the MCP server
    3. Loads the tutorial markdown file
    4. Creates a ReAct agent with the LLM and tools
    5. Executes the agent in streaming mode
    6. Saves results to JSON file

    The agent processes the tutorial and makes decisions about what
    steps to execute based on the tutorial content.

    Raises:
        Exception: If MCP server fails, tutorial file not found, or execution errors occur.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("Starting Tutorial Tester Agent")
    logger.info("=" * 70)

    try:
        # Initialize MCP server for Playwright browser automation
        logger.info("Initializing MCP server for Playwright automation...")
        server_params = StdioServerParameters(
            command=MCP_SERVER_COMMAND, args=MCP_SERVER_ARGS
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("MCP session initialized successfully")

                # Load tools from MCP server
                logger.info("Loading tools from MCP server...")
                tools = await load_mcp_tools(session)
                logger.info(f"Loaded {len(tools)} tools from MCP server")

                # Format tool descriptions for display and prompt
                tool_names = "\n".join(
                    [f"- {tool.name}: {tool.description}" for tool in tools]
                )

                # Load tutorial file
                logger.info(f"Loading tutorial file: {TUTORIAL_FILE}")
                try:
                    markdown = load_tutorial_file(TUTORIAL_FILE)
                    logger.info(
                        f"Tutorial loaded successfully ({len(markdown)} characters)"
                    )
                except FileNotFoundError as e:
                    logger.error(f"Tutorial file not found: {e}")
                    raise

                # Create ReAct agent
                logger.info("Creating ReAct agent with LLM and tools...")
                agent = create_react_agent(LLM, tools)
                logger.info("Agent created successfully")

                # Display startup information
                console.print(
                    "[bold magenta]🤖 Tutorial Tester Agent[/bold magenta]"
                )
                console.print("[bold cyan]─" * 60 + "─[/bold cyan]")
                console.print(f"[bold yellow]Model:[/bold yellow] {MODEL_ID}")
                console.print(f"[bold yellow]Tutorial:[/bold yellow] {TUTORIAL_FILE}")
                console.print(f"[bold yellow]Output:[/bold yellow] {get_output_file_path()}")
                console.print(
                    f"[bold green]Available Tools ({len(tools)}):[/bold green]"
                )
                console.print(tool_names)
                console.print("[bold cyan]─" * 60 + "─[/bold cyan]")

                # Retrieve system prompt
                logger.info("Retrieving system prompt from Prompt Registry...")
                try:
                    agent_system_prompt = retrieve_system_prompt(tool_names)
                    logger.info("System prompt retrieved successfully")
                except Exception as e:
                    logger.error(f"Failed to retrieve system prompt: {e}")
                    raise

                # Request confirmation before starting
                user_input = input(
                    "\n[?] Start the tutorial tester agent? (yes/no): "
                ).strip().lower()
                if user_input not in ("yes", "y"):
                    logger.info("Agent execution cancelled by user")
                    console.print("[yellow]Agent execution cancelled[/yellow]")
                    return

                # Execute agent with streaming
                logger.info("Starting agent execution in streaming mode...")
                output_path = get_output_file_path(use_timestamp=True)
                step_count = 0

                async for chunk in agent.astream(
                    {
                        "messages": [
                            {"role": "system", "content": agent_system_prompt},
                            {
                                "role": "user",
                                "content": f"Please test this tutorial:\n\n{markdown}",
                            },
                        ]
                    },
                    stream_mode="updates",
                    config={"recursion_limit": RECURSION_LIMIT},
                ):
                    for step, data in chunk.items():
                        step_count += 1
                        logger.info(f"Step {step_count}: {step}")

                        # Extract message content
                        try:
                            message = data.get("messages", [])
                            if message:
                                content = message[-1].content
                                console.print(
                                    f"[bold blue]Step {step_count}: {step}[/bold blue]"
                                )
                                console.print(f"{content}\n")
                                logger.debug(f"Content: {content}")

                                # Save result
                                save_agent_result(step, content, output_path)
                        except (KeyError, IndexError, AttributeError) as e:
                            logger.warning(
                                f"Failed to extract message from step {step}: {e}"
                            )

                logger.info(f"Agent execution completed ({step_count} steps)")
                console.print(
                    f"[bold green]✓ Agent execution completed ({step_count} steps)[/bold green]"
                )
                console.print(f"[bold green]Results saved to: {output_path}[/bold green]")

    except Exception as e:
        logger.error(f"Fatal error during agent execution: {e}", exc_info=True)
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        raise


# ============================================================================
# Entry Point and Logging Setup
# ============================================================================


def setup_logging() -> logging.Logger:
    """
    Configure logging with both file and console handlers.

    Returns:
        Configured logger instance for the module.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # File handler - logs all levels
    log_file = ROOT_DIR / "tutorial_tester_agent.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Console handler - info level and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    """
    Script entry point.

    Sets up logging and runs the async main() function using asyncio.
    """
    logger = setup_logging()
    logger.info("Tutorial Tester Agent starting...")
    logger.info(f"Configuration: Model={MODEL_ID}, Tutorial={TUTORIAL_FILE}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    try:
        asyncio.run(main())
        logger.info("Tutorial Tester Agent completed successfully")
    except KeyboardInterrupt:
        logger.info("Agent interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise

