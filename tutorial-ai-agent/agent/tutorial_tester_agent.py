import os
import json
import logging
import pprint
import re
from datetime import datetime
from typing import Any

import genaihub_client
genaihub_client.set_environment_variables()

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langgraph.prebuilt import create_react_agent

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add these imports for better formatting
import rich
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(ROOT_DIR, '..', 'data', 'output', 'output_formatted_mcp.json')
TUTORIAL_FILE = "appstudio-devspace-create.md" #"sap-subscribe-booster.md" #"cp-aibus-dox-booster-key.md"

llm = init_llm('anthropic--claude-4-sonnet')

# Initialize rich console for pretty printing
console = Console()

# Configuration for text truncation
MAX_TEXT_LENGTH = 500  # Characters before truncation
MAX_LINES = 15  # Lines before truncation

def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH, max_lines: int = MAX_LINES) -> tuple[str, bool]:
    """Truncate text if it's too long, return (truncated_text, was_truncated)"""
    lines = text.split('\n')
    
    # Check if truncation is needed
    needs_truncation = len(text) > max_length or len(lines) > max_lines
    
    if not needs_truncation:
        return text, False
    
    # Truncate by lines first
    if len(lines) > max_lines:
        truncated_lines = lines[:max_lines]
        truncated_text = '\n'.join(truncated_lines)
    else:
        truncated_text = text
    
    # Then truncate by character length
    if len(truncated_text) > max_length:
        truncated_text = truncated_text[:max_length] + "..."
    
    return truncated_text, True

def is_final_summary(content: str) -> bool:
    """Check if content appears to be a final summary"""
    summary_indicators = [
        "summary",
        "completed",
        "finished",
        "final report",
        "conclusion",
        "overall",
        "in total",
        "to summarize"
    ]
    content_lower = content.lower()
    
    # Check if it contains summary indicators and appears near the end
    has_summary_words = any(indicator in content_lower for indicator in summary_indicators)
    
    # Additional heuristics for final summary
    has_completion_marker = "completed" in content_lower
    is_substantial = len(content) > 200  # Final summaries are usually substantial
    
    return has_summary_words and (has_completion_marker or is_substantial)

def show_content(content: str, title: str, border_style: str = "cyan", force_full: bool = False):
    """Show content with automatic truncation unless it's marked as final summary"""
    if force_full or is_final_summary(content):
        # Show full content for final summaries
        if isinstance(content, str):
            try:
                md = Markdown(content)
                panel = Panel(md, title=f"{title} [dim](full)[/dim]", border_style=border_style)
            except:
                panel = Panel(content, title=f"{title} [dim](full)[/dim]", border_style=border_style)
        else:
            panel = Panel(str(content), title=f"{title} [dim](full)[/dim]", border_style=border_style)
        console.print(panel)
    else:
        # Automatically truncate other content
        truncated_content, was_truncated = truncate_text(content)
        
        if was_truncated:
            # Show truncated version with indication
            try:
                md = Markdown(truncated_content)
                panel = Panel(
                    md, 
                    title=f"{title} [dim](truncated - {len(content)} chars, {len(content.split(chr(10)))} lines)[/dim]", 
                    border_style=border_style
                )
            except:
                panel = Panel(
                    truncated_content, 
                    title=f"{title} [dim](truncated - {len(content)} chars, {len(content.split(chr(10)))} lines)[/dim]", 
                    border_style=border_style
                )
            console.print(panel)
        else:
            # Content is short enough, show directly
            try:
                md = Markdown(content)
                panel = Panel(md, title=title, border_style=border_style)
            except:
                panel = Panel(content, title=title, border_style=border_style)
            console.print(panel)

def print_chunk_formatted(chunk):
    """Pretty print chunk with tool usage and formatted content"""
    console.print("\n" + "="*80)
    console.print(f"[bold blue]CHUNK UPDATE[/bold blue]")
    console.print("="*80)
    
    for node_name, node_data in chunk.items():
        console.print(f"\n[bold green]Node: {node_name}[/bold green]")
        
        if 'messages' in node_data:
            for i, message in enumerate(node_data['messages']):
                console.print(f"\n[bold yellow]Message {i+1}:[/bold yellow]")
                
                # Print message type and role if available
                if hasattr(message, 'type'):
                    console.print(f"[dim]Type: {message.type}[/dim]")
                
                # Handle different message types
                if hasattr(message, 'content'):
                    content = message.content
                    
                    # If content is a list (tool calls/responses)
                    if isinstance(content, list):
                        for j, item in enumerate(content):
                            if isinstance(item, dict):
                                if item.get('type') == 'tool_use':
                                    # Tool usage
                                    tool_input = json.dumps(item.get('input', {}), indent=2)
                                    tool_content = (
                                        f"[bold]Tool:[/bold] {item.get('name', 'Unknown')}\n"
                                        f"[bold]ID:[/bold] {item.get('id', 'N/A')}\n"
                                        f"[bold]Input:[/bold]\n{tool_input}"
                                    )
                                    show_content(tool_content, "🔧 Tool Call", "blue")
                                    
                                elif item.get('type') == 'tool_result':
                                    # Tool result
                                    result_content = item.get('content', '')
                                    if isinstance(result_content, list):
                                        result_content = json.dumps(result_content, indent=2)
                                    
                                    tool_result_content = (
                                        f"[bold]Tool ID:[/bold] {item.get('tool_use_id', 'N/A')}\n"
                                        f"[bold]Result:[/bold]\n{result_content}"
                                    )
                                    show_content(tool_result_content, "✅ Tool Result", "green")
                                    
                                elif item.get('type') == 'text':
                                    # Text content - render as markdown
                                    text_content = item.get('text', '')
                                    if text_content.strip():
                                        show_content(text_content, "💬 Message Content", "cyan")
                                else:
                                    # Other content types
                                    other_content = json.dumps(item, indent=2)
                                    show_content(other_content, f"📄 Content item {j+1}", "yellow")
                            else:
                                console.print(f"[dim]Content item {j+1}: {item}[/dim]")
                    
                    # If content is a string
                    elif isinstance(content, str) and content.strip():
                        show_content(content, "💬 Message Content", "cyan")
                    
                    # If content is other type
                    else:
                        content_str = str(content)
                        show_content(content_str, "📄 Content", "yellow")
                
                # Print other message attributes
                if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                    kwargs_str = json.dumps(message.additional_kwargs, indent=2)
                    show_content(kwargs_str, "⚙️ Additional kwargs", "magenta")
        
        # Handle other node data
        else:
            node_data_str = json.dumps(node_data, indent=2, default=str)
            show_content(node_data_str, "📊 Node data", "white")
    
    console.print("="*80 + "\n")

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
            
            console.print("[bold magenta]🤖 Starting Tutorial Tester Agent[/bold magenta]")
            console.print(f"[bold]Available Tools:[/bold]\n{tool_names}")
            console.print(f"[bold]Tutorial File:[/bold] {TUTORIAL_FILE}")
            console.print("[dim]Note: Large content will be automatically truncated except for final summaries.[/dim]\n")
            
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
                config={"recursion_limit": 50}
            ):
                print_chunk_formatted(chunk)

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