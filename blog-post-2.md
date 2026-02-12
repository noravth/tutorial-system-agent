# I Built a "Tutorial Tester" Agent and Gave it Internet Access with MCP, LangGraph, and Generative AI Hub.

My colleague asked me: what if we could have an AI agent read our markdown tutorials and actually **execute** the steps in a real browser to verify them?

Challenge accepted! I decided to build exactly that.

In this post, I’m going to walk you through the Python code for a "Tutorial Tester Agent." I’m combining the governance of **SAP Generative AI Hub**, the orchestration power of **LangGraph**, and the exciting new **Model Context Protocol (MCP)** to give my agent eyes and hands on the web.

Let's dive into the code.

---

## The Core Stack

Before we look at the script, let's briefly look at the "ingredients" used in this solution:

* **SAP Generative AI Hub:** This is my gateway to the LLM. It provides secure access to models (I'm using **Claude 3.5 Sonnet**) and a managed **Prompt Registry** so I don't have to hardcode system instructions.
* **Model Context Protocol (MCP):** This is the game-changer. Instead of writing custom functions to control a browser, I'm using the standard **Playwright MCP Server**. It connects the LLM to browser automation tools via a standardized protocol.
* **LangGraph:** The orchestration engine that allows the agent to reason, plan, and loop through steps (Reason  Act  Observe).

---

## Step 1: Setting the Stage

First, we handle our imports and setup. We need to initialize the SAP Generative AI Hub environment to ensure our proxy client is ready to talk to the AI Core.

```python
import os
import json
import logging
from datetime import datetime
import genaihub_client
from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.prompt_registry.client import PromptTemplateClient
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Setup environment variables for SAP AI Core
genaihub_client.set_environment_variables()

```

I’m also setting up **Rich** for pretty console printing because, let’s be honest, staring at raw logs is painful.

---

## Step 2: The Brain (LLM) and The Instructions

I am using `gen_ai_hub` to initialize my model. I chose `anthropic--claude-4-sonnet` because of its superior reasoning capabilities when dealing with complex DOM elements and web navigation.

```python
from gen_ai_hub.proxy.langchain.init_models import init_llm
LLM = init_llm("anthropic--claude-4-sonnet")

```

However, a model is only as good as its prompt. Instead of hardcoding a massive system prompt in my Python file, I stored it in **SAP's Prompt Registry**. This allows me to version-control the prompt and tweak the agent's persona without touching the code.

The function `retrieve_agent_prompt` fetches the template by ID and injects the list of available tools dynamically:

```python
def retrieve_agent_prompt(tools):
    proxy_client = get_proxy_client(proxy_version="gen-ai-hub")
    prompt_registry_client = PromptTemplateClient(proxy_client=proxy_client)
    
    # We fetch a managed prompt template by its unique ID
    response = prompt_registry_client.fill_prompt_template_by_id(
        template_id="c0b92dd9-cfd2-49d8-bfe0-7c65a9c2f9eb",
        input_params={"tool_names": tools},
    )
    return response.parsed_prompt[0].content

```

---

## Step 3: Connecting Tools via MCP (The Magic Part)

This is where things get interesting. Traditionally, if I wanted an agent to browse the web, I’d have to write a custom wrapper around Selenium or Playwright.

With **MCP**, I don't have to.

I simply point my script to the `@playwright/mcp` NPM package. The `StdioServerParameters` tells the script to launch the Playwright MCP server as a subprocess. The agent then automatically "discovers" the tools (like `Maps`, `click`, `fill`, etc.) exposed by that server.

```python
# Set up MCP client for Playwright MCP Server - Browser Automation
server_params = StdioServerParameters(
    command="npx", args=["@playwright/mcp@latest"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # LangChain adapter converts MCP capabilities into LangChain-compatible tools
        tools = await load_mcp_tools(session)

```

This makes the code incredibly modular. If I want to switch from Playwright to a database tool, I just change the `command` and `args`.

---

## Step 4: The Agent Loop

Now we bring it all together using **LangGraph**. We create a prebuilt ReAct agent (`create_react_agent`) which takes our LLM and the tools we loaded from MCP.

We feed the agent the **Markdown Tutorial** as input. The agent's job is to read the tutorial and verify it by actually doing it.

```python
# Initialize the prebuilt agent
agent = create_react_agent(LLM, tools)

# Stream the agent's thought process
async for chunk in agent.astream(
    {
        "messages": [
            {"role": "system", "content": agent_system_prompt},
            {"role": "user", "content": f"Tutorial in Markdown: {markdown}"},
        ]
    },
    stream_mode="updates",
    config={"recursion_limit": RECURSION_LIMIT},
):
    # Process and log the agent's steps...

```

The `astream` function allows us to watch the agent "think" in real-time. It will:

1. **Read** a step from the markdown (e.g., "Go to sap.com").
2. **Decide** to use the `Maps` tool.
3. **Wait** for the MCP server to perform the action in the browser.
4. **Observe** the result and move to the next step.

---

## Why this Architecture?

I built it this way for three specific reasons:

1. **Standardization:** By using MCP, I decouple the *tool definition* from the *agent logic*. I can swap tools in and out easily.
2. **Enterprise Grade:** Using SAP Generative AI Hub means I'm not pasting API keys into my code. The prompt management is handled centrally, which is vital for productionizing AI.
3. **Observability:** The logging setup and output file generation ensure that if the agent fails (e.g., the tutorial is wrong), I have a JSON trace of exactly what step failed and why.

### Conclusion

This script is just the beginning. Imagine an agent that runs every night, checking your internal documentation or onboarding guides, ensuring they are always up to date. By combining the governance of SAP with the flexibility of MCP, we can build agents that are not just chatty, but actually *capable*.

