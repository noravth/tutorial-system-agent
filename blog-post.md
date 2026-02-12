## I Built a Tutorial-Testing AI Agent with LangGraph, MCP, and Generative AI Hub — Here’s What’s Actually Going On

I wanted an **agent** that could *read a technical tutorial*, open a browser, actually try things out, and tell me whether the tutorial works.

So I wired together:

* **SAP Generative AI Hub** (for enterprise-grade LLM access)
* **LangGraph’s prebuilt ReAct agent**
* **MCP (Model Context Protocol)** with Playwright (for browser automation)
* A prompt from **SAP AI Core Prompt Registry**
* And a Markdown tutorial as input

The result? A “Tutorial Tester Agent” that reads a Markdown file and actively evaluates it using browser automation tools.

Let’s break down what this script is doing — and more importantly, what architectural ideas are hiding underneath.

---

# The Big Picture

At a high level, this agent:

1. Loads a tutorial written in Markdown
2. Connects to a Playwright MCP server
3. Turns Playwright into usable AI tools
4. Pulls a system prompt from SAP AI Core
5. Creates a ReAct-style agent
6. Streams its reasoning step-by-step
7. Logs everything to JSON for inspection

This is not just LLM prompting.

This is **LLM + Tools + Structured Orchestration + Observability**.

That’s what makes it an *agent*.

---

# Step 1 — The LLM Comes from SAP Generative AI Hub

```python
from gen_ai_hub.proxy.langchain.init_models import init_llm
LLM = init_llm("anthropic--claude-4-sonnet")
```

Instead of calling Anthropic directly, the model is initialized via **SAP Generative AI Hub**.

Why does that matter?

Because in enterprise environments:

* You don’t hardcode API keys
* You don’t hit public endpoints
* You go through managed AI infrastructure
* You can swap models without rewriting your app

This line abstracts all of that away.

From the agent’s perspective, it just has an LLM.

From an enterprise perspective, governance is intact.

---

# Step 2 — MCP: Turning Playwright into AI Tools

This is where it gets interesting.

```python
server_params = StdioServerParameters(
    command="npx", args=["@playwright/mcp@latest"]
)
```

This launches a **Playwright MCP server**.

Then:

```python
tools = await load_mcp_tools(session)
```

This converts MCP capabilities into LangChain-compatible tools.

## What’s happening conceptually?

MCP (Model Context Protocol) acts as a bridge between:

* The LLM
* External systems (like browsers)

Instead of writing custom tool wrappers, MCP exposes:

* navigate()
* click()
* fill()
* extract_text()
* etc.

And the agent can dynamically use them.

This means your LLM can:

* Open websites
* Interact with UI
* Validate tutorial steps
* Check if instructions actually work

This is where it stops being theoretical.

It becomes executable reasoning.

---

# Step 3 — The Agent Pattern: ReAct

```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(LLM, tools)
```

This uses a prebuilt **ReAct agent**.

ReAct = Reason + Act

The loop looks like this internally:

1. Think
2. Decide which tool to call
3. Execute tool
4. Observe result
5. Think again

That loop continues until it decides it’s done.

You don’t manually code that loop.

LangGraph handles the orchestration.

---

# Step 4 — The System Prompt Is Retrieved Dynamically

This part is subtle — and powerful.

```python
response = prompt_registry_client.fill_prompt_template_by_id(
    template_id="c0b92dd9-cfd2-49d8-bfe0-7c65a9c2f9eb",
    input_params={"tool_names": tools},
)
```

The agent’s system prompt isn’t hardcoded.

It’s pulled from **SAP AI Core Prompt Registry**.

Why this matters:

* You can update behavior without redeploying code
* Prompts are versioned
* Prompts are centrally managed
* Multiple agents can share governance

This is production-grade prompt engineering.

Not notebook experimentation.

---

# Step 5 — Feeding the Tutorial to the Agent

```python
with open(tutorial_path, "r") as f:
    markdown = f.read()
```

Then:

```python
{
    "role": "user",
    "content": f"""Tutorial in Markdown: {markdown}""",
}
```

The entire Markdown tutorial becomes input context.

Now the agent has:

* A goal (defined in the system prompt)
* A tutorial
* A browser
* A toolset

And it can start evaluating.

---

# Step 6 — Streaming the Agent’s Brain

This is one of my favorite parts:

```python
async for chunk in agent.astream(...):
```

Instead of waiting for a final answer, we stream updates.

Each chunk contains:

* The current step
* The latest message
* Tool invocations
* Observations

And we log it:

```python
output_dict = {"step": step, "content": content}
```

Why this matters:

If you’re building agents and you don’t log intermediate reasoning, you’re flying blind.

Streaming gives you:

* Debug visibility
* Traceability
* Reproducibility
* Post-run analysis

This is how you turn experimentation into engineering.

---

# Step 7 — Recursion Limit (Yes, This Is Important)

```python
RECURSION_LIMIT = 500
```

Agents can loop.

Bad prompts = infinite loops.

This config ensures:

“If you exceed 500 reasoning steps, stop.”

It’s a safety rail.

In agent systems, guardrails matter.

---

# The Architecture in Plain English

Here’s what’s really happening:

| Layer           | Responsibility             |
| --------------- | -------------------------- |
| SAP Gen AI Hub  | Enterprise LLM access      |
| Prompt Registry | Controlled system behavior |
| LangGraph       | Agent reasoning loop       |
| MCP             | Tool protocol abstraction  |
| Playwright      | Real browser automation    |
| Rich + Logging  | Observability              |
| JSON output     | Persisted trace            |

Each layer has one job.

Together, they create a system that can:

> Read instructions.
> Try them.
> Validate them.
> Report back.

That’s powerful.

---

# Why This Isn’t “Just Another Agent”

Most tutorials stop at:

“Here’s how to build a chatbot.”

This goes further:

* It interacts with real systems.
* It uses enterprise AI infrastructure.
* It separates prompts from code.
* It streams reasoning.
* It logs everything.

It’s closer to **AI QA automation** than chat.

And that’s where agents get interesting.

---

# What You’re Really Learning From This Script

Underneath the imports and async calls, this code teaches five big ideas:

### 1️⃣ LLMs are reasoning engines — not just text generators

### 2️⃣ Tools turn LLMs into actors

### 3️⃣ MCP standardizes tool connectivity

### 4️⃣ LangGraph manages agent loops safely

### 5️⃣ Observability is mandatory in production agents

If you remove any of these, the system gets weaker.

Together, they create a controllable AI worker.

---

# What I’d Improve Next

If I were evolving this:

* Add structured evaluation scoring
* Add pass/fail outputs
* Capture screenshots on failure
* Persist full tool traces
* Integrate with CI/CD

Because once an agent can test tutorials…

It can test documentation.
It can test onboarding flows.
It can test apps.

That’s where this goes.

---

# Final Thoughts

This script isn’t flashy.

It doesn’t generate memes.

It doesn’t chat.

But it demonstrates something far more important:

**How to architect a real AI agent system in an enterprise setting.**

LLM + Tools + Governance + Observability.

That’s the pattern.

And once you understand that pattern, you stop building demos — and start building systems.
