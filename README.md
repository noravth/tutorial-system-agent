# Tutorial Tester Agent

An AI-powered agent that automatically tests and validates tutorials using browser automation and intelligent decision-making.

## 🚀 Features

- **Browser Automation**: Uses Playwright MCP Server for realistic browser interactions
- **AI-Powered Testing**: LangGraph ReAct agent with Claude 4 Sonnet through SAP's Generative AI Hub 
- **Tutorial Validation**: Automatically executes tutorial steps and identifies issues


## 🛠️ Technology Stack

- **LLM**: Claude 4 Sonnet access through SAP's Generative AI Hub (Cloud SDK for AI - Python)
- **Agent Framework**: LangGraph with ReAct pattern
- **Browser Automation**: Playwright MCP Server


## 📁 Project Structure

```
tutorial-system-agent/
├── tutorial-ai-agent/
│   ├── agent/
│   │   └── tutorial_tester_agent.py    # Main agent
│   └── data/
│       ├── tutorials/                  # Tutorial markdown files
│       └── output/                     # JSON results
├── requirements.txt                    # Python dependencies
├── docker-compose.yml                  # Container orchestration
└── Dockerfile                         # Container definition
```

## 🎯 How It Works

1. **Load Tutorial**: Reads markdown tutorial files
2. **Initialize Agent**: Creates LangGraph ReAct agent with MCP tools
3. **Execute Steps**: Agent follows tutorial instructions in browser
4. **Validate Results**: Checks for completion and identifies issues
5. **Generate Report**: Saves structured JSON logs with timestamps

## 🔧 Configuration

Key settings in `tutorial_tester_agent.py`:
- `TUTORIAL_FILE`: Tutorial to test (default: "ailaunchpad-orchestration.md")
- `RECURSION_LIMIT`: Max agent steps (default: 500)
- `MODEL_ID`: LLM model (default: "anthropic--claude-4-sonnet")




