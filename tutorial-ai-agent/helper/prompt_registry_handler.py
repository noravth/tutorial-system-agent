import os
import json

ROOT_PATH_DIR = os.path.dirname(os.getcwd())
AICORE_CONFIG_FILENAME = (
    ROOT_PATH_DIR + "/tutorial-system-agent/tutorial-ai-agent/.aicore-config.json"
)
RESOURCE_GROUP = "default"

with open(os.path.join(ROOT_PATH_DIR, AICORE_CONFIG_FILENAME), "r") as config_file:
    config_data = json.load(config_file)

os.environ["AICORE_AUTH_URL"] = config_data["url"] + "/oauth/token"
os.environ["AICORE_CLIENT_ID"] = config_data["clientid"]
os.environ["AICORE_CLIENT_SECRET"] = config_data["clientsecret"]
os.environ["AICORE_BASE_URL"] = config_data["serviceurls"]["AI_API_URL"]

os.environ["AICORE_RESOURCE_GROUP"] = RESOURCE_GROUP

from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.prompt_registry.client import PromptTemplateClient


from gen_ai_hub.prompt_registry.models.prompt_template import (
    PromptTemplateSpec,
    PromptTemplate,
)


def create_prompt():
    proxy_client = get_proxy_client(proxy_version="gen-ai-hub")
    prompt_registry_client = PromptTemplateClient(proxy_client=proxy_client)
    prompt_template_spec = PromptTemplateSpec(
        template=[
            PromptTemplate(
                role="system",
                content="""You are a tutorial tester agent. Use the scratchpad for your reasoning and tool selection.
                        Read the following tutorial in markdown.
                        Extract the steps you need to follow to complete the tutorial.
                        On the way take notes on where the tutorial was not clear enough and provide feedback.
                        You have the following tools available to you: {{ ?tool_names }}
                        If necessary due to browser loading times use the wait tool.
                        Important: ALWAYS end with a short feedback section with bullet points to improve the tutorial. 
                        Use the headline FEEDBACK: to indicate the beginning of the feedback section.
                        """,
            )
        ]
    )

    template_id = prompt_registry_client.create_prompt_template(
        scenario="TechEd25",
        name="tutorial_tester_ai_agent_prompt",
        version="1.0.0",
        prompt_template_spec=prompt_template_spec,
    ).id

    print(f"Created Prompt Template with ID: {template_id}")


def retrieve_prompt():
    proxy_client = get_proxy_client(proxy_version="gen-ai-hub")
    prompt_registry_client = PromptTemplateClient(proxy_client=proxy_client)
    response = prompt_registry_client.fill_prompt_template_by_id(
        template_id="c0b92dd9-cfd2-49d8-bfe0-7c65a9c2f9eb",
        input_params={"tool_names": "test"},
    )
    print(response.parsed_prompt[0].content)
    print(type(response.parsed_prompt[0].content))


if __name__ == "__main__":
    #create_prompt()
    #retrieve_prompt()
    print('''
    ### ⚠️ Issues Encountered:

1. **Grounding Configuration Issue** - The initial template included grounding variables that weren't needed for the sentiment analysis use case, causing parameter validation errors
2. **Output Filtering Limitation** - Could not disable output filtering due to model-specific restrictions
3. **Model Selection** - The tutorial mentions GPT-4 or Claude, but only Pharia-1 7b Control was available

---

## FEEDBACK:

• **Tutorial Structure Clarity**: The tutorial provides good step-by-step instructions but could benefit from clearer prerequisites about which models should be available and configured.

• **Grounding Module Confusion**: The tutorial doesn't clearly explain when grounding should be used vs. disabled. For the sentiment analysis use case, grounding isn't necessary, but this 
isn't explicitly stated. Consider adding guidance on when to enable/disable grounding.

• **Model Selection Gap**: The tutorial mentions using "GPT-4 or Claude" but the available model was "Pharia-1 7b Control". Update the tutorial to reflect actually available models or provide
guidance on model deployment.

• **Output Filtering Limitation**: Add a note that some models (like Pharia-1 7b Control) have built-in content filtering that cannot be disabled, so users shouldn't expect to be able to turn
off output filtering for all models.

• **Error Handling**: Include a troubleshooting section for common errors like "Unused parameters" validation issues and how to resolve them.

• **Data Masking Verification**: The tutorial could benefit from showing how to verify that data masking actually worked by examining the trace or providing before/after examples.

• **Variable Management**: Better explain how template variables work and when they get created/removed automatically based on module configuration changes.

• **Screenshot References**: Some steps reference "see the screenshot below" but no screenshots are provided in the markdown tutorial.

• **Success Criteria**: Add clearer success criteria so users know what a successful execution should look like (token counts, response format, etc.).

• **Real-world Context**: Provide more context about when and why you would use each orchestration module in production scenarios.
    ''')
