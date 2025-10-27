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
                        the response in your final report in the scratchpad.""",
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
        template_id="268e2f6b-92c8-44a5-ae4b-0c3a6305fa5b",
        input_params={"tool_names": "test"},
    )
    print(response.parsed_prompt[0].content)
    print(type(response.parsed_prompt[0].content))


if __name__ == "__main__":
    # create_prompt()
    retrieve_prompt()
