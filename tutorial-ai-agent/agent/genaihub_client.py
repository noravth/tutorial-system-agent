import os
import json

import variables

ROOT_PATH_DIR = os.path.dirname(os.getcwd())
AICORE_CONFIG_FILENAME = ROOT_PATH_DIR+'/tutorial-system-agent/tutorial-ai-agent/.aicore-config.json'

def set_environment_variables() -> None:
    with open(os.path.join(ROOT_PATH_DIR, AICORE_CONFIG_FILENAME), 'r') as config_file:
        config_data = json.load(config_file)

    os.environ["AICORE_AUTH_URL"]=config_data["url"]+"/oauth/token"
    os.environ["AICORE_CLIENT_ID"]=config_data["clientid"]
    os.environ["AICORE_CLIENT_SECRET"]=config_data["clientsecret"]
    os.environ["AICORE_BASE_URL"]=config_data["serviceurls"]["AI_API_URL"]

    os.environ["AICORE_RESOURCE_GROUP"]=variables.RESOURCE_GROUP