from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.document_grounding import (GroundingModule, GroundingType, DocumentGrounding,DocumentGroundingFilter)
from gen_ai_hub.orchestration.models.llm import LLM
import variables

import genaihub_client
genaihub_client.set_environment_variables()

orchestration_service_url = variables.AICORE_ORCHESTRATION_DEPLOYMENT_URL
orchestration_service = OrchestrationService(api_url=orchestration_service_url)

llm = LLM(
    name="gemini-1.5-flash",
    parameters={
        'temperature': 0.0,
    }
)

prompt = Template(messages=[
        SystemMessage("You are an expert on SAP Product features."),
        UserMessage("""Context: {{ ?grounding_response }}
                       Question: What are the features of {{ ?product }}
                    """),
    ])

filters = [DocumentGroundingFilter(id="SAPHelp", data_repository_type="help.sap.com")]

grounding_config = GroundingModule(type=GroundingType.DOCUMENT_GROUNDING_SERVICE.value,
                                   config=DocumentGrounding(input_params=["product"],
                                                            output_param="grounding_response",
                                                            filters=filters
                                                            )
                                  )

config = OrchestrationConfig(template=prompt, llm=llm, grounding=grounding_config)

response = orchestration_service.run(config=config, template_values=[TemplateValue("product", "Generative AI Hub")])
print(response)
print(response.orchestration_result.choices[0].message.content)