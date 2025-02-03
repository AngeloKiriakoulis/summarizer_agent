#!/usr/bin/env python
from dotenv import load_dotenv
from typing import Dict
from naptha_sdk.schemas import AgentRunInput, AgentDeployment
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from naptha_sdk.inference import InferenceClient
from summarizer_agent.schemas import InputSchema, SystemPromptSchema
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

load_dotenv()

logger = get_logger(__name__)

# You can create your module as a class or function
class SummarizerAgent:
    def __init__(self, deployment: AgentDeployment):
        self.deployment = deployment
        self.system_prompt = SystemPromptSchema(role=deployment.config.system_prompt["role"])
        self.node = InferenceClient(self.deployment.node)

    async def summarize(self, inputs: InputSchema):
        text = inputs.tool_input_data if isinstance(inputs.tool_input_data, str) else " ".join([msg["content"] for msg in inputs.tool_input_data])
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
        summary_text = " ".join(str(sentence) for sentence in summary)
        logger.info(f"Summary: {summary_text}")
        return [{"role": "assistant", "content": summary_text}]

# Default entrypoint when the module is executed
async def run(module_run: Dict):
    module_run = AgentRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    basic_module = SummarizerAgent(module_run.deployment)
    method = getattr(basic_module, module_run.inputs.tool_name, None)
    return await method(module_run.inputs)

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("agent", "summarizer_agent/configs/deployment.json", node_url = os.getenv("NODE_URL")))

    input_params = {
        "tool_name": "summarize",
        "tool_input_data": "To build your first module manually, you can:Follow the module template README Follow our interactive tutorial Your First Agent Module in the docs. To automatically convert an existing agent application from frameworks like CrewAI, you should: Read the documentation on Naptha Decorators Inspect examples of decorated agents e.g. crewAI-examples To get up to speed with building more complex applications, composed of many modules, check out the tutorials in the Naptha Learn Hub.",
    }

    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = asyncio.run(run(module_run))

    print("Response: ", response)