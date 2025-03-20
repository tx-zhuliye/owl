# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import os
import sys

from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType

from owl.utils import run_society
from camel.societies import RolePlaying
from camel.logger import set_log_level

import pathlib

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")


def construct_society(question: str) -> RolePlaying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=os.getenv("USER_ROLE_API_MODEL_TYPE", os.getenv("LLM_ROLE_API_MODEL_TYPE", "qwen-max")),
            api_key=os.getenv("USER_ROLE_API_KEY", os.getenv("LLM_ROLE_API_KEY", os.getenv("QWEN_API_KEY", "Your_Key"))),
            url=os.getenv("USER_ROLE_API_BASE_URL", os.getenv("LLM_ROLE_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            model_config_dict={
                "temperature": float(os.getenv("USER_ROLE_API_MODEL_TEMPERATURE", os.getenv("LLM_ROLE_API_MODEL_TEMPERATURE", "0.4"))), 
                "max_tokens": int(os.getenv("USER_ROLE_API_MODEL_MAX_TOKENS", os.getenv("LLM_ROLE_API_MODEL_MAX_TOKENS", "4096")))
            },
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=os.getenv("ASSISTANT_ROLE_API_MODEL_TYPE", os.getenv("LLM_ROLE_API_MODEL_TYPE", "qwen-max")),
            api_key=os.getenv("ASSISTANT_ROLE_API_KEY", os.getenv("LLM_ROLE_API_KEY", os.getenv("QWEN_API_KEY", "Your_Key"))),
            url=os.getenv("ASSISTANT_ROLE_API_BASE_URL", os.getenv("LLM_ROLE_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            model_config_dict={
                "temperature": float(os.getenv("ASSISTANT_ROLE_API_MODEL_TEMPERATURE", os.getenv("LLM_ROLE_API_MODEL_TEMPERATURE", "0.4"))), 
                "max_tokens": int(os.getenv("ASSISTANT_ROLE_API_MODEL_MAX_TOKENS", os.getenv("LLM_ROLE_API_MODEL_MAX_TOKENS", "4096")))
            },
        ),
        "web": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=os.getenv("WEB_ROLE_API_BASE_URL", os.getenv("VLLM_ROLE_API_MODEL_TYPE", "qwen-vl-max")),
            api_key=os.getenv("WEB_ROLE_API_KEY", os.getenv("VLLM_ROLE_API_KEY", os.getenv("QWEN_API_KEY", "Your_Key"))),
            url=os.getenv("USER_ROLE_API_BASE_URL", os.getenv("VLLM_ROLE_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            model_config_dict={
                "temperature": float(os.getenv("WEB_ROLE_API_MODEL_TEMPERATURE", os.getenv("VLLM_ROLE_API_MODEL_TEMPERATURE", "0.4"))), 
                "max_tokens": int(os.getenv("WEB_ROLE_API_MODEL_MAX_TOKENS", os.getenv("VLLM_ROLE_API_MODEL_MAX_TOKENS", "4096")))
            },
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=os.getenv("PLANNING_ROLE_API_MODEL_TYPE", os.getenv("LLM_ROLE_API_MODEL_TYPE", "qwen-max")),
            api_key=os.getenv("PLANNING_ROLE_API_KEY", os.getenv("LLM_ROLE_API_KEY", os.getenv("QWEN_API_KEY", "Your_Key"))),
            url=os.getenv("PLANNING_ROLE_API_BASE_URL", os.getenv("LLM_ROLE_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            model_config_dict={
                "temperature": float(os.getenv("PLANNING_ROLE_API_MODEL_TEMPERATURE", os.getenv("LLM_ROLE_API_MODEL_TEMPERATURE", "0.4"))), 
                "max_tokens": int(os.getenv("PLANNING_ROLE_API_MODEL_MAX_TOKENS", os.getenv("LLM_ROLE_API_MODEL_MAX_TOKENS", "4096")))
            },
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=os.getenv("IMAGE_ROLE_API_MODEL_TYPE", os.getenv("VLLM_ROLE_API_MODEL_TYPE", "qwen-vl-max")),
            api_key=os.getenv("IMAGE_ROLE_API_KEY", os.getenv("VLLM_ROLE_API_KEY", os.getenv("QWEN_API_KEY", "Your_Key"))),
            url=os.getenv("IMAGE_ROLE_API_BASE_URL", os.getenv("VLLM_ROLE_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            model_config_dict={
                "temperature": float(os.getenv("IMAGE_ROLE_API_MODEL_TEMPERATURE", os.getenv("VLLM_ROLE_API_MODEL_TEMPERATURE", "0.4"))), 
                "max_tokens": int(os.getenv("IMAGE_ROLE_API_MODEL_MAX_TOKENS", os.getenv("VLLM_ROLE_API_MODEL_MAX_TOKENS", "4096")))
            },
        ),
    }

    # Configure toolkits
    tools = [
        *BrowserToolkit(
            headless=False,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["web"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,  # Comment this out if you don't have google search
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
    ]

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = RolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    return society


def main(question: str = "Navigate to Amazon.com and identify one product that is attractive to coders. Please provide me with the product name and price. No need to verify your answer."):
    r"""Main function to run the OWL system with an example question.
    Args:
        question (str): The task or question to be addressed by the society.
            If not provided, a default question will be used.
            Defaults to "Navigate to Amazon.com and identify one product that is attractive to coders. Please provide me with the product name and price. No need to verify your answer."
    Returns:
        None
    """
    # Construct and run the society
    society = construct_society(question)
    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")
    # Output the token count
    print(f"\033[94mToken count: {token_count}\033[0m")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
