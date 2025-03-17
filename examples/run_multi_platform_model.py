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
# run_ollama.py by tj-scripts（https://github.com/tj-scripts）
import os

from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.models.base_model import BaseModelBackend
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

# Create models for different roles
def create_role_playing_model(
    role_name: str = "USER",
    default_role_name: str = "LLM",
    ) -> BaseModelBackend:
    r"""Creates an instance of `BaseModelBackend` of the specified role.
    Args:
        role_name (str): The role of the model.
        default_role_name (str): The default role of the model.
    Returns:
            BaseModelBackend: The initialized backend.
    Raises:
        ValueError: If the role name is invalid.
        ValueError: If there is no backend for the model.
    """

    # Check if role name is valid
    if role_name not in ["USER", "ASSISTANT", "WEB", "PLANNING", "IMAGE"]:
        raise ValueError(f"Invalid role name: {role_name}")
    
    # Check if default role name is valid
    if default_role_name not in ["LLM", "VLLM"]:
        raise ValueError(f"Invalid default role name: {default_role_name}")

    # Get model type from environment variable
    model_type=os.getenv(f"{role_name}_ROLE_API_MODEL_TYPE", os.getenv(f"{default_role_name}_ROLE_API_MODEL_TYPE"))
    # Get API key from environment variable
    api_key=os.getenv(f"{role_name}_ROLE_API_KEY", os.getenv(f"{default_role_name}_ROLE_API_KEY"))
    # Get URL from environment variable
    url=os.getenv(f"{role_name}_ROLE_API_BASE_URL", os.getenv(f"{default_role_name}_ROLE_API_BASE_URL"))

    try:
        # Get temperature from environment variable
        temperature_str = os.getenv(f"{role_name}_ROLE_API_MODEL_TEMPERATURE", os.getenv(f"{default_role_name}_ROLE_API_MODEL_TEMPERATURE"))
        temperature = float(temperature_str) if temperature_str else 0.0
    except ValueError:
        temperature = 0.0
    
    try:
        # Get max_tokens from environment variable
        max_tokens_str = os.getenv(f"{role_name}_ROLE_API_MODEL_MAX_TOKENS", os.getenv(f"{default_role_name}_ROLE_API_MODEL_MAX_TOKENS"))
        max_tokens = int(max_tokens_str) if max_tokens_str else 0
    except ValueError:
        max_tokens = 0

    # Check if max_tokens is valid
    if max_tokens < 1:
        model_config_dict = {"temperature": temperature}
    else:
        model_config_dict = {"temperature": temperature, "max_tokens": max_tokens}

    # Create model for different components
    return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            api_key=api_key,
            url=url,
            model_config_dict=model_config_dict,
        )

def construct_society(question: str) -> RolePlaying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components
    models = {
        "user": create_role_playing_model("USER"),
        "assistant": create_role_playing_model("ASSISTANT"),
        "web": create_role_playing_model("WEB", "VLLM"),
        "planning": create_role_playing_model("PLANNING"),
        "image": create_role_playing_model("IMAGE", "VLLM"),
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
        SearchToolkit().search_baidu,
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


def main():
    r"""Main function to run the OWL system with an example question."""
    # Example research question
    question = "Navigate to Amazon.com and identify one product that is attractive to coders. Please provide me with the product name and price. No need to verify your answer."

    # Construct and run the society
    society = construct_society(question)
    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    main()
