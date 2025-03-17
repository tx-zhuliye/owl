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

# To run this file, you need to configure the Qwen API key
# You can obtain your API key from Bailian platform: bailian.console.aliyun.com
# Set it as QWEN_API_KEY="your-api-key" in your .env file or add it to your environment variables

from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.configs import SiliconFlowConfig
from camel.toolkits import (
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.societies import RolePlaying

from owl.utils import run_society, DocumentProcessingToolkit

from camel.logger import set_log_level


import pathlib, sys

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")


def construct_society(question: str) -> RolePlaying:
    """
    Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type="Qwen/QwQ-32B",
            model_config_dict=SiliconFlowConfig(temperature=0.2).as_dict(),
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type="Qwen/QwQ-32B",
            model_config_dict=SiliconFlowConfig(temperature=0.2).as_dict(),
        ),
        "web": ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type="Qwen/Qwen2-VL-72B-Instruct",
            model_config_dict=SiliconFlowConfig(temperature=0.2).as_dict(),
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type="Qwen/QwQ-32B",
            model_config_dict=SiliconFlowConfig(temperature=0.2).as_dict(),
        ),
        "video": ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type="Qwen/Qwen2-VL-72B-Instruct",
            model_config_dict=SiliconFlowConfig(temperature=0.2).as_dict(),
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type="Qwen/Qwen2-VL-72B-Instruct",
            model_config_dict=SiliconFlowConfig(temperature=0.2).as_dict(),
        ),
        "document": ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type="Qwen/Qwen2-VL-72B-Instruct",
            model_config_dict=SiliconFlowConfig(temperature=0.2).as_dict(),
        ),
    }

    # Configure toolkits
    tools = [
        *BrowserToolkit(
            headless=False,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["web"],
            planning_agent_model=models["planning"],
            output_language="Chinese",
        ).get_tools(),
        *VideoAnalysisToolkit(model=models["video"]).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,  # Comment this out if you don't have google search
        SearchToolkit().search_wiki,
        SearchToolkit().search_baidu,
        *ExcelToolkit().get_tools(),
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
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
        output_language="Chinese",
    )

    return society


def main():
    r"""Main function to run the OWL system with an example question."""
    # Example research question
    if len(sys.argv) > 1:
        question = str(sys.argv[1])
        
    else:
        question = "浏览百度告诉我未来一周北京天气！并帮我制定北京的旅游计划！"
    # Construct and run the society
    society = construct_society(question)
    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")
    


if __name__ == "__main__":
    main()
