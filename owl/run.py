from dotenv import load_dotenv
load_dotenv()

from camel.models import ModelFactory
from camel.toolkits import *
from camel.types import ModelPlatformType, ModelType
#from camel.configs import ChatGPTConfig
from camel.configs import create_config

from typing import List, Dict

from retry import retry
from loguru import logger

from utils import OwlRolePlaying, run_society
import os




def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct the society based on the question."""

    user_role_name = "user"
    assistant_role_name = "assistant"
    
    user_model_config_params = {
        "temperature": 0, 
        "top_p": 1
    }
    assistant_model_config_params = {
        "temperature": 0, 
        "top_p": 1
    }
    
    user_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
        model_config_dict=create_config(ModelPlatformType.DEFAULT, **user_model_config_params), # [Optional] the config for model
    )

    assistant_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
        model_config_dict=create_config(ModelPlatformType.DEFAULT, **assistant_model_config_params), # [Optional] the config for model
    )

    tools_list = [
        *WebToolkit(
            headless=False, 
            web_agent_model=assistant_model, 
            planning_agent_model=assistant_model
        ).get_tools(),
        *DocumentProcessingToolkit().get_tools(),
        *VideoAnalysisToolkit().get_tools(),  # This requires OpenAI and Qwen Key
        *CodeExecutionToolkit().get_tools(),
        *ImageAnalysisToolkit(model=assistant_model).get_tools(),
        *AudioAnalysisToolkit().get_tools(),  # This requires OpenAI Key
        *SearchToolkit(model=assistant_model).get_tools(),
        *ExcelToolkit().get_tools()
    ]

    user_role_name = 'user'
    user_agent_kwargs = dict(model=user_model)
    assistant_role_name = 'assistant'
    assistant_agent_kwargs = dict(model=assistant_model,
    tools=tools_list)
    
    task_kwargs = {
        'task_prompt': question,
        'with_task_specify': False,
    }

    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name=user_role_name,
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name=assistant_role_name,
        assistant_agent_kwargs=assistant_agent_kwargs,
    )
    
    return society


# Example case
question = "What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper `Can Hiccup Supply Enough Fish to Maintain a Dragon’s Diet?` "

society = construct_society(question)
answer, chat_history, token_count = run_society(society)

logger.success(f"Answer: {answer}")





