from camel.models import ModelFactory
from camel.toolkits import *
from camel.types import ModelPlatformType, ModelType
from camel.configs import QwenConfig

from typing import List, Dict
from dotenv import load_dotenv
from retry import retry
from loguru import logger

from utils import OwlRolePlaying, process_tools, run_society
import os


load_dotenv()


def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct the society based on the question."""

    user_role_name = "user"
    assistant_role_name = "assistant"
    
    user_model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_PLUS,
        model_config_dict=QwenConfig(temperature=0.3, top_p=0.9).as_dict(),
    )

    assistant_model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_PLUS,
        model_config_dict=QwenConfig(temperature=0.3, top_p=0.9).as_dict(),
    )
 
    
    user_tools = []
    assistant_tools = [
        "WebToolkit",
        'DocumentProcessingToolkit', 
        'VideoAnalysisToolkit', 
        'CodeExecutionToolkit', 
        'ImageAnalysisToolkit', 
        'AudioAnalysisToolkit', 
        "SearchToolkit",
        "ExcelToolkit",
        ]

    user_role_name = 'user'
    user_agent_kwargs = {
        'model': user_model,
        'tools': process_tools(user_tools),
    }
    assistant_role_name = 'assistant'
    assistant_agent_kwargs = {
        'model': assistant_model,
        'tools': process_tools(assistant_tools),
    }
    
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
question = "我需要创建一个AI日程管理助手的微信小程序，请你作为产品经理规划工作流程和分工，制定相关的开发计划和内容。然后，你作为UI设计师，设计小程序的UI界面。最后，你作为开发工程师，编写代码实现小程序的功能。"

society = construct_society(question)
answer, chat_history, token_count = run_society(society)

logger.success(f"Answer: {answer}")





