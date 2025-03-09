from dotenv import load_dotenv
load_dotenv()

from camel.models import ModelFactory
from camel.toolkits import WebToolkit,SearchToolkit,FunctionTool
from camel.types import ModelPlatformType,ModelType

from loguru import logger

from utils import OwlRolePlaying, run_society
import os


model_scope_api_key = os.getenv("MODELSCOPE_API_KEY")

def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct the society based on the question."""

    user_role_name = "user"
    assistant_role_name = "assistant"
    
    user_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type=ModelType.QWEN_MAX,
             model_config_dict={"temperature": 0},
         )

    assistant_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type=ModelType.QWEN_MAX,
             model_config_dict={"temperature": 0},
         )

    search_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type=ModelType.QWEN_MAX,
             model_config_dict={"temperature": 0},
         )

    planning_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type=ModelType.QWEN_MAX,
             model_config_dict={"temperature": 0},
         )

    web_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type=ModelType.QWEN_VL_MAX,
             model_config_dict={"temperature": 0},
         )

    tools_list = [
        *WebToolkit(
            headless=False, 
            web_agent_model=web_model, 
            planning_agent_model=planning_model,
            output_language='中文'
        ).get_tools(),
        FunctionTool(SearchToolkit(model=search_model).search_duckduckgo),
    ]

    user_role_name = 'user'
    user_agent_kwargs = dict(model=user_model)
    assistant_role_name = 'assistant'
    assistant_agent_kwargs = dict(model=assistant_model,
    tools=tools_list)
    
    task_kwargs = {
        'task_prompt': question,
        'with_task_specify': False,
        'output_language': '中文',
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
question = "打开小红书上浏览推荐栏目下的前三个笔记内容，之后给我一个总结报告"

society = construct_society(question)
answer, chat_history, token_count = run_society(society)

logger.success(f"Answer: {answer}")
