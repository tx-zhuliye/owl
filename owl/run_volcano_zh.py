#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 Volcano Engine API 运行 Owl
"""

import os
import json
import logging
import requests
import time
import uuid
from typing import Dict, List, Any, Optional, Union, cast
from dotenv import load_dotenv

from camel.types import ModelPlatformType, ModelType
from camel.models import BaseModelBackend
from camel.models.openai_compatible_model import OpenAICompatibleModel
from camel.messages import BaseMessage, OpenAIMessage
from camel.toolkits import WebToolkit, SearchToolkit

from utils import OwlRolePlaying, run_society

from camel.logger import set_log_level, get_logger

set_log_level("DEBUG")
logger = get_logger("camel.__main__")

# 加载环境变量（获取volcano API密钥）
load_dotenv()

class VolcanoEngineModel(OpenAICompatibleModel):
    """自定义 Volcano Engine 模型类，继承 OpenAICompatibleModel """
    
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter = None,
    ) -> None:
        super().__init__(model_type, model_config_dict, api_key, url, token_counter)
        logger.debug(f"Initialized VolcanoEngineModel with URL: {self.url} and model: {model_type}")
    
    def _run(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        # 使用 requests 库直接调用 volcano API
        try:
            config = self.model_config_dict
            api_messages = self._parse_messages(messages)
            
            data = {
                "model": self.model_type,
                "messages": api_messages,
                "temperature": config.get("temperature", 0),
                "max_tokens": config.get("max_tokens", 4096),
                "stream": False,  # 禁用流式处理！
            }
            
            if response_format:
                data["response_format"] = response_format
                
            if tools:
                data["tools"] = tools
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            logger.debug(f"Sending request to Volcano Engine API: {json.dumps(data, ensure_ascii=False)}")
            
            response = requests.post(
                self.url,
                headers=headers,
                data=json.dumps(data)
            )
            
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Volcano Engine API response: {json.dumps(result, ensure_ascii=False)}")
            
            # 将响应转换为与 OpenAI 兼容的格式
            return self._convert_to_openai_format(result)
            
        except Exception as e:
            logger.error(f"Error calling Volcano Engine API: {e}")
            return self._create_error_response(str(e))
    
    def _parse_messages(self, messages):
        """将 CAMEL 消息格式转换为 Volcano Engine API 格式"""
        api_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                api_messages.append(msg)
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        return api_messages
    
    def _convert_to_openai_format(self, volcano_response):
        """将 Volcano Engine API 响应转换为与 OpenAI 兼容的格式"""
        # 确保响应包含 choices
        if "choices" not in volcano_response or not volcano_response["choices"]:
            raise ValueError("API response does not contain choices")
        
        # 创建与 OpenAI 兼容的响应对象
        openai_response = {
            "id": volcano_response.get("id", f"volcano-{uuid.uuid4()}"),
            "object": "chat.completion",
            "created": volcano_response.get("created", int(time.time())),
            "model": str(self.model_type),
            "choices": [],
            "usage": volcano_response.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
        }
        
        # 转换 choices
        for i, choice in enumerate(volcano_response["choices"]):
            openai_choice = {
                "index": choice.get("index", i),
                "message": choice.get("message", {}),
                "finish_reason": choice.get("finish_reason", "stop")
            }
            openai_response["choices"].append(openai_choice)
        
        from camel.types import ChatCompletion
        return ChatCompletion(**openai_response)
    
    def _create_error_response(self, error_message):
        # 错误响应
        error_response = {
            "id": f"volcano-error-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(self.model_type),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Error: {error_message}"
                    },
                    "finish_reason": "error"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        from camel.types import ChatCompletion
        return ChatCompletion(**error_response)


def construct_society(question: str) -> OwlRolePlaying:
    user_role_name = "user"
    assistant_role_name = "assistant"

    api_key = os.getenv("VOLCANO_API_KEY")
    if not api_key:
        raise ValueError("VOLCANO_API_KEY environment variable is not set")
    
    # 注意API url和模型名称正确
    user_model = VolcanoEngineModel(
        model_type="deepseek-r1-250120",  
        api_key=api_key,
        url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",  
        model_config_dict={"temperature": 0, "max_tokens": 4096}
    )

    assistant_model = VolcanoEngineModel(
        model_type="deepseek-r1-250120",  
        api_key=api_key,
        url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",  
        model_config_dict={"temperature": 0, "max_tokens": 4096}
    )

    planning_model = VolcanoEngineModel(
        model_type="deepseek-r1-250120",  
        api_key=api_key,
        url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",  
        model_config_dict={"temperature": 0, "max_tokens": 4096}
    )

    web_model = VolcanoEngineModel(
        model_type="deepseek-r1-250120",  
        api_key=api_key,
        url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",  
        model_config_dict={"temperature": 0, "max_tokens": 4096}
    )

    tools_list = [
        *WebToolkit(
            headless=False,
            web_agent_model=web_model,
            planning_agent_model=planning_model,
            output_language="zh",
        ).get_tools(),
        SearchToolkit().search_duckduckgo,
    ]

    society = OwlRolePlaying(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task_prompt=question,
        with_task_specify=False,
        assistant_agent_kwargs={
            "model": assistant_model,
        },
        user_agent_kwargs={
            "model": user_model,
            "tools": tools_list,
        },
        output_language="zh",
    )

    return society


if __name__ == "__main__":
    try:
        question = "浏览https://www.volcengine.com/并总结一下这个网站的内容" # test

        society = construct_society(question)

        answer, chat_history, token_count = run_society(society)

        print("\n=== Final Answer ===")
        print(answer)
        print("\n=== Token Count ===")
        print(token_count)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n请检查以下可能的问题：")
        print("1. 确保您的 Volcano Engine API 密钥正确")
        print("2. 确保您有权限访问 deepseek-r1-250120 模型")