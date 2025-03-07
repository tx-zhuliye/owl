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
import base64
import logging
import os
from typing import List, Optional
from urllib.parse import urlparse

import openai
import requests

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool

# logger = logging.getLogger(__name__)
from loguru import logger

from camel.models import ModelFactory
from camel.configs import QwenConfig
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.messages import BaseMessage


class AudioAnalysisToolkit(BaseToolkit):
    r"""A class representing a toolkit for audio operations.

    This class provides methods for processing and understanding audio data.
    """

    def __init__(self, cache_dir: Optional[str] = None, reasoning: bool = False):
        self.cache_dir = 'tmp/'
        if cache_dir:
            self.cache_dir = cache_dir

        # 创建通义千问Omni模型
        self.audio_model = ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_OMNI_TURBO,
            model_config_dict=QwenConfig(
                temperature=0.3, 
                top_p=0.9, 
                stream=False  # 设置为False以避免设置stream_options
            ).as_dict(),
        )
        self.audio_agent = ChatAgent(
            model=self.audio_model,
            output_language="English"
        )
        self.reasoning = reasoning


    def ask_question_about_audio(self, audio_path: str, question: str) -> str:
        r"""Ask any question about the audio and get the answer using
            multimodal model.

        Args:
            audio_path (str): The path to the audio file.
            question (str): The question to ask about the audio.

        Returns:
            str: The answer to the question.
        """

        logger.debug(
            f"Calling ask_question_about_audio method for audio file \
            `{audio_path}` and question `{question}`."
        )

        parsed_url = urlparse(audio_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])
        encoded_string = None

        if is_url:
            # 使用URL直接传递给模型
            audio_url = audio_path
        else:
            # 如果是本地文件，则需要进行base64编码
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            audio_file.close()
            encoded_string = base64.b64encode(audio_data).decode('utf-8')
            # 在实际场景中，我们需要将此base64字符串上传到服务器或CDN，获取URL
            # 这里我们假设已经上传，并获得了URL
            audio_url = f"data:audio/mp3;base64,{encoded_string}"

        file_suffix = os.path.splitext(audio_path)[1]
        file_format = file_suffix[1:]

        if self.reasoning:
            # 使用通义千问的多模态能力
            logger.info("Using reasoning mode with Qwen-Omni model for audio analysis")
            
            msg = BaseMessage.make_user_message(
                role_name="User",
                content=f"请分析这段音频并回答以下问题：{question}"
            )
            
            # 通过OpenAI兼容接口实现
            from camel.messages import OpenAIMessage
            openai_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_url,  # 使用URL或base64
                                "format": file_format,
                            },
                        },
                        {"type": "text", "text": f"请分析这段音频并回答以下问题：{question}"},
                    ],
                },
            ]
            
            # 直接使用OpenAI兼容的客户端
            import os
            from openai import OpenAI
            
            client = OpenAI(
                api_key=os.getenv("QWEN_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            completion = client.chat.completions.create(
                model="qwen-omni-turbo",
                messages=openai_messages,
                modalities=["text"],
                stream=True,
            )
            
            # 处理流式响应
            answer_parts = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    answer_parts.append(chunk.choices[0].delta.content)
            
            return "".join(answer_parts)
        else:
            # 非reasoning模式，使用简单的步骤
            # 假设不需要复杂的处理逻辑
            msg = BaseMessage.make_user_message(
                role_name="User",
                content=f"请分析这段音频并回答问题：{question}"
            )
            
            response = self.audio_agent.step(msg)
            return response.msgs[0].content

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
            in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the
                functions in the toolkit.
        """
        return [FunctionTool(self.ask_question_about_audio)]