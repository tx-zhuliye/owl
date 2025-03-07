from typing import Any, Dict, Optional

from camel.configs import (
    ChatGPTConfig, DeepSeekConfig, QwenConfig, MistralConfig,
    GeminiConfig, AnthropicConfig, VLLMConfig, ZhipuAIConfig,
    YiConfig, GroqConfig, CohereConfig, RekaConfig, NvidiaConfig
)
from camel.types import ModelPlatformType

def create_config(model_platform: ModelPlatformType, **kwargs) -> Dict[str, Any]:
    """创建适合指定模型平台的配置。
    
    Args:
        model_platform: 模型平台类型
        **kwargs: 配置参数
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    # 根据平台类型选择合适的配置类
    if model_platform.is_openai:
        config = ChatGPTConfig(**kwargs)
    elif model_platform.is_deepseek:
        config = DeepSeekConfig(**kwargs)
    elif model_platform.is_qwen:
        config = QwenConfig(**kwargs)
    elif model_platform.is_mistral:
        config = MistralConfig(**kwargs)
    elif model_platform.is_gemini:
        config = GeminiConfig(**kwargs)
    elif model_platform.is_anthropic:
        config = AnthropicConfig(**kwargs)
    elif model_platform.is_vllm:
        config = VLLMConfig(**kwargs)
    elif model_platform.is_zhipuai:
        config = ZhipuAIConfig(**kwargs)
    elif model_platform.is_yi:
        config = YiConfig(**kwargs)
    elif model_platform.is_groq:
        config = GroqConfig(**kwargs)
    elif model_platform.is_cohere:
        config = CohereConfig(**kwargs)
    elif model_platform.is_reka:
        config = RekaConfig(**kwargs)
    elif model_platform.is_nvidia:
        config = NvidiaConfig(**kwargs)
    else:
        # 默认使用OpenAI配置
        config = ChatGPTConfig(**kwargs)
    
    return config.as_dict()