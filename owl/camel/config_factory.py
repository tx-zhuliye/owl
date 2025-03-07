from typing import Any, Dict, Optional

from camel.configs import (
    ChatGPTConfig, DeepSeekConfig, QwenConfig, MistralConfig,
    GeminiConfig, AnthropicConfig, VLLMConfig, ZhipuAIConfig,
    YiConfig, GroqConfig, CohereConfig, RekaConfig, NvidiaConfig
)
from camel.types import ModelPlatformType

def create_config(model_platform: ModelPlatformType, **kwargs) -> Dict[str, Any]:
    """Create configuration suitable for the specified model platform.
    
    Args:
        model_platform: Type of model platform
        **kwargs: Configuration parameters
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Select appropriate config class based on platform type
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
        # Use OpenAI config as default
        config = ChatGPTConfig(**kwargs)
    
    return config.as_dict()
