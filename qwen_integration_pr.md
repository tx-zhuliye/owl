# 通义千问(Qwen)模型集成PR文档

## 功能概述

本PR为OWL项目添加了对阿里云通义千问(Qwen)模型的全面支持，让OWL能够利用Qwen系列模型的强大能力，特别是其多模态功能。

## 主要改进

1. **模型支持**
   - 添加对通义千问(Qwen)文本模型的支持：`qwen-turbo`、`qwen-plus`、`qwen-max`等
   - 添加对通义千问多模态模型的支持：`qwen-omni-turbo`，支持图像、音频和视频输入

2. **工具集成**
   - 优化`AudioAnalysisToolkit`，使其能够使用通义千问的多模态能力处理音频
   - 优化`VideoAnalysisToolkit`，支持使用通义千问模型进行视频内容分析
   - 修复了工具包中与模态处理相关的问题

3. **配置与环境**
   - 添加通义千问所需的环境变量配置
   - 设置默认模型配置选项，便于用户快速切换

4. **文档与示例**
   - 提供完整的通义千问API调用示例文档
   - 说明OpenAI兼容方式和DashScope方式两种调用方法
   - 包含流式输出、多模态输入等高级用例

## 技术细节

### 修复的问题
- 修复了`ModelPlatformType`和`ModelType`的导入路径问题
- 修复了`QwenConfig`类的导入路径和使用问题
- 解决了`modalities`参数传递问题，确保与通义千问API兼容
- 解决了由于`stream_options`设置导致的验证错误
- 在`token_limit`方法中添加了对`QWEN_OMNI_TURBO`的支持

### 改进的组件
- `camel/toolkits/audio_analysis_toolkit.py`: 支持通义千问模型处理音频
- `camel/toolkits/video_analysis_toolkit.py`: 支持通义千问模型处理视频
- `camel/types/enums.py`: 添加通义千问多模态模型的token限制
- `owl/.env`: 新增通义千问API相关环境变量配置

### 环境变量配置
```
# 通义千问API (https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key)
QWEN_API_KEY=""
DASHSCOPE_API_KEY=""  # OpenAI兼容方式使用同一个密钥

# 默认模型设置
DEFAULT_MODEL_PLATFORM_TYPE="tongyi-qianwen"
DEFAULT_MODEL_TYPE="qwen-turbo"
```

## 使用说明

通过设置环境变量可以轻松切换到通义千问模型：

1. 在`.env`文件中设置`QWEN_API_KEY`和`DASHSCOPE_API_KEY`
2. 将`DEFAULT_MODEL_PLATFORM_TYPE`设置为`"tongyi-qianwen"`
3. 将`DEFAULT_MODEL_TYPE`设置为所需的通义千问模型，如`"qwen-turbo"`

多模态功能使用示例：
```python
# 使用通义千问Omni模型分析音频
audio_tool = AudioAnalysisToolkit()
result = audio_tool.ask_question_about_audio("path/to/audio.mp3", "这段音频说了什么？")
```

## 测试与验证

- 验证了通义千问API的连接和基本功能
- 测试了音频和视频分析工具包的正常工作
- 验证了模型的流式输出功能
- 测试了OpenAI兼容方式调用的稳定性

## 后续工作

- 进一步优化多模态模型的参数配置
- 扩展对更多通义千问模型的支持
- 添加更多使用通义千问的上层应用示例 