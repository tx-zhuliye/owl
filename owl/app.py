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
import os
import sys
import gradio as gr
import subprocess
import threading
import time
from datetime import datetime
import queue
from pathlib import Path
import json
import signal
import dotenv

# 设置日志队列
log_queue: queue.Queue[str] = queue.Queue()

# 当前运行的进程
current_process = None
process_lock = threading.Lock()

# 脚本选项
SCRIPTS = {
    "Qwen Mini (中文)": "run_qwen_mini_zh.py",
    "Qwen （中文）": "run_qwen_zh.py",
    "Mini": "run_mini.py",
    "DeepSeek （中文）": "run_deepseek_zh.py",
    "Default": "run.py",
    "GAIA Roleplaying": "run_gaia_roleplaying.py",
    "OpenAI Compatible": "run_openai_compatiable_model.py",
}

# 脚本描述
SCRIPT_DESCRIPTIONS = {
    "Qwen Mini (中文)": "使用阿里云Qwen模型的中文版本，适合中文问答和任务",
    "Qwen （中文）": "使用阿里云Qwen模型，支持多种工具和功能",
    "Mini": "轻量级版本，使用OpenAI GPT-4o模型",
    "DeepSeek （中文）": "使用DeepSeek模型，适合非多模态任务",
    "Default": "默认OWL实现，使用OpenAI GPT-4o模型和全套工具",
    "GAIA Roleplaying": "GAIA基准测试实现，用于评估模型能力",
    "OpenAI Compatible": "使用兼容OpenAI API的第三方模型，支持自定义API端点",
}

# 环境变量分组
ENV_GROUPS = {
    "模型API": [
        {
            "name": "OPENAI_API_KEY",
            "label": "OpenAI API密钥",
            "type": "password",
            "required": False,
            "help": "OpenAI API密钥，用于访问GPT模型。获取方式：https://platform.openai.com/api-keys",
        },
        {
            "name": "OPENAI_API_BASE_URL",
            "label": "OpenAI API基础URL",
            "type": "text",
            "required": False,
            "help": "OpenAI API的基础URL，可选。如果使用代理或自定义端点，请设置此项。",
        },
        {
            "name": "QWEN_API_KEY",
            "label": "阿里云Qwen API密钥",
            "type": "password",
            "required": False,
            "help": "阿里云Qwen API密钥，用于访问Qwen模型。获取方式：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key",
        },
        {
            "name": "DEEPSEEK_API_KEY",
            "label": "DeepSeek API密钥",
            "type": "password",
            "required": False,
            "help": "DeepSeek API密钥，用于访问DeepSeek模型。获取方式：https://platform.deepseek.com/api_keys",
        },
    ],
    "搜索工具": [
        {
            "name": "GOOGLE_API_KEY",
            "label": "Google API密钥",
            "type": "password",
            "required": False,
            "help": "Google搜索API密钥，用于网络搜索功能。获取方式：https://developers.google.com/custom-search/v1/overview",
        },
        {
            "name": "SEARCH_ENGINE_ID",
            "label": "搜索引擎ID",
            "type": "text",
            "required": False,
            "help": "Google自定义搜索引擎ID，与Google API密钥配合使用。获取方式：https://developers.google.com/custom-search/v1/overview",
        },
    ],
    "其他工具": [
        {
            "name": "HF_TOKEN",
            "label": "Hugging Face令牌",
            "type": "password",
            "required": False,
            "help": "Hugging Face API令牌，用于访问Hugging Face模型和数据集。获取方式：https://huggingface.co/join",
        },
        {
            "name": "CHUNKR_API_KEY",
            "label": "Chunkr API密钥",
            "type": "password",
            "required": False,
            "help": "Chunkr API密钥，用于文档处理功能。获取方式：https://chunkr.ai/",
        },
        {
            "name": "FIRECRAWL_API_KEY",
            "label": "Firecrawl API密钥",
            "type": "password",
            "required": False,
            "help": "Firecrawl API密钥，用于网页爬取功能。获取方式：https://www.firecrawl.dev/",
        },
    ],
    "自定义环境变量": [],  # 用户自定义的环境变量将存储在这里
}


def get_script_info(script_name):
    """获取脚本的详细信息"""
    return SCRIPT_DESCRIPTIONS.get(script_name, "无描述信息")


def load_env_vars():
    """加载环境变量"""
    env_vars = {}
    # 尝试从.env文件加载
    dotenv.load_dotenv()

    # 获取所有环境变量
    for group in ENV_GROUPS.values():
        for var in group:
            env_vars[var["name"]] = os.environ.get(var["name"], "")

    # 加载.env文件中可能存在的其他环境变量
    if Path(".env").exists():
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")

                    # 检查是否是已知的环境变量
                    known_var = False
                    for group in ENV_GROUPS.values():
                        if any(var["name"] == key for var in group):
                            known_var = True
                            break

                    # 如果不是已知的环境变量，添加到自定义环境变量组
                    if not known_var and key not in env_vars:
                        ENV_GROUPS["自定义环境变量"].append(
                            {
                                "name": key,
                                "label": key,
                                "type": "text",
                                "required": False,
                                "help": "用户自定义环境变量",
                            }
                        )
                        env_vars[key] = value

    return env_vars


def save_env_vars(env_vars):
    """保存环境变量到.env文件"""
    # 读取现有的.env文件内容
    env_path = Path(".env")
    existing_content = {}

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    existing_content[key.strip()] = value.strip()

    # 更新环境变量
    for key, value in env_vars.items():
        if value:  # 只保存非空值
            # 确保值是字符串形式，并用引号包裹
            value = str(value)  # 确保值是字符串
            if not (value.startswith('"') and value.endswith('"')) and not (
                value.startswith("'") and value.endswith("'")
            ):
                value = f'"{value}"'
            existing_content[key] = value
            # 同时更新当前进程的环境变量
            os.environ[key] = value.strip("\"'")

    # 写入.env文件
    with open(env_path, "w", encoding="utf-8") as f:
        for key, value in existing_content.items():
            f.write(f"{key}={value}\n")

    return "✅ 环境变量已保存"


def add_custom_env_var(name, value, var_type):
    """添加自定义环境变量"""
    if not name:
        return "❌ 环境变量名不能为空", None

    # 检查是否已存在同名环境变量
    for group in ENV_GROUPS.values():
        if any(var["name"] == name for var in group):
            return f"❌ 环境变量 {name} 已存在", None

    # 添加到自定义环境变量组
    ENV_GROUPS["自定义环境变量"].append(
        {
            "name": name,
            "label": name,
            "type": var_type,
            "required": False,
            "help": "用户自定义环境变量",
        }
    )

    # 保存环境变量
    env_vars = {name: value}
    save_env_vars(env_vars)

    # 返回成功消息和更新后的环境变量组
    return f"✅ 已添加环境变量 {name}", ENV_GROUPS["自定义环境变量"]


def terminate_process():
    """终止当前运行的进程"""
    global current_process

    with process_lock:
        if current_process is not None and current_process.poll() is None:
            try:
                # 在Windows上使用taskkill强制终止进程树
                if os.name == "nt":
                    # 获取进程ID
                    pid = current_process.pid
                    # 使用taskkill命令终止进程及其子进程
                    subprocess.run(f"taskkill /F /T /PID {pid}", shell=True)
                else:
                    # 在Unix上使用SIGTERM和SIGKILL
                    current_process.terminate()
                    try:
                        current_process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        current_process.kill()

                # 等待进程终止
                try:
                    current_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass  # 已经尝试强制终止，忽略超时

                log_queue.put("进程已终止\n")
                return "✅ 进程已终止"
            except Exception as e:
                log_queue.put(f"终止进程时出错: {str(e)}\n")
                return f"❌ 终止进程时出错: {str(e)}"
        else:
            return "❌ 没有正在运行的进程"


def run_script(script_dropdown, question, progress=gr.Progress()):
    """运行选定的脚本并返回输出"""
    global current_process

    script_name = SCRIPTS.get(script_dropdown)
    if not script_name:
        return "❌ 无效的脚本选择", "", "", "", None

    if not question.strip():
        return "请输入问题！", "", "", "", None

    # 清空日志队列
    while not log_queue.empty():
        log_queue.get()

    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name.replace('.py', '')}_{timestamp}.log"

    # 构建命令
    cmd = [
        sys.executable,
        os.path.join("owl", "script_adapter.py"),
        os.path.join("owl", script_name),
    ]

    # 创建环境变量副本并添加问题
    env = os.environ.copy()
    # 确保问题是字符串类型
    if not isinstance(question, str):
        question = str(question)
    # 保留换行符，但确保是有效的字符串
    env["OWL_QUESTION"] = question

    # 启动进程
    with process_lock:
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

    # 创建线程来读取输出
    def read_output():
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                for line in iter(current_process.stdout.readline, ""):
                    if line:
                        # 写入日志文件
                        f.write(line)
                        f.flush()
                        # 添加到队列
                        log_queue.put(line)
        except Exception as e:
            log_queue.put(f"读取输出时出错: {str(e)}\n")

    # 启动读取线程
    threading.Thread(target=read_output, daemon=True).start()

    # 收集日志
    logs = []
    progress(0, desc="正在运行...")

    # 等待进程完成或超时
    start_time = time.time()
    timeout = 1800  # 30分钟超时

    while current_process.poll() is None:
        # 检查是否超时
        if time.time() - start_time > timeout:
            with process_lock:
                if current_process.poll() is None:
                    if os.name == "nt":
                        current_process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        current_process.terminate()
                    log_queue.put("执行超时，已终止进程\n")
            break

        # 从队列获取日志
        while not log_queue.empty():
            log = log_queue.get()
            logs.append(log)

        # 更新进度
        elapsed = time.time() - start_time
        progress(min(elapsed / 300, 0.99), desc="正在运行...")

        # 短暂休眠以减少CPU使用
        time.sleep(0.1)

        # 每秒更新一次日志显示
        yield (
            status_message(current_process),
            extract_answer(logs),
            "".join(logs),
            str(log_file),
            None,
        )

    # 获取剩余日志
    while not log_queue.empty():
        logs.append(log_queue.get())

    # 提取聊天历史（如果有）
    chat_history = extract_chat_history(logs)

    # 返回最终状态和日志
    return (
        status_message(current_process),
        extract_answer(logs),
        "".join(logs),
        str(log_file),
        chat_history,
    )


def status_message(process):
    """根据进程状态返回状态消息"""
    if process.poll() is None:
        return "⏳ 正在运行..."
    elif process.returncode == 0:
        return "✅ 执行成功"
    else:
        return f"❌ 执行失败 (返回码: {process.returncode})"


def extract_answer(logs):
    """从日志中提取答案"""
    answer = ""
    for log in logs:
        if "Answer:" in log:
            answer = log.split("Answer:", 1)[1].strip()
            break
    return answer


def extract_chat_history(logs):
    """尝试从日志中提取聊天历史"""
    try:
        chat_json_str = ""
        capture_json = False

        for log in logs:
            if "chat_history" in log:
                # 开始捕获JSON
                start_idx = log.find("[")
                if start_idx != -1:
                    capture_json = True
                    chat_json_str = log[start_idx:]
            elif capture_json:
                # 继续捕获JSON直到找到匹配的结束括号
                chat_json_str += log
                if "]" in log:
                    # 找到结束括号，尝试解析JSON
                    end_idx = chat_json_str.rfind("]") + 1
                    if end_idx > 0:
                        try:
                            # 清理可能的额外文本
                            json_str = chat_json_str[:end_idx].strip()
                            chat_data = json.loads(json_str)

                            # 格式化为Gradio聊天组件可用的格式
                            formatted_chat = []
                            for msg in chat_data:
                                if "role" in msg and "content" in msg:
                                    role = "用户" if msg["role"] == "user" else "助手"
                                    formatted_chat.append([role, msg["content"]])
                            return formatted_chat
                        except json.JSONDecodeError:
                            # 如果解析失败，继续捕获
                            pass
                        except Exception:
                            # 其他错误，停止捕获
                            capture_json = False
    except Exception:
        pass
    return None


def create_ui():
    """创建Gradio界面"""
    # 加载环境变量
    env_vars = load_env_vars()

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
        gr.Markdown(
            """
            # 🦉 OWL 智能助手运行平台
            
            选择一个模型并输入您的问题，系统将运行相应的脚本并显示结果。
            """
        )

        with gr.Tabs():
            with gr.TabItem("运行模式"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 确保默认值是SCRIPTS中存在的键
                        default_script = list(SCRIPTS.keys())[0] if SCRIPTS else None
                        script_dropdown = gr.Dropdown(
                            choices=list(SCRIPTS.keys()),
                            value=default_script,
                            label="选择模式",
                        )

                        script_info = gr.Textbox(
                            value=get_script_info(default_script)
                            if default_script
                            else "",
                            label="模型描述",
                            interactive=False,
                        )

                        script_dropdown.change(
                            fn=lambda x: get_script_info(x),
                            inputs=script_dropdown,
                            outputs=script_info,
                        )

                        question_input = gr.Textbox(
                            lines=8,
                            placeholder="请输入您的问题...",
                            label="问题",
                            elem_id="question_input",
                            show_copy_button=True,
                        )

                        gr.Markdown(
                            """
                            > **注意**: 您输入的问题将替换脚本中的默认问题。系统会自动处理问题的替换，确保您的问题被正确使用。
                            > 支持多行输入，换行将被保留。
                            """
                        )

                        with gr.Row():
                            run_button = gr.Button("运行", variant="primary")
                            stop_button = gr.Button("终止", variant="stop")

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("结果"):
                                status_output = gr.Textbox(label="状态")
                                answer_output = gr.Textbox(label="回答", lines=10)
                                log_file_output = gr.Textbox(label="日志文件路径")

                            with gr.TabItem("运行日志"):
                                log_output = gr.Textbox(label="完整日志", lines=25)

                            with gr.TabItem("聊天历史"):
                                chat_output = gr.Chatbot(label="对话历史")

                # 示例问题
                examples = [
                    [
                        "Qwen Mini (中文)",
                        "浏览亚马逊并找出一款对程序员有吸引力的产品。请提供产品名称和价格",
                    ],
                    [
                        "DeepSeek （中文）",
                        "请分析GitHub上CAMEL-AI项目的最新统计数据。找出该项目的星标数量、贡献者数量和最近的活跃度。然后，创建一个简单的Excel表格来展示这些数据，并生成一个柱状图来可视化这些指标。最后，总结CAMEL项目的受欢迎程度和发展趋势。",
                    ],
                    [
                        "Default",
                        "Navigate to Amazon.com and identify one product that is attractive to coders. Please provide me with the product name and price. No need to verify your answer.",
                    ],
                ]

                gr.Examples(examples=examples, inputs=[script_dropdown, question_input])

            with gr.TabItem("环境变量配置"):
                env_inputs = {}
                save_status = gr.Textbox(label="保存状态", interactive=False)

                # 添加自定义环境变量部分
                with gr.Accordion("添加自定义环境变量", open=True):
                    with gr.Row():
                        new_var_name = gr.Textbox(
                            label="环境变量名", placeholder="例如：MY_CUSTOM_API_KEY"
                        )
                        new_var_value = gr.Textbox(
                            label="环境变量值", placeholder="输入值"
                        )
                        new_var_type = gr.Dropdown(
                            choices=["text", "password"], value="text", label="类型"
                        )

                    add_var_button = gr.Button("添加环境变量", variant="primary")
                    add_var_status = gr.Textbox(label="添加状态", interactive=False)

                    # 自定义环境变量列表
                    custom_vars_list = gr.JSON(
                        value=ENV_GROUPS["自定义环境变量"],
                        label="已添加的自定义环境变量",
                        visible=len(ENV_GROUPS["自定义环境变量"]) > 0,
                    )

                    # 添加环境变量按钮点击事件
                    add_var_button.click(
                        fn=add_custom_env_var,
                        inputs=[new_var_name, new_var_value, new_var_type],
                        outputs=[add_var_status, custom_vars_list],
                    )

                # 现有环境变量配置
                for group_name, vars in ENV_GROUPS.items():
                    if (
                        group_name != "自定义环境变量" or len(vars) > 0
                    ):  # 只显示非空的自定义环境变量组
                        with gr.Accordion(
                            group_name, open=(group_name != "自定义环境变量")
                        ):
                            for var in vars:
                                # 添加帮助信息
                                gr.Markdown(f"**{var['help']}**")

                                if var["type"] == "password":
                                    env_inputs[var["name"]] = gr.Textbox(
                                        value=env_vars.get(var["name"], ""),
                                        label=var["label"],
                                        placeholder=f"请输入{var['label']}",
                                        type="password",
                                    )
                                else:
                                    env_inputs[var["name"]] = gr.Textbox(
                                        value=env_vars.get(var["name"], ""),
                                        label=var["label"],
                                        placeholder=f"请输入{var['label']}",
                                    )

                save_button = gr.Button("保存环境变量", variant="primary")

                # 保存环境变量
                save_inputs = [
                    env_inputs[var_name]
                    for group in ENV_GROUPS.values()
                    for var in group
                    for var_name in [var["name"]]
                    if var_name in env_inputs
                ]
                save_button.click(
                    fn=lambda *values: save_env_vars(
                        dict(
                            zip(
                                [
                                    var["name"]
                                    for group in ENV_GROUPS.values()
                                    for var in group
                                    if var["name"] in env_inputs
                                ],
                                values,
                            )
                        )
                    ),
                    inputs=save_inputs,
                    outputs=save_status,
                )

        # 运行脚本
        run_button.click(
            fn=run_script,
            inputs=[script_dropdown, question_input],
            outputs=[
                status_output,
                answer_output,
                log_output,
                log_file_output,
                chat_output,
            ],
            show_progress=True,
        )

        # 终止运行
        stop_button.click(fn=terminate_process, inputs=[], outputs=[status_output])

        # 添加页脚
        gr.Markdown(
            """
            ### 📝 使用说明
            
            - 选择一个模型并输入您的问题
            - 点击"运行"按钮开始执行
            - 如需终止运行，点击"终止"按钮
            - 在"结果"标签页查看执行状态和回答
            - 在"运行日志"标签页查看完整日志
            - 在"聊天历史"标签页查看对话历史（如果有）
            - 在"环境变量配置"标签页配置API密钥和其他环境变量
            - 您可以添加自定义环境变量，满足特殊需求
            
            ### ⚠️ 注意事项
            
            - 运行某些模型可能需要API密钥，请确保在"环境变量配置"标签页中设置了相应的环境变量
            - 某些脚本可能需要较长时间运行，请耐心等待
            - 如果运行超过30分钟，进程将自动终止
            - 您输入的问题将替换脚本中的默认问题，确保问题与所选模型兼容
            """
        )

    return app


if __name__ == "__main__":
    # 创建并启动应用
    app = create_ui()
    app.queue().launch(share=True)
