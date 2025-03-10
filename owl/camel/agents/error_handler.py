from camel.societies import RolePlaying
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.utils import print_text_animated
from colorama import Fore
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(dotenv_path='.env')

# 创建模型实例
model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=os.getenv("MODEL_TYPE"),
    url=os.getenv("OPENAI_API_BASE_URL"),
    api_key=os.getenv("QWEN_API_KEY")
)

class TaskTreeNode:
    """
    任务树节点，用于管理任务和子任务。
    """
    def __init__(self, task_name, task_description, parent=None):
        self.task_name = task_name
        self.task_description = task_description
        self.parent = parent
        self.children = []
        self.status = "pending"  # pending, in_progress, completed, failed

    def add_child(self, child_node):
        self.children.append(child_node)

    def set_status(self, status):
        self.status = status

    def __repr__(self):
        return f"TaskTreeNode(name={self.task_name}, status={self.status})"


class TaskManager:
    """
    任务管理器，用于创建和管理任务树。
    """
    def __init__(self, root_task_name, root_task_description):
        self.root = TaskTreeNode(root_task_name, root_task_description)

    def add_task(self, task_name, task_description, parent_node=None):
        new_task = TaskTreeNode(task_name, task_description, parent_node)
        if parent_node:
            parent_node.add_child(new_task)
        else:
            self.root.add_child(new_task)
        return new_task

    def get_task_status(self, task_node):
        return task_node.status

    def update_task_status(self, task_node, status):
        task_node.set_status(status)

    def __repr__(self):
        return f"TaskManager(root={self.root})"


class ErrorHandlingAgent(ChatAgent):
    def __init__(self, model, name, system_message, task_manager=None):
        super().__init__(system_message=system_message, model=model)
        self.error_plan = []
        self.retry_count = 0
        self.max_retries = 3  # 最大重试次数
        self.task_manager = task_manager

    def handle_error(self, error_message, original_step, task_node=None):
        """
        处理错误并生成解决计划。
        """
        self.error_plan = self.generate_error_plan(error_message, original_step, task_node =None)
        print(Fore.RED + f"错误发生: {error_message}")
        print(Fore.YELLOW + f"生成的解决计划: {self.error_plan}")

    def generate_error_plan(self, error_message, original_step, task_node=None):
        """
        根据错误生成解决计划。
        """

        return f"""
    ===== RULES OF ERROR HANDLER =====
    Never forget you are an error handler and I am a user. Never flip roles! Your primary goal is to help resolve errors encountered during task execution.
    I must help you to resolve errors and ensure the task can continue smoothly.
    You must analyze the error based on the information provided and generate a resolution plan. The format of your resolution plan is: Resolution Plan: [YOUR PLAN], where "Resolution Plan" describes the steps to fix the error.
    You must provide one resolution step at a time.
    I must implement the resolution steps you suggest and provide feedback on whether they were successful.
    You should not ask me questions but instead provide clear instructions.
    Please note that errors can be complex. Do not attempt to resolve an error with a single step. You must break down the resolution into manageable steps.
    Here are some tips that will help you create effective resolution plans:
    <tips>
    I have various tools to use, such as search toolkit, web browser simulation toolkit, document relevant toolkit, code execution toolkit, etc. Thus, You must think how human will resolve the error step-by-step, and give me instructions just like that. For example, one may first use google search to find solutions related to the error message, then test the proposed solutions.
    Although the error may seem complex, a resolution does exist. If the initial resolution plan fails, try alternative approaches or methods that can achieve similar results.
    Always remind me to verify if the error has been resolved after implementing each step. This work can be done by retesting the failed step or checking system logs.
    If the resolution involves code changes, please remind me to test the code and get the result.
    Search results typically do not provide direct solutions. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the information, e.g. interact with the webpage, extract relevant content, etc.
    If the error involves external services or APIs, consider checking the service status or trying alternative APIs.
    For configuration-related errors, you can either use configuration management toolkits or write scripts to correct the settings.
    </tips>
    Now, here is the error context: <error>{error_message}</error>. Never forget our task!
    Now you must start to provide resolution steps to fix the error. Do not add anything else other than your resolution plan!
    Keep providing resolution steps until you think the error is resolved.
    When the error is resolved, you must only reply with a single word <ERROR_RESOLVED>.
    Never say <ERROR_RESOLVED> unless my responses indicate the error has been successfully resolved.
    """

        # return [
        #     f"分析错误原因: {error_message}",
        #     f"尝试修复错误（针对步骤: {original_step}）",
        #     "验证修复是否成功"
        # ]

    def execute_error_plan(self, original_step, task_node):
        """
        执行解决计划。
        """
        for step in self.error_plan:
            print(Fore.CYAN + f"执行解决计划步骤: {step}")
            if "ERROR_RESOLVED" in step:
                try:
                    # 动态创建一个新的 RolePlaying 会话来解决错误
                    error_resolution_session = RolePlaying(
                        assistant_role_name="Error Resolver",
                        assistant_agent_kwargs=dict(model=self.model),
                        user_role_name="Error Handler",
                        user_agent_kwargs=dict(model=self.model),
                        task_prompt=step,
                        with_task_specify=True,
                        task_specify_agent_kwargs=dict(model=self.model),
                        output_language='中文'
                    )
                    resolution_result = error_resolution_session.step()
                    if "成功" in resolution_result[0].msg.content:
                        self.task_manager.update_task_status(task_node, "completed")
                        return True  # 修复成功
                    else:
                        raise Exception("修复失败")
                except Exception as e:
                    self.retry_count += 1
                    if self.retry_count < self.max_retries:
                        print(Fore.RED + f"修复过程中再次发生错误: {e}")
                        print(Fore.YELLOW + "尝试新的解决方法...")
                        self.handle_error(str(e), original_step, task_node)
                    else:
                        print(Fore.RED + "达到最大重试次数，无法修复错误。")
                        self.task_manager.update_task_status(task_node, "failed")
                        return False
        return False


def main(model=model, chat_turn_limit=50):
    task_prompt = "开发一个股票市场交易机器人"
    task_manager = TaskManager("Root Task", task_prompt)

    # 创建任务树
    task1 = task_manager.add_task("Task 1", "设计交易策略")
    task2 = task_manager.add_task("Task 2", "实现数据采集模块")
    task3 = task_manager.add_task("Task 3", "开发交易执行模块")
    task4 = task_manager.add_task("Subtask 1", "测试数据采集模块", parent_node=task2)

    role_play_session = RolePlaying(
        assistant_role_name="Python 程序员",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="股票交易员",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        output_language='中文'
    )

    assistant_agent = role_play_session.assistant_agent
    user_agent = role_play_session.user_agent
    error_handling_agent = ErrorHandlingAgent(model, "Error Handler", "处理任务执行中的错误", task_manager)

    print(Fore.GREEN + f"AI 助手系统消息:\n{role_play_session.assistant_sys_msg}\n")
    print(Fore.BLUE + f"AI 用户系统消息:\n{role_play_session.user_sys_msg}\n")
    print(Fore.YELLOW + f"原始任务提示:\n{task_prompt}\n")
    print(Fore.CYAN + f"指定的任务提示:\n{role_play_session.specified_task_prompt}\n")
    print(Fore.RED + f"最终任务提示:\n{role_play_session.task_prompt}\n")

    n = 0
    input_msg = role_play_session.init_chat()
    current_task = task_manager.root

    while n < chat_turn_limit:
        n += 1
        try:
            assistant_response, user_response = role_play_session.step(input_msg)
            print_text_animated(Fore.BLUE + f"AI 用户:\n\n{user_response.msg.content}\n")
            print_text_animated(Fore.GREEN + f"AI 助手:\n\n{assistant_response.msg.content}\n")

            # 检查任务是否完成
            if "CAMEL_TASK_DONE" in user_response.msg.content:
                task_manager.update_task_status(current_task, "completed")
                print(Fore.GREEN + f"任务完成：{current_task.task_name}")
                if current_task.parent:
                    current_task = current_task.parent
                else:
                    print(Fore.GREEN + "所有任务完成！")
                    break

        except Exception as e:
            print(Fore.RED + f"在执行任务时发生错误: {e}")
            error_handling_agent.handle_error(str(e), input_msg, current_task)
            if error_handling_agent.execute_error_plan(input_msg, current_task):
                print(Fore.GREEN + "错误已成功修复，继续执行任务。")
            else:
                print(Fore.RED + "错误无法修复，任务终止。")
                break

        input_msg = assistant_response.msg

if __name__ == "__main__":
#     main()
    pass