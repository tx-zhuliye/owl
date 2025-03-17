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
import importlib.util
import re
from pathlib import Path
import traceback


def load_module_from_path(module_name, file_path):
    """从文件路径加载Python模块"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"错误: 无法从 {file_path} 创建模块规范")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"加载模块时出错: {e}")
        traceback.print_exc()
        return None


def run_script_with_env_question(script_name):
    """使用环境变量中的问题运行脚本"""
    # 获取环境变量中的问题
    question = os.environ.get("OWL_QUESTION")
    if not question:
        print("错误: 未设置OWL_QUESTION环境变量")
        sys.exit(1)

    # 脚本路径
    script_path = Path(script_name).resolve()
    if not script_path.exists():
        print(f"错误: 脚本 {script_path} 不存在")
        sys.exit(1)

    # 创建临时文件路径
    temp_script_path = script_path.with_name(f"temp_{script_path.name}")

    try:
        # 读取脚本内容
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"读取脚本文件时出错: {e}")
            sys.exit(1)

        # 检查脚本是否有main函数
        has_main = re.search(r"def\s+main\s*\(\s*\)\s*:", content) is not None

        # 转义问题中的特殊字符
        escaped_question = (
            question.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
        )

        # 查找脚本中所有的question赋值 - 改进的正则表达式
        # 匹配单行和多行字符串赋值
        question_assignments = re.findall(
            r'question\s*=\s*(?:["\'].*?["\']|""".*?"""|\'\'\'.*?\'\'\'|\(.*?\))',
            content,
            re.DOTALL,
        )
        print(f"在脚本中找到 {len(question_assignments)} 个question赋值")

        # 修改脚本内容，替换所有的question赋值
        modified_content = content

        # 如果脚本中有question赋值，替换所有的赋值
        if question_assignments:
            for assignment in question_assignments:
                modified_content = modified_content.replace(
                    assignment, f'question = "{escaped_question}"'
                )
            print(f"已替换脚本中的所有question赋值为: {question}")
        else:
            # 如果没有找到question赋值，尝试在main函数前插入
            if has_main:
                main_match = re.search(r"def\s+main\s*\(\s*\)\s*:", content)
                if main_match:
                    insert_pos = main_match.start()
                    modified_content = (
                        content[:insert_pos]
                        + f'\n# 用户输入的问题\nquestion = "{escaped_question}"\n\n'
                        + content[insert_pos:]
                    )
                    print(f"已在main函数前插入问题: {question}")
            else:
                # 如果没有main函数，在文件开头插入
                modified_content = (
                    f'# 用户输入的问题\nquestion = "{escaped_question}"\n\n' + content
                )
                print(f"已在文件开头插入问题: {question}")

        # 添加monkey patch代码，确保construct_society函数使用用户的问题
        monkey_patch_code = f"""
# 确保construct_society函数使用用户的问题
original_construct_society = globals().get('construct_society')
if original_construct_society:
    def patched_construct_society(*args, **kwargs):
        # 忽略传入的参数，始终使用用户的问题
        return original_construct_society("{escaped_question}")
    
    # 替换原始函数
    globals()['construct_society'] = patched_construct_society
    print("已修补construct_society函数，确保使用用户问题")
"""

        # 在文件末尾添加monkey patch代码
        modified_content += monkey_patch_code

        # 如果脚本没有调用main函数，添加调用代码
        if has_main and "__main__" not in content:
            modified_content += """

# 确保调用main函数
if __name__ == "__main__":
    main()
"""
            print("已添加main函数调用代码")

        # 如果脚本没有construct_society调用，添加调用代码
        if (
            "construct_society" in content
            and "run_society" in content
            and "Answer:" not in content
        ):
            modified_content += f"""

# 确保执行construct_society和run_society
if "construct_society" in globals() and "run_society" in globals():
    try:
        society = construct_society("{escaped_question}")
        from utils import run_society
        answer, chat_history, token_count = run_society(society)
        print(f"Answer: {{answer}}")
    except Exception as e:
        print(f"运行时出错: {{e}}")
        import traceback
        traceback.print_exc()
"""
            print("已添加construct_society和run_society调用代码")

        # 执行修改后的脚本
        try:
            # 将脚本目录添加到sys.path
            script_dir = script_path.parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))

            # 创建临时文件
            try:
                with open(temp_script_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)
                print(f"已创建临时脚本文件: {temp_script_path}")
            except Exception as e:
                print(f"创建临时脚本文件时出错: {e}")
                sys.exit(1)

            try:
                # 直接执行临时脚本
                print("开始执行脚本...")

                # 如果有main函数，加载模块并调用main
                if has_main:
                    # 加载临时模块
                    module_name = f"temp_{script_path.stem}"
                    module = load_module_from_path(module_name, temp_script_path)

                    if module is None:
                        print(f"错误: 无法加载模块 {module_name}")
                        sys.exit(1)

                    # 确保模块中有question变量，并且值是用户输入的问题
                    setattr(module, "question", question)

                    # 如果模块中有construct_society函数，修补它
                    if hasattr(module, "construct_society"):
                        original_func = module.construct_society

                        def patched_func(*args, **kwargs):
                            return original_func(question)

                        module.construct_society = patched_func
                        print("已在模块级别修补construct_society函数")

                    # 调用main函数
                    if hasattr(module, "main"):
                        print("调用main函数...")
                        module.main()
                    else:
                        print(f"错误: 脚本 {script_path} 中没有main函数")
                        sys.exit(1)
                else:
                    # 如果没有main函数，直接执行修改后的脚本
                    print("直接执行脚本内容...")
                    # 使用更安全的方式执行脚本
                    with open(temp_script_path, "r", encoding="utf-8") as f:
                        script_code = f.read()

                    # 创建一个安全的全局命名空间
                    safe_globals = {
                        "__file__": str(temp_script_path),
                        "__name__": "__main__",
                    }
                    # 添加内置函数
                    safe_globals.update(
                        {k: v for k, v in globals().items() if k in ["__builtins__"]}
                    )

                    # 执行脚本
                    exec(script_code, safe_globals)

            except Exception as e:
                print(f"执行脚本时出错: {e}")
                traceback.print_exc()
                sys.exit(1)

        except Exception as e:
            print(f"处理脚本时出错: {e}")
            traceback.print_exc()
            sys.exit(1)

    except Exception as e:
        print(f"处理脚本时出错: {e}")
        traceback.print_exc()
        sys.exit(1)

    finally:
        # 删除临时文件
        if temp_script_path.exists():
            try:
                temp_script_path.unlink()
                print(f"已删除临时脚本文件: {temp_script_path}")
            except Exception as e:
                print(f"删除临时脚本文件时出错: {e}")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python script_adapter.py <script_path>")
        sys.exit(1)

    # 运行指定的脚本
    run_script_with_env_question(sys.argv[1])
