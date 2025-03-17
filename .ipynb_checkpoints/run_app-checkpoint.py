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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OWL 智能助手运行平台启动脚本
"""

import os
import sys
from pathlib import Path


def main():
    """主函数，启动OWL智能助手运行平台"""
    # 确保当前目录是项目根目录
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    # 创建日志目录
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # 导入并运行应用
    sys.path.insert(0, str(project_root))

    try:
        from owl.app import create_ui

        # 创建并启动应用
        app = create_ui()
        app.queue().launch(share=False)

    except ImportError as e:
        print(f"错误: 无法导入必要的模块。请确保已安装所有依赖项: {e}")
        print("提示: 运行 'pip install -r requirements.txt' 安装所有依赖项")
        sys.exit(1)
    except Exception as e:
        print(f"启动应用程序时出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
