# 使用ARG定义可配置的构建参数
ARG PYTHON_VERSION=3.10
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright

# 第一阶段：构建依赖
FROM python:${PYTHON_VERSION}-slim AS builder

# 设置工作目录
WORKDIR /build

# 设置pip镜像源以加速下载
ARG PIP_INDEX_URL
RUN pip config set global.index-url ${PIP_INDEX_URL}

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 第二阶段：运行时环境
FROM python:${PYTHON_VERSION}-slim

# 添加构建信息标签
ARG BUILD_DATE
ARG VERSION
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.title="OWL Project" \
      org.opencontainers.image.description="OWL Project Docker Image" \
      org.opencontainers.image.source="https://github.com/yourusername/owl"

# 设置工作目录
WORKDIR /app

# 设置pip镜像源以加速下载
ARG PIP_INDEX_URL
RUN pip config set global.index-url ${PIP_INDEX_URL}

# 从builder阶段复制已安装的Python包
COPY --from=builder /install /usr/local

# 优化apt安装，减少层数
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    # 添加xvfb和相关依赖
    xvfb \
    xauth \
    x11-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 Playwright 依赖（使用国内镜像源）
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright
ARG PLAYWRIGHT_DOWNLOAD_HOST
ENV PLAYWRIGHT_DOWNLOAD_HOST=${PLAYWRIGHT_DOWNLOAD_HOST}
RUN pip install --no-cache-dir playwright && \
    playwright install --with-deps chromium

# 创建非root用户
RUN groupadd -r owl && useradd -r -g owl -m owl

# 复制项目文件
COPY owl/ ./owl/
COPY licenses/ ./licenses/
COPY assets/ ./assets/
COPY README.md .
COPY README_zh.md .

# 设置环境变量文件
COPY owl/.env_template ./owl/.env

# 创建启动脚本
RUN echo '#!/bin/bash\nxvfb-run --auto-servernum --server-args="-screen 0 1280x960x24" python "$@"' > /usr/local/bin/xvfb-python && \
    chmod +x /usr/local/bin/xvfb-python

# 创建欢迎脚本
RUN echo '#!/bin/bash\necho "欢迎使用OWL项目Docker环境！"\necho ""\necho "可用的脚本:"\nls -1 *.py | grep -v "__" | sed "s/^/- /"\necho ""\necho "运行示例:"\necho "  xvfb-python run.py                     # 运行默认脚本"\necho "  xvfb-python run_deepseek_example.py      # 运行DeepSeek示例"\necho ""\necho "或者使用自定义查询:"\necho "  xvfb-python run.py \"你的问题\""\necho ""' > /usr/local/bin/owl-welcome && \
    chmod +x /usr/local/bin/owl-welcome

# 设置工作目录
WORKDIR /app/owl

# 设置适当的权限
RUN chown -R owl:owl /app
RUN mkdir -p /root/.cache && chown -R owl:owl /root/.cache
RUN chmod 644 /app/owl/.env


USER owl

# 添加健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0 if __import__('os').path.exists('/app/owl') else 1)"

# 容器启动命令
CMD ["/bin/bash", "-c", "owl-welcome && /bin/bash"] 