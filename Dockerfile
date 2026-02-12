FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# 环境变量：非交互模式 + 时区
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境（优雅地避开 --break-system-packages）
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装推理相关依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi uvicorn sentence-transformers torch \
    --extra-index-url https://download.pytorch.org/whl/cu130

# 复制本地的模型和代码
COPY ./models /app/models
COPY vector_server.py /app/

EXPOSE 8000

# 启动服务
CMD ["uvicorn", "vector_server:app", "--host", "0.0.0.0", "--port", "8000"]