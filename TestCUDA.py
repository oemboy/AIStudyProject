import torch
import os
from sentence_transformers import SentenceTransformer

# 1. 加载模型（会自动下载到 ~/.cache/torch，约 100MB）
# 使用 cuda 强制让 3070 显卡工作
# 运行一次下载到本地

# 以后代码里改用
# model = SentenceTransformer('BAAI/bge-small-zh-v1.5', device="cuda")
# model.save('./models/bge-small-zh-v1.5')
# model = SentenceTransformer('./models/bge-small-zh-v1.5', device='cuda', local_files_only=True)

model_path = './models/bge-large-zh-v1.5'
# 远程 HuggingFace 模型名称
model_name = 'BAAI/bge-large-zh-v1.5'

if os.path.exists(model_path):
    print(f"--- Found local model at {model_path}, loading... ---")
    # 如果本地文件夹存在，直接从本地路径加载
    model = SentenceTransformer(model_path, device="cuda")
else:
    print(f"--- Local model not found. Downloading {model_name}... ---")
    # 如果本地不存在，从远程下载
    model = SentenceTransformer(model_name, device="cuda")
    # 下载后立即保存到本地
    print(f"--- Saving model to {model_path} ---")
    model.save(model_path)

print("--- Model is ready on 3070! ---")

# 2. 数据库配置
conn_info = "postgresql://rag_user:Abc110430#@127.0.0.1:5432/rag_db"

# 1. 检查 CUDA 是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# 2. 查看当前显卡名称
if torch.cuda.is_available():
    print(f"当前使用的显卡: {torch.cuda.get_device_name(0)}")

# 3. 确认模型所在的设备
# model 是你定义的 SentenceTransformer 对象
print(f"模型当前运行设备: {model.device}")

print(f"CUDA 版本: {torch.version.cuda}")
