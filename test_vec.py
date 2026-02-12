import json
from uuid import uuid4

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# 1. 加载模型（会自动下载到 ~/.cache/torch，约 100MB）
# 使用 cuda 强制让 3070 显卡工作
model = SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cuda')

# 2. 数据库配置
conn_info = "postgresql://rag_user:Abc110430#@127.0.0.1:5432/rag_db"


def save_to_pg():
    # 原始数据
    raw_data = {
        "file_id": uuid4(),
        "page_num": 12,
        "content": "PostgreSQL 18 配合 pgvector 是构建本地 RAG 系统的理想选择。",
        "text_type": "paragraph",
        "bbox": {"x": 100, "y": 200, "w": 300, "h": 20}
    }

    # 3. 生成真实向量 (bge-small-zh 维度是 512)
    print("正在使用 RTX 3070 生成向量...")
    vector = model.encode(raw_data["content"]).tolist()

    try:
        with psycopg.connect(conn_info, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)

            # 4. 存入数据库
            conn.execute("""
                         INSERT INTO doc_chunks (file_id, content, page_num, bbox, embedding)
                         VALUES (%s, %s, %s, %s, %s)
                         """, (raw_data["file_id"], raw_data["content"], raw_data["page_num"],
                               json.dumps(raw_data["bbox"]), vector))

            print(f"✅ 成功！已将文本存入 Postgres，向量维度: {len(vector)}")

    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    save_to_pg()
