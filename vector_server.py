from fastapi import FastAPI, Body
from sentence_transformers import SentenceTransformer
import torch
import time

app = FastAPI()

print("--- Initializing BGE Model on nvidia ---")
model = SentenceTransformer('./models/bge-large-zh-v1.5', device="cuda")


@app.post("/v1/embeddings")
async def get_embeddings(payload: dict = Body(...)):
    texts = payload.get("input", [])
    if not texts:
        return []

    # --- 开始计时 ---
    start_time = time.perf_counter()

    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            embeddings = model.encode(texts, normalize_embeddings=True)

    # --- 计算耗时 ---
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    print(f"🚀 [nvidia Inference] Processed {len(texts)} chunks | Time: {duration_ms:.2f}ms")

    return embeddings.tolist()