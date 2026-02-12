import pytest
from httpx import ASGITransport, AsyncClient
from vector_server import app
import torch


@pytest.mark.asyncio
async def test_embeddings_endpoint():
    # 使用 ASGITransport 直接调用 app，无需启动真实网络端口，速度极快
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 模拟 Java 发来的数据
        payload = {
            "input": ["你好，这是测试文本", "Hello BGE on 3070"]
        }

        response = await ac.post("/v1/embeddings", json=payload)

        # 1. 验证状态码
        assert response.status_code == 200

        data = response.json()

        # 2. 验证返回的是列表且长度匹配
        assert isinstance(data, list)
        assert len(data) == 2

        # 3. 验证向量维度 (BGE-Small-zh 应该是 512)
        assert len(data[0]) == 512

        # 4. 验证数值类型
        assert isinstance(data[0][0], float)


def test_gpu_health():
    """ 验证模型是否确实加载到了 3070 显存 """
    assert torch.cuda.is_available()
    # 这里的 model 需要能从 server.py 访问到
    from vector_server import model
    assert str(model.device).startswith("cuda")
