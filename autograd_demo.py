import torch

# 1. 定义数据 (类似我们数据库里的原始记录)
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([5.0, 8.0, 11.0, 14.0]) # 这是真实规律 y = 3x + 2

# 2. 初始化参数 (w 是权重, b 是截距)
# requires_grad=True 表示我们要 PyTorch 跟踪这个变量的梯度，以便后续优化
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 3. 模拟一次预测 (Forward pass)
y_pred = w * x + b

# 4. 计算损失 (Loss) - 预测值和真实值差了多少
loss = ((y_pred - y) ** 2).mean()

# 5. 反向传播 (Backward pass) - 核心魔法
loss.backward()

# 查看梯度：w 和 b 应该朝哪个方向调整？
print(f"w 的梯度: {w.grad}")
print(f"b 的梯度: {b.grad}")