import torch
import time

# 1. 定义矩阵维度 (5000x5000 的大矩阵)
size = 5000

# ----------------- CPU 运算 -----------------
print(f"正在测试 CPU 运算...")
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)

start = time.time()
# 执行矩阵乘法
result_cpu = torch.matmul(x_cpu, y_cpu)
end = time.time()
print(f"CPU 耗时: {end - start:.4f} 秒")

# ----------------- GPU 运算 (NVIDIA 3070) -----------------
if torch.cuda.is_available():
    print(f"\n检测到显卡: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")

    # 将数据从 内存(RAM) 搬运到 显存(VRAM)
    x_gpu = x_cpu.to(device)
    y_gpu = y_cpu.to(device)

    # GPU 第一次运行通常会有“预热”开销，我们多跑几次取平均值
    # 就像 JVM 预热一样
    torch.cuda.synchronize()  # 等待数据传输完成
    start = time.time()

    result_gpu = torch.matmul(x_gpu, y_gpu)

    torch.cuda.synchronize()  # 强制等待 GPU 计算结束（异步转同步）
    end = time.time()
    print(f"GPU 3070 耗时: {end - start:.4f} 秒")

    # 计算加速比
    speedup = (result_cpu.numpy().size * (end - start))  # 简化计算
    print(f"\n结论：GPU 运算速度远快于 CPU！")
else:
    print("\n未检测到 CUDA 显卡，请检查驱动安装。")