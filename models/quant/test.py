import torch
from utils.device import get_optimal_device_name, get_optimal_device
import torch.nn as nn
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

# 1. 在GPU上创建一个简单的双层线性模型，并设置为评估模式
model = nn.Sequential(
    nn.Linear(1024, 512),   # 第一层：输入1024维，输出512维
    nn.Linear(512, 256)     # 第二层：输入512维，输出256维
).eval().to(get_optimal_device())

# 2. 应用 int8 动态量化的配置，同时量化激活值和权重
quantize_(model, Int8DynamicActivationInt8WeightConfig())

# 3. 打印检查量化后的权重类型，应输出类似 'Int8Tensor' 的结果
print(type(model[0].weight).__name__)

# 4. 创建与模型第一层输入维度匹配的随机输入
input_tensor = torch.randn(1, 1024, device=get_optimal_device_name())

# 5. 执行推理，整个过程会自动进行int8计算
output = model(input_tensor)

# 6. 打印输出张量的形状和设备，以验证推理成功
print(f"Output shape: {output.shape}, device: {output.device}")
