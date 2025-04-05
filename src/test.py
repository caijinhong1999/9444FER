import torch
print(torch.__version__)             # 应该是 2.x.x
print(torch.version.cuda)            # 应该输出 12.6
print(torch.cuda.is_available())     # True 表示 GPU 可用
print(torch.cuda.get_device_name(0)) # 应该是你的 RTX 4060