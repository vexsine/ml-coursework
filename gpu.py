import torch
print(torch.cuda.is_available())  # 如果返回True，则表示GPU可用
print(torch.cuda.current_device())  # 查看当前使用的GPU
print(torch.cuda.get_device_name(0))  # 获取GPU的名称
