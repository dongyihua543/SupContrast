# -*- coding: utf-8 -*-
# ModuleName: mse_loss
# Author: dongyihua543
# Time: 2023/7/6 15:44


import torch
import torch.nn as nn

# 创建输入和目标张量
# input_tensor = torch.tensor([1.0, 2.0, 3.0])
# target_tensor = torch.tensor([2.0, 4.0, 6.0])

input_tensor = torch.tensor([1.0, 1.0, 1.0])
target_tensor = torch.tensor([-1.0, -0.2, -0.1])

# 创建 MSELoss 对象
criterion = nn.MSELoss()

# 计算损失
loss = criterion(input_tensor, target_tensor)

print(loss)
