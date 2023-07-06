# -*- coding: utf-8 -*-
# ModuleName: mse_loss
# Author: dongyihua543
# Time: 2023/7/6 15:44

import torch
import torch.nn.functional as F

# 定义两个张量
input_tensor = torch.tensor([1.0, 1.0, 1.0])
target_tensor = torch.tensor([-1.0, -0.2, -0.1])

# 计算均方误差
loss = F.mse_loss(input_tensor, target_tensor)


print(loss)
