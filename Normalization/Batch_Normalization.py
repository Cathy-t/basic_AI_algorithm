# encoding: utf-8
# @File  : Batch_Normalization.py
# @Author: cathy
# @Desc : 
# @Date  :  2025/07/22

import torch
import torch.nn as nn

# 输入通道数=3，Batch Normalization
bn = nn.BatchNorm2d(num_features=3)  # 对3个通道分别归一化

# 输入：[批次大小, 通道数, 高度, 宽度]
x = torch.randn(16, 3, 32, 32)  # 批次大小16，3通道，32×32图像
output = bn(x)
print(output.shape)
print(output)