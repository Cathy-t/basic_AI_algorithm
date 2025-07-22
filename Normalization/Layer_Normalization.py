# encoding: utf-8
# @File  : Layer_Normalization.py
# @Author: cathy
# @Desc : 
# @Date  :  2025/07/22

import torch
import torch.nn as nn

# Layer Normalization，指定归一化维度
ln = nn.LayerNorm(normalized_shape=[3, 32, 32])  # 对后三个维度（C,H,W）归一化

# 输入相同
x = torch.randn(16, 3, 32, 32)
output = ln(x)
print(output.shape)
print(output)
