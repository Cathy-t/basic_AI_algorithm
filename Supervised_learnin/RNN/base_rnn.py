# encoding: utf-8
# @File  : base_rnn.py
# @Author: cathy
# @Desc : 
# @Date  :  2025/07/22

import torch
import torch.nn as nn

rnn = nn.RNN(5, 6, 2)  # 实例化rnn对象
# 参数1：输入张量x的维度-input_size
# 参数2：隐藏层的维度（隐藏层神经元个数）-hidden_size
# 参数3：隐藏层的层数-num_layers

# torch.randn-随机产生正态分布的随机数
input1 = torch.randn(1, 3, 5)  # 设定输入张量x-1层3行5列
# 参数1：输入序列长度-sequence_lengh
# 参数2：批次的样本-batch_size（表示：3个样本每个样本一个字母（序列长））
# 参数3：输入张量x的维度-input_size

h0 = torch.randn(2, 3, 6)  # 设定初始化的h0
# 第一个参数: num_layers * num_directions(层数+正向/反向(1或2))
# 第二个参酸: batch_size(批次的样本数)
# 第三个参酸: hidden_size(隐藏层的维度)

output, hn = rnn(input1, h0)
# 最后输出和最后一层的隐藏层输出

print(output)
print(output.shape)
print(hn)
print(hn.shape)
