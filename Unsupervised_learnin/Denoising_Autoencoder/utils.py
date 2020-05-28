#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/4/15 17:54
# @Author  : Cathy 
# @FileName: utils.py

import torch

def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand < frac] = 0
    return data_noise

