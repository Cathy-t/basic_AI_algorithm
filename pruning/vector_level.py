# encoding: utf-8
# @File  : vector_level.py
# @Author: cathy
# @Desc : 
# @Date  :  2025/07/26

import numpy as np
import matplotlib.pyplot as plt

def vetor_pruning(matrix, idx):
    row, col = idx
    pruned_matrix = matrix.copy()
    pruned_matrix[row, :] = 0
    pruned_matrix[:, col] = 0

    return pruned_matrix

matrix = np.random.randn(3, 4)
idx = (1, 2)

# prune the matrix
pruned_matrix = vetor_pruning(matrix, idx)
print(matrix)
print(pruned_matrix)
