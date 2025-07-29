# encoding: utf-8
# @File  : mask_test.py
# @Author: cathy
# @Desc : 
# @Date  :  2025/07/22

import torch
import torch.nn.functional as F


def create_padding_mask(seq, pad_idx=0):
    """创建padding mask，将pad_idx位置标记为False"""
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    return mask


def create_sequence_mask(seq_len):
    """创建sequence mask（下三角矩阵），防止未来信息泄露"""
    mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩阵
    return mask  # [seq_len, seq_len]


def attention_with_mask(q, k, v, padding_mask=None, sequence_mask=None):
    """计算带mask的注意力机制"""
    # 计算注意力分数 [batch_size, num_heads, seq_len_q, seq_len_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))

    # 应用padding mask（将mask中为False的位置对应的scores设为负无穷）
    if padding_mask is not None:
        scores = scores.masked_fill(padding_mask == False, -1e9)

    # 应用sequence mask（下三角矩阵，屏蔽未来信息）
    if sequence_mask is not None:
        scores = scores.masked_fill(sequence_mask == 0, -1e9)

    # softmax获取注意力权重
    attn_weights = F.softmax(scores, dim=-1)

    # 加权求和得到输出
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# 示例：修改第二个序列，使其padding位置与未来信息不重叠
batch_size = 2
seq_len = 4
hidden_size = 8
num_heads = 2
head_dim = hidden_size // num_heads

# 创建输入序列（包含padding）
sequences = torch.tensor([
    [1, 2, 3, 4],  # 真实序列
    [5, 6, 7, 0]  # 真实序列长度为3，最后一个是padding
])

# 创建Q, K, V矩阵（模拟多头注意力）
q = torch.randn(batch_size, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim)
v = torch.randn(batch_size, num_heads, seq_len, head_dim)

# 1. 仅使用padding mask
padding_mask = create_padding_mask(sequences)
output_padding, attn_padding = attention_with_mask(q, k, v, padding_mask=padding_mask)

# 2. 仅使用sequence mask
sequence_mask = create_sequence_mask(seq_len)
output_sequence, attn_sequence = attention_with_mask(q, k, v, sequence_mask=sequence_mask)

# 3. 同时使用两种mask
output_both, attn_both = attention_with_mask(q, k, v, padding_mask, sequence_mask)

print("\nPadding Mask 效果:")
print(attn_padding[1, 0].detach().numpy().round(2))  # 打印第二个样本的第一个头

print("\nSequence Mask 效果:")
print(attn_sequence[1, 0].detach().numpy().round(2))

print("\n两种Mask结合的效果:")
print(attn_both[1, 0].detach().numpy().round(2))

# 验证差异：比较两种mask的注意力权重
print("\n验证差异（比较sequence mask和两种mask结合的结果）:")
print("是否完全相同:", torch.allclose(attn_sequence, attn_both))