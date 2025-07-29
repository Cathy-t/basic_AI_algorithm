# encoding: utf-8
# @File  : fine_tuning.py
# @Author: cathy
# @Desc : 
# @Date  :  2025/07/23

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from transformers import GPT2LMHeadModel, BertModel, BertConfig
from transformers import AutoTokenizer
import math


# 1. 简单数据集类
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encoded_texts = tokenizer(texts, padding='max_length',
                                       truncation=True, max_length=max_length,
                                       return_tensors='pt')
        self.input_ids = self.encoded_texts['input_ids']
        self.attention_mask = self.encoded_texts['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }


# 2. 自回归预训练模型（类似GPT）
class ARPretrainer(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=6):
        super().__init__()
        # 使用TransformerDecoder作为基础架构
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_layers
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        # 创建因果掩码（防止看到未来token）
        seq_len = input_ids.size(1)
        causal_mask = generate_square_subsequent_mask(seq_len).to(input_ids.device)

        # 嵌入层
        src = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)

        # 转换为Transformer期望的维度 [seq_len, batch_size, embed_dim]
        src = src.permute(1, 0, 2)
        tgt_mask = causal_mask

        # 通过Transformer层
        output = self.transformer(src, src, tgt_mask=tgt_mask)

        # 转换回 [batch_size, seq_len, embed_dim]
        output = output.permute(1, 0, 2)

        # 语言模型头部
        logits = self.lm_head(output)
        return logits


# 3. 自编码预训练模型（类似BERT）
class AEPretrainer(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=6):
        super().__init__()
        # 使用TransformerEncoder作为基础架构
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        # 嵌入层
        src = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)

        # 转换为Transformer期望的维度 [seq_len, batch_size, embed_dim]
        src = src.permute(1, 0, 2)
        # 创建注意力掩码
        attn_mask = ~attention_mask.bool()  # 0位置被mask

        # 通过Transformer层
        output = self.transformer(src, src_key_padding_mask=attn_mask)

        # 转换回 [batch_size, seq_len, embed_dim]
        output = output.permute(1, 0, 2)

        # 掩码语言模型头部
        logits = self.mlm_head(output)
        return logits


# 4. 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        return x + self.pe[:x.size(1), :].unsqueeze(0)


# 5. 生成因果掩码（自回归模型用）
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# 6. 自回归预训练过程
def autoregressive_pretraining(model, dataloader, optimizer, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 目标是预测下一个token（右移一位）
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()

            # 前向传播
            logits = model(input_ids, attention_mask)

            # 计算损失（忽略padding位置）
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0  # 假设0是padding token
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


# 7. 自编码预训练过程
def autoencoding_pretraining(model, dataloader, optimizer, device, epochs=1, mask_prob=0.15):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 创建掩码版本的输入和标签
            masked_inputs, labels = create_masked_lm_predictions(
                input_ids, mask_prob, model.embedding.num_embeddings
            )

            # 前向传播
            logits = model(masked_inputs, attention_mask)

            # 计算损失（只计算被mask的token）
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100  # 忽略未被mask的位置
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


# 8. 创建掩码语言模型的输入和标签
def create_masked_lm_predictions(tokens, mask_prob, vocab_size, ignore_index=-100):
    """
    创建掩码语言模型的训练样本
    tokens: [batch_size, seq_len]
    """
    labels = tokens.clone()

    # 80%的概率用[MASK]替换
    mask_token_id = 103  # 假设103是[MASK] token

    # 随机选择要mask的位置
    for i in range(tokens.size(0)):
        # 不mask特殊token和padding
        maskable_positions = (tokens[i] > 100) & (tokens[i] != 0)
        num_mask = int(mask_prob * maskable_positions.sum().item())
        mask_indices = torch.nonzero(maskable_positions, as_tuple=True)[0]

        if len(mask_indices) > 0:
            # 随机选择mask的位置
            mask_indices = mask_indices[torch.randperm(len(mask_indices))[:num_mask]]

            # 80%的概率替换为[MASK]
            indices_replaced = torch.bernoulli(torch.full_like(mask_indices, 0.8, dtype=torch.float)).bool()
            tokens[i, mask_indices[indices_replaced]] = mask_token_id

            # 10%的概率替换为随机token
            indices_random = (~indices_replaced) & torch.bernoulli(
                torch.full_like(mask_indices, 0.5, dtype=torch.float)).bool()
            random_tokens = torch.randint(100, vocab_size, size=mask_indices[indices_random].size(),
                                          device=tokens.device)
            tokens[i, mask_indices[indices_random]] = random_tokens

            # 剩下10%保持不变

            # 设置标签（只计算被mask的位置）
            labels[i, ~maskable_positions] = ignore_index

    return tokens, labels


# 9. 主函数示例
def main():
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 示例文本数据
    texts = [
        "自然语言处理是人工智能的一个重要领域",
        "Transformer架构在NLP任务中取得了巨大成功",
        "预训练模型可以显著提高下游任务的性能",
        "自回归模型和自编码模型是两种主要的预训练范式"
    ]

    # 创建数据集和数据加载器
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 自回归模型训练
    ar_model = ARPretrainer(vocab_size=tokenizer.vocab_size).to(device)
    ar_optimizer = torch.optim.Adam(ar_model.parameters(), lr=1e-4)
    print("开始自回归预训练...")
    autoregressive_pretraining(ar_model, dataloader, ar_optimizer, device, epochs=2)

    # 自编码模型训练
    ae_model = AEPretrainer(vocab_size=tokenizer.vocab_size).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-4)
    print("\n开始自编码预训练...")
    autoencoding_pretraining(ae_model, dataloader, ae_optimizer, device, epochs=2)


if __name__ == "__main__":
    main()
