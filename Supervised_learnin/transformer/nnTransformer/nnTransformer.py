# encoding: utf-8
# @File  : nnTransformer.py
# @Author: cathy
# @Desc : 
# @Date  :  2025/07/28

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
from typing import Tuple

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# 简单的序列到序列数据集
class Seq2SeqDataset(Dataset):
    def __init__(self, num_samples, max_len=10, vocab_size=10):
        self.num_samples = num_samples
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.data = []

        # 生成简单的源序列和目标序列（例如，源序列反转后作为目标序列）
        for _ in range(num_samples):
            seq_len = random.randint(3, max_len)
            src = torch.randint(1, vocab_size, (seq_len,))  # 0用于填充
            tgt = torch.flip(src, dims=[0])  # 目标序列是源序列的反转

            self.data.append((src, tgt))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


# 填充和批处理函数
def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)

    # 填充源序列
    src_lengths = [len(seq) for seq in src_seqs]
    src_padded = torch.zeros(len(src_seqs), max(src_lengths)).long()
    for i, seq in enumerate(src_seqs):
        src_padded[i, :len(seq)] = seq

    # 填充目标序列（注意Transformer的目标序列需要包含<sos>和<eos>）
    tgt_lengths = [len(seq) + 2 for seq in tgt_seqs]  # +2 for <sos> and <eos>
    tgt_padded = torch.zeros(len(tgt_seqs), max(tgt_lengths)).long()

    # 添加<sos>(1)和<eos>(2)
    for i, seq in enumerate(tgt_seqs):
        tgt_padded[i, 0] = 1  # <sos>
        tgt_padded[i, 1:len(seq) + 1] = seq
        tgt_padded[i, len(seq) + 1] = 2  # <eos>

    return src_padded, tgt_padded, src_lengths, tgt_lengths


# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, ninp: int, nhead: int, nhid: int, nlayers: int, dropout: float = 0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformer = nn.Transformer(
            d_model=ninp,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=nhid,
            dropout=dropout
        )
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)
        self.ninp = ninp
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        # 生成上三角矩阵（用于Decoder的掩码），指定设备
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor) -> torch.Tensor:
        # 添加位置编码
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        tgt = self.encoder(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)

        # 生成目标序列的掩码（防止看到未来的token），使用src的设备
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), device=src.device)

        # 调整维度顺序以匹配Transformer的期望 (S, N, E)
        src = src.permute(1, 0, 2)  # [batch_size, src_seq_len, d_model] → [src_seq_len, batch_size, d_model]
        tgt = tgt.permute(1, 0, 2)  # [batch_size, tgt_seq_len, d_model] → [tgt_seq_len, batch_size, d_model]

        # 运行Transformer：掩码保持[batch_size, seq_len]形状
        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=None,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,  # 形状[batch_size, src_seq_len]
            tgt_key_padding_mask=tgt_padding_mask,  # 形状[batch_size, tgt_seq_len]
            memory_key_padding_mask=src_padding_mask
        )

        # 解码输出：转换回[batch_size, seq_len, ntoken]
        output = self.decoder(output)
        return output.permute(1, 0, 2)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, embedding_dim]
        # 需要截取的是seq_len，而不是batch_size
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)  # 添加batch维度 [1, seq_len, d_model]
        return self.dropout(x)


# 训练函数
def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
          device: torch.device) -> float:
    model.train()
    total_loss = 0

    for src, tgt, src_lengths, tgt_lengths in train_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # 准备目标输入和目标输出（目标输出是目标输入右移一位）
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # 创建填充掩码
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt_input == 0)

        optimizer.zero_grad()
        output = model(src, tgt_input, src_padding_mask, tgt_padding_mask)

        # 计算损失（忽略填充位置）
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


# 评估函数
def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt, src_lengths, tgt_lengths in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_padding_mask = (src == 0)
            tgt_padding_mask = (tgt_input == 0)

            output = model(src, tgt_input, src_padding_mask, tgt_padding_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(data_loader)


# 测试函数（生成序列）
def test(model: nn.Module, src: torch.Tensor, max_len: int, sos_idx: int, eos_idx: int,
         device: torch.device) -> torch.Tensor:
    model.eval()

    src = src.to(device).unsqueeze(0)  # 添加batch维度
    src_padding_mask = (src == 0)

    # 初始化输出序列
    tgt = torch.ones(1, 1).fill_(sos_idx).long().to(device)

    with torch.no_grad():
        for i in range(max_len - 1):
            tgt_padding_mask = (tgt == 0)

            output = model(src, tgt, src_padding_mask, tgt_padding_mask)
            prob = output[:, -1, :]  # 获取最后一个token的预测

            # 贪婪解码：选择概率最高的token
            _, next_token = torch.max(prob, dim=1)
            next_token = next_token.unsqueeze(1)

            tgt = torch.cat([tgt, next_token], dim=1)

            # 如果生成了<eos>，则停止生成
            if next_token.item() == eos_idx:
                break

    return tgt


# 主函数
def main():
    # 模型参数
    vocab_size = 10  # 词汇表大小
    d_model = 128  # 模型维度
    nhead = 8  # 注意力头数
    nhid = 256  # 前馈网络隐藏层维度
    nlayers = 2  # 编码器/解码器层数
    dropout = 0.1  # Dropout率

    # 训练参数
    batch_size = 32
    epochs = 10
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集和数据加载器
    train_dataset = Seq2SeqDataset(num_samples=1000)
    val_dataset = Seq2SeqDataset(num_samples=200)
    test_dataset = Seq2SeqDataset(num_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  # 测试时batch_size=1

    # 初始化模型、损失函数和优化器
    model = TransformerModel(vocab_size, d_model, nhead, nhid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充索引0
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'transformer_model.pt')

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('transformer_model.pt'))

    # 测试生成功能
    print("\n测试生成功能:")
    for i, (src, tgt, _, _) in enumerate(test_loader):
        if i >= 5:  # 只测试前5个样本
            break

        generated = test(model, src[0], max_len=20, sos_idx=1, eos_idx=2, device=device)

        print(f"\n源序列: {src[0].tolist()}")
        print(f"目标序列: {tgt[0].tolist()[1:-1]}")  # 去除<sos>和<eos>
        print(f"生成序列: {generated[0].tolist()[1:-1]}")  # 去除<sos>和<eos>


if __name__ == '__main__':
    main()
