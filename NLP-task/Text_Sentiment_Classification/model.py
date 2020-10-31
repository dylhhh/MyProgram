import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, dropout):
        super(GRU, self).__init__()

        # 使用训练好的embedding初始化模型的embedding层
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = True  # embedding会一起被训练

        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 模型
        self.gru = self.rnn = nn.GRU(self.embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 GRU 最后一维的 hidden state
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
