import math
import torch
from torch import nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        scale = math.sqrt(key.shape[2])
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        Q = self.fc_q(query)  # Q = [batch size, query len, hid dim]
        K = self.fc_k(key)  # K = [batch size, key len, hid dim]
        V = self.fc_v(value)  # V = [batch size, value len, hid dim]
        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)  # Q = [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)  # K = [batch size, n heads, key len, head dim]
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)  # V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        # energy = [batch size, n heads, query len, key len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # attention = [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim=-1)
        # x = [batch size, n heads, query len, head dim]
        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, query len, n heads, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)  # x = [batch size, query len, hid dim]
        return x, attention