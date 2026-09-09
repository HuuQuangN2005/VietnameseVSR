import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.in_layer = nn.Linear(hidden_dim, ffn_dim)
        self.out_layer = nn.Linear(ffn_dim, hidden_dim)

        self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.in_layer(inputs)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        return x
