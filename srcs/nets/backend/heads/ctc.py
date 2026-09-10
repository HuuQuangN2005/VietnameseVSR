import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(self, input_size, vocab_size, dropout=0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_size, vocab_size)

    def forward(self, inputs):
        return self.linear1(self.dropout1(inputs))
