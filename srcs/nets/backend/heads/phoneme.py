import torch.nn as nn


class PhonemeHead(nn.Module):
    def __init__(self, hidden_dim, initial_size, rhyme_size, tone_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.act1 = nn.ReLU()

        self.linear2 = nn.Linear(hidden_dim, rhyme_size)
        self.act2 = nn.Softmax(dim=-1)
        self.linear3 = nn.Linear(rhyme_size, hidden_dim, bias=False)

        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.linear5 = nn.Linear(hidden_dim, initial_size)
        self.linear6 = nn.Linear(hidden_dim, tone_size)

    def forward(self, inputs):
        hidden = self.act1(self.norm1(self.linear1(inputs)))

        rhyme = self.linear2(hidden)
        context = self.linear3(self.act2(rhyme))

        fused = self.dropout1(self.linear4(hidden + context))
        conditioned = self.norm2(inputs + fused)

        return {
            "initial": self.linear5(conditioned),
            "rhyme": rhyme,
            "tone": self.linear6(conditioned),
        }
