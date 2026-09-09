import torch.nn as nn


class IndependentMCTCHead(nn.Module):
    def __init__(
        self,
        input_size,
        initial_size,
        rhyme_size,
        tone_size,
        dropout=0.1,
    ):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_size, 1)

        self.linear2 = nn.Linear(input_size, initial_size)
        self.linear3 = nn.Linear(input_size, rhyme_size)
        self.linear4 = nn.Linear(input_size, tone_size)

    def forward(self, inputs):
        inputs = self.dropout1(inputs)

        blank = self.linear1(inputs)

        initial = self.linear2(inputs)
        rhyme = self.linear3(inputs)
        tone = self.linear4(inputs)

        return {
            "blank": blank.squeeze(-1),
            "initial": initial,
            "rhyme": rhyme,
            "tone": tone,
        }


class CascadedMCTCHead(nn.Module):
    def __init__(
        self,
        input_size,
        initial_size,
        rhyme_size,
        tone_size,
        dropout=0.1,
    ):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_size, 1)
        self.linear2 = nn.Linear(input_size, initial_size)

        self.act1 = nn.Softmax(dim=-1)
        self.linear3 = nn.Linear(initial_size, input_size, bias=False)
        self.norm1 = nn.LayerNorm(input_size)

        self.linear4 = nn.Linear(input_size, rhyme_size)

        self.act2 = nn.Softmax(dim=-1)
        self.linear5 = nn.Linear(rhyme_size, input_size, bias=False)
        self.norm2 = nn.LayerNorm(input_size)

        self.linear6 = nn.Linear(input_size, tone_size)

    def forward(self, inputs):
        inputs = self.dropout1(inputs)

        blank = self.linear1(inputs)

        initial = self.linear2(inputs)
        initial_guide = self.linear3(self.act1(initial))
        rhyme_inputs = self.norm1(inputs + initial_guide)

        rhyme = self.linear4(rhyme_inputs)
        rhyme_guide = self.linear5(self.act2(rhyme))
        tone_inputs = self.norm2(inputs + rhyme_guide)

        tone = self.linear6(tone_inputs)

        return {
            "blank": blank.squeeze(-1),
            "initial": initial,
            "rhyme": rhyme,
            "tone": tone,
        }


class RhymeGuidedMCTCHead(nn.Module):
    def __init__(
        self,
        input_size,
        initial_size,
        rhyme_size,
        tone_size,
        dropout=0.1,
    ):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_size, 1)

        self.linear2 = nn.Linear(input_size, rhyme_size)
        self.act1 = nn.Softmax(dim=-1)

        self.linear3 = nn.Linear(rhyme_size, input_size, bias=False)
        self.norm1 = nn.LayerNorm(input_size)

        self.linear4 = nn.Linear(input_size, initial_size)
        self.linear5 = nn.Linear(input_size, tone_size)

    def forward(self, inputs):
        inputs = self.dropout1(inputs)

        blank = self.linear1(inputs)

        rhyme = self.linear2(inputs)
        rhyme_guide = self.linear3(self.act1(rhyme))
        guided_inputs = self.norm1(inputs + rhyme_guide)

        initial = self.linear4(guided_inputs)
        tone = self.linear5(guided_inputs)

        return {
            "blank": blank.squeeze(-1),
            "initial": initial,
            "rhyme": rhyme,
            "tone": tone,
        }
