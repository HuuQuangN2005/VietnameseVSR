import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from srcs.nets.backend.heads.phoneme import PhonemeHead
from srcs.nets.backend.nets_utils import (
    COMP_NAMES,
    make_causal_mask,
    make_non_pad_mask,
)
from srcs.nets.loss.ce import PhonemeCELoss
from srcs.nets.backend.transformer.embedding import PositionalEncoding
from srcs.nets.backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


class SyllabicDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs, memory, causal_mask, memory_mask, target_mask):
        normalized = self.norm1(inputs)
        attn, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=None if target_mask is None else ~target_mask,
            need_weights=False,
        )
        hidden = inputs + self.dropout1(attn)

        normalized = self.norm2(hidden)
        attn, _ = self.cross_attn(
            normalized,
            memory,
            memory,
            key_padding_mask=None if memory_mask is None else ~memory_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout2(attn)

        return hidden + self.dropout3(self.feed_forward(self.norm3(hidden)))


class SyllabicDecoder(nn.Module):
    ignore_id = -1

    def __init__(
        self,
        hidden_dim,
        vocab_size,
        num_layers=6,
        num_heads=4,
        ffn_dim=1024,
        dropout=0.1,
        label_smoothing=0.1,
        max_length=64,
    ):
        super().__init__()
        self.vocab_size = {name: vocab_size[name] for name in COMP_NAMES}
        self.eos_id = {name: size for name, size in self.vocab_size.items()}
        self.sos_id = {name: size + 1 for name, size in self.vocab_size.items()}

        self.max_length = max_length

        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size + 2, hidden_dim)
                for name, size in self.vocab_size.items()
            }
        )
        self.projection = nn.Linear(len(COMP_NAMES) * hidden_dim, hidden_dim)
        self.position = PositionalEncoding(hidden_dim, dropout, max_length + 2)
        self.layers = nn.ModuleList(
            [
                SyllabicDecoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.loss_fn = PhonemeCELoss(label_smoothing)

        self.head = PhonemeHead(
            hidden_dim,
            initial_size=self.vocab_size["initial"] + 1,
            rhyme_size=self.vocab_size["rhyme"] + 1,
            tone_size=self.vocab_size["tone"] + 1,
            dropout=dropout,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() > 1 and not name.startswith("embeddings."):
                nn.init.xavier_uniform_(param)

    def forward(self, tokens, memory, memory_mask, target_lengths=None):
        embedded = torch.cat(
            [
                self.embeddings[name](tokens[..., index])
                for index, name in enumerate(COMP_NAMES)
            ],
            dim=-1,
        )

        hidden = self.projection(embedded) * math.sqrt(self.hidden_dim)
        hidden = self.position(hidden)

        length = tokens.size(1)
        causal_mask = make_causal_mask(length, tokens.device)
        target_mask = None

        if target_lengths is not None:
            target_mask = make_non_pad_mask(target_lengths.to(tokens.device), length)

        for layer in self.layers:
            hidden = layer(hidden, memory, causal_mask, memory_mask, target_mask)

        return self.head(self.norm(hidden))

    def build_targets(self, labels, label_lengths):
        device = labels.device
        sos = torch.tensor([self.sos_id[name] for name in COMP_NAMES], device=device)
        eos = torch.tensor([self.eos_id[name] for name in COMP_NAMES], device=device)

        positions = torch.arange(labels.size(1) + 1, device=device).view(1, -1, 1)
        lengths = label_lengths.to(device).view(-1, 1, 1)

        inputs = torch.cat(
            [sos.expand(labels.size(0), 1, -1), labels.clamp(min=0)],
            dim=1,
        )

        targets = F.pad(labels, (0, 0, 0, 1), value=self.ignore_id)
        targets = torch.where(positions == lengths, eos, targets)

        return inputs, targets

    def compute_loss(self, memory, memory_mask, labels, label_lengths):
        inputs, targets = self.build_targets(labels, label_lengths)
        logits = self.forward(inputs, memory, memory_mask, label_lengths + 1)

        return self.loss_fn(logits, targets), logits

    @staticmethod
    @torch.no_grad()
    def teacher_forced_preds(logits, label_lengths):
        predicted = torch.stack(
            [logits[name].argmax(dim=-1) for name in COMP_NAMES],
            dim=-1,
        ).cpu()
        lengths = label_lengths.cpu()

        return [
            predicted[index, : lengths[index]].tolist()
            for index in range(predicted.size(0))
        ]

    @torch.no_grad()
    def generate(self, memory, memory_mask):
        batch_size = memory.size(0)
        device = memory.device

        sos = torch.tensor([self.sos_id[name] for name in COMP_NAMES], device=device)
        eos = torch.tensor([self.eos_id[name] for name in COMP_NAMES], device=device)

        tokens = sos.expand(batch_size, 1, -1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        for _ in range(self.max_length):
            logits = self.forward(tokens, memory, memory_mask)
            step = torch.stack(
                [logits[name][:, -1].argmax(dim=-1) for name in COMP_NAMES],
                dim=-1,
            )

            finished = finished | (step == eos).any(dim=-1)
            lengths = lengths + (~finished).long()

            if bool(finished.all()):
                break

            tokens = torch.cat([tokens, step.unsqueeze(1)], dim=1)

        preds = tokens[:, 1:].cpu()
        lengths = lengths.cpu()

        return [preds[index, : lengths[index]].tolist() for index in range(batch_size)]
