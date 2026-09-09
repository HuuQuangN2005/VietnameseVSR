import torch.nn as nn

from srcs.nets.backend.frontend.resnet import video_resnet
from srcs.nets.backend.heads.mctc import (
    CascadedMCTCHead,
    IndependentMCTCHead,
    RhymeGuidedMCTCHead,
)
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.backend.transformer.encoder import TransformerEncoder
from srcs.nets.loss.mctc import MCTCWELoss
from srcs.nets.utils import load_weights


class VisualEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        ffn_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.output_size = hidden_dim
        self.frontend = video_resnet()
        self.projection = nn.Linear(512, hidden_dim)
        self.transformer = TransformerEncoder(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, videos, video_lengths):
        features = self.projection(self.frontend(videos))
        valid_mask = make_non_pad_mask(
            video_lengths.to(features.device),
            features.size(1),
        )
        features = features * valid_mask.unsqueeze(-1)

        return self.transformer(features, valid_mask)


class IndependentMCTCVSR(nn.Module):
    def __init__(self, vocab_size, dropout=0.1, **encoder_config):
        super().__init__()
        self.encoder = VisualEncoder(dropout=dropout, **encoder_config)
        self.head = IndependentMCTCHead(
            input_size=self.encoder.output_size,
            initial_size=vocab_size["initial"],
            rhyme_size=vocab_size["rhyme"],
            tone_size=vocab_size["tone"],
            dropout=dropout,
        )
        self.loss_fn = MCTCWELoss()

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        features = self.encoder(videos, video_lengths)
        logits = self.head(features)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels, video_lengths, label_lengths)

        return {
            "loss": loss,
            "logits": logits,
            "input_lengths": video_lengths,
        }


class CascadedMCTCVSR(nn.Module):
    def __init__(self, vocab_size, dropout=0.1, **encoder_config):
        super().__init__()
        self.encoder = VisualEncoder(dropout=dropout, **encoder_config)
        self.head = CascadedMCTCHead(
            input_size=self.encoder.output_size,
            initial_size=vocab_size["initial"],
            rhyme_size=vocab_size["rhyme"],
            tone_size=vocab_size["tone"],
            dropout=dropout,
        )
        self.loss_fn = MCTCWELoss()

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        features = self.encoder(videos, video_lengths)
        logits = self.head(features)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels, video_lengths, label_lengths)

        return {
            "loss": loss,
            "logits": logits,
            "input_lengths": video_lengths,
        }


class RhymeGuidedMCTCVSR(nn.Module):
    def __init__(self, vocab_size, dropout=0.1, **encoder_config):
        super().__init__()
        self.encoder = VisualEncoder(dropout=dropout, **encoder_config)
        self.head = RhymeGuidedMCTCHead(
            input_size=self.encoder.output_size,
            initial_size=vocab_size["initial"],
            rhyme_size=vocab_size["rhyme"],
            tone_size=vocab_size["tone"],
            dropout=dropout,
        )
        self.loss_fn = MCTCWELoss()

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        features = self.encoder(videos, video_lengths)
        logits = self.head(features)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels, video_lengths, label_lengths)

        return {
            "loss": loss,
            "logits": logits,
            "input_lengths": video_lengths,
        }


def get_model(model, vocab_size, checkpoint=None, **model_config):
    model_classes = {
        "IndependentMCTCVSR": IndependentMCTCVSR,
        "CascadedMCTCVSR": CascadedMCTCVSR,
        "RhymeGuidedMCTCVSR": RhymeGuidedMCTCVSR,
    }
    network = model_classes[model](vocab_size=vocab_size, **model_config)

    if checkpoint:
        load_weights(network, checkpoint)

    return network
