#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Source repository: https://github.com/mpc001/auto_avsr
# Source path: espnet/nets/pytorch_backend/e2e_asr_conformer.py

"""CTC-only visual speech recognition model adapted from Auto-AVSR."""

import torch

from srcs.nets.pytorch_backend.ctc import CTC
from srcs.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
from srcs.nets.pytorch_backend.frontend.resnet import video_resnet
from srcs.nets.pytorch_backend.nets_utils import make_non_pad_mask


class E2ECommon(torch.nn.Module):
    """Auto-AVSR visual frontend and Conformer encoder with a CTC head."""

    def __init__(
        self,
        odim: int,
        attention_dim: int = 256,
        attention_heads: int = 4,
        linear_units: int = 1024,
        num_blocks: int = 4,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        cnn_module_kernel: int = 31,
        ignore_id: int = -1,
    ) -> None:
        super().__init__()
        self.frontend = video_resnet()
        self.proj_encoder = torch.nn.Linear(512, attention_dim)
        self.encoder = ConformerEncoder(
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            cnn_module_kernel=cnn_module_kernel,
        )
        self.ctc = CTC(
            odim=odim, eprojs=attention_dim, dropout_rate=dropout_rate, reduce=True
        )
        self.ctc.ignore_id = ignore_id
        self.blank = 0
        self.odim = odim
        self.ignore_id = ignore_id

    def encode(
        self, videos: torch.Tensor, video_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode padded videos and return their padding mask."""
        padding_mask = make_non_pad_mask(video_lengths).to(videos.device).unsqueeze(-2)
        encoded_videos = self.frontend(videos)
        encoded_videos = self.proj_encoder(encoded_videos)
        return self.encoder(encoded_videos, padding_mask)

    def forward(
        self,
        videos: torch.Tensor,
        video_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CTC loss and uncollapsed logits shaped (B, T, V)."""
        del label_lengths
        encoded_videos, _ = self.encode(videos, video_lengths)
        loss, logits = self.ctc(encoded_videos, video_lengths, labels)
        return loss, logits.transpose(0, 1)
