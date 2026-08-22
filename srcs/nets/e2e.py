# Source (modified): https://github.com/mpc001/auto_avsr/blob/main/espnet/nets/pytorch_backend/e2e_asr_conformer.py
# License: Apache-2.0 (https://github.com/mpc001/auto_avsr/blob/main/LICENSE)

import torch
import torch.nn as nn

from srcs.nets.backend.ctc import CTC
from srcs.nets.backend.encoder.conformer_encoder import ConformerEncoder
from srcs.nets.backend.frontend.resnet import video_resnet
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.utils import ctc_decode, freeze


class VSRModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        attention_dim=768,
        attention_heads=12,
        linear_units=3072,
        num_blocks=12,
        dropout_rate=0.1,
        attention_dropout_rate=0.0,
        cnn_module_kernel=31,
        blank_id=0,
        ignore_id=-1,
    ):
        super().__init__()
        self.frontend = video_resnet()
        self.proj_encoder = nn.Linear(512, attention_dim)

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
            output_size=vocab_size,
            input_size=attention_dim,
            dropout_rate=dropout_rate,
            blank_id=blank_id,
            ignore_id=ignore_id,
        )

        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self._frozen_modules = []

    def finetune(self):
        blocks = self.encoder.encoders
        freeze(self)
        blocks[-2:].requires_grad_(True)
        self.ctc.ctc_lo.requires_grad_(True)

        self._frozen_modules = [
            self.frontend,
            self.proj_encoder,
            self.encoder.embed,
            *blocks[:-2],
            self.encoder.after_norm,
        ]

    def train(self, mode=True):
        super().train(mode)

        for module in self._frozen_modules:
            module.eval()

        return self

    def encode(self, videos, video_lengths, return_visual=False):
        # videos: [B, T, 1, H, W], video_lengths: [B]
        input_mask = (
            make_non_pad_mask(video_lengths).to(videos.device).unsqueeze(-2)
        )  # [B, 1, T]
        visual_features = self.frontend(videos)
        encoder_inputs = self.proj_encoder(visual_features)  # [B, T, D]
        encoder_features = self.encoder(encoder_inputs, input_mask)[0]  # [B, T, D]

        if return_visual:
            return encoder_features, video_lengths, visual_features

        return encoder_features, video_lengths

    def get_contexts(self, videos, video_lengths):
        encoder_features, input_lengths, visual_features = self.encode(
            videos, video_lengths, return_visual=True
        )
        _, logits = self.ctc(encoder_features, input_lengths)

        return {
            "visual_features": visual_features,
            "encoder_features": encoder_features,
            "logits": logits,
            "input_lengths": input_lengths,
        }

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        encoder_features, input_lengths = self.encode(videos, video_lengths)

        loss, logits = self.ctc(encoder_features, input_lengths, labels, label_lengths)

        return {"loss": loss, "logits": logits, "input_lengths": input_lengths}

    def decode(self, videos, video_lengths):
        output = self(videos, video_lengths)
        return ctc_decode(output["logits"], output["input_lengths"], self.blank_id)


def get_model(model: str, vocab_size: int, size: str = "large", **model_config):
    if model != "auto-vsr":
        raise ValueError("model must be 'auto-vsr'.")

    block_counts = {"large": 12, "small": 6}

    if size not in block_counts:
        raise ValueError("size must be 'large' or 'small'.")

    return VSRModel(vocab_size=vocab_size, num_blocks=block_counts[size], **model_config)
