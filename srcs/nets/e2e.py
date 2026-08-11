# Source (modified): https://github.com/mpc001/auto_avsr/blob/main/espnet/nets/pytorch_backend/e2e_asr_conformer.py
# License: Apache-2.0 (https://github.com/mpc001/auto_avsr/blob/main/LICENSE)

import torch
import torch.nn as nn

from srcs.nets.backend.ctc import CTC
from srcs.nets.backend.encoder.conformer_encoder import ConformerEncoder
from srcs.nets.backend.frontend.resnet import video_resnet
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.backend.refiner import TemperedVisualCTCRefiner
from srcs.nets.utils import ctc_decode, freeze, load_weights


class VSRModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        attention_dim=256,
        attention_heads=4,
        linear_units=1024,
        num_blocks=4,
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
            vocab_size,
            attention_dim,
            dropout_rate,
            blank_id=blank_id,
            ignore_id=ignore_id,
        )

        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.ignore_id = ignore_id
        self.frontend_frozen = False

    def freeze_frontend(self):
        freeze(self.frontend)
        self.frontend_frozen = True

    def train(self, mode=True):
        super().train(mode)

        if self.frontend_frozen:
            self.frontend.eval()

        return self

    def encode(self, videos, video_lengths, return_visual=False):
        # videos: [B, T, 1, H, W], video_lengths: [B]
        mask = (
            make_non_pad_mask(video_lengths).to(videos.device).unsqueeze(-2)
        )  # [B, 1, T]
        visual_feats = self.frontend(videos)
        encoded_features = self.proj_encoder(visual_feats)  # [B, T, D]
        encoded_features = self.encoder(encoded_features, mask)[0]  # [B, T, D]

        if return_visual:
            return encoded_features, video_lengths, visual_feats

        return encoded_features, video_lengths

    def get_contexts(self, videos, video_lengths):
        encoded_features, input_lengths, visual_feats = self.encode(
            videos, video_lengths, return_visual=True
        )
        _, logits = self.ctc(encoded_features, input_lengths)

        return {
            "visual_feats": visual_feats,
            "logits": logits,
            "input_lengths": input_lengths,
        }

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        encoded_features, input_lengths = self.encode(videos, video_lengths)
        loss, logits = self.ctc(encoded_features, input_lengths, labels, label_lengths)

        return {
            "loss": loss,
            "logits": logits,
            "input_lengths": input_lengths,
        }  # loss: scalar, logits: [B, T, V], input_lengths: [B]

    def decode(self, videos, video_lengths):
        output = self(videos, video_lengths)
        return ctc_decode(
            output["logits"], output["input_lengths"], self.blank_id
        )  # [B, T]


class VisualRefinerVSRModel(nn.Module):
    def __init__(
        self,
        vsr_model,
        visual_dim=512,
        attn_dim=256,
        num_heads=4,
        dropout=0.1,
        temperature=2.0,
        checkpoint_dir=None,
        freeze_baseline=True,
    ):
        super().__init__()
        self.vsr_model = vsr_model
        self.refiner = TemperedVisualCTCRefiner(
            vocab_size=vsr_model.vocab_size,
            visual_dim=visual_dim,
            attn_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            temperature=temperature,
        )
        self.blank_id = vsr_model.blank_id
        self.freeze_baseline = freeze_baseline

        if checkpoint_dir is not None:
            info = load_weights(self, checkpoint_dir)

            if info["missing"]:
                raise RuntimeError(
                    f"Full checkpoint is missing {len(info['missing'])} tensors."
                )

        if freeze_baseline:
            freeze(self.vsr_model)
        else:
            self.vsr_model.requires_grad_(True)

            if self.vsr_model.frontend_frozen:
                freeze(self.vsr_model.frontend)

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_baseline:
            self.vsr_model.eval()

        return self

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        if self.freeze_baseline:
            with torch.no_grad():
                contexts = self.vsr_model.get_contexts(videos, video_lengths)
        else:
            contexts = self.vsr_model.get_contexts(videos, video_lengths)

        visual_feats = contexts["visual_feats"]
        input_lengths = contexts["input_lengths"]

        positions = torch.arange(visual_feats.size(1), device=visual_feats.device)
        padding_mask = positions.unsqueeze(0) >= input_lengths.to(
            visual_feats.device
        ).unsqueeze(1)

        refined = self.refiner(
            contexts["logits"], visual_feats, padding_mask=padding_mask
        )

        loss = None

        if labels is not None:
            loss = self.vsr_model.ctc.loss_from_logits(
                refined["logits"], input_lengths, labels, label_lengths
            )

        return {"loss": loss, "logits": refined["logits"], "input_lengths": input_lengths}

    def decode(self, videos, video_lengths):
        output = self(videos, video_lengths)
        return ctc_decode(output["logits"], output["input_lengths"], self.blank_id)


def get_model(
    model_name,
    vocab_size,
    pretrained_weights=None,
    checkpoint_path=None,
    freeze_frontend=False,
    **model_config,
):
    valid_models = ["baseline"]

    if model_name not in valid_models:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name == "baseline":
        model = VSRModel(vocab_size, **model_config)

    if pretrained_weights:
        load_weights(model, pretrained_weights, "frontend")

    if checkpoint_path:
        load_weights(model, checkpoint_path)

    if freeze_frontend:
        model.freeze_frontend()

    return model
