# Source (modified): https://github.com/mpc001/auto_avsr/blob/main/espnet/nets/pytorch_backend/e2e_asr_conformer.py
# License: Apache-2.0 (https://github.com/mpc001/auto_avsr/blob/main/LICENSE)

import torch
import torch.nn as nn

from srcs.nets.backend.ctc import CTC, WERGuidedCTCLoss
from srcs.nets.backend.encoder.conformer_encoder import ConformerEncoder
from srcs.nets.backend.frontend.resnet import video_resnet
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.backend.refiner.refiner import VisualEvidenceCTCRefiner
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
        visual_features = self.frontend(videos)
        visual_features = self.proj_encoder(visual_features)  # [B, T, D]
        encoded_features = self.encoder(visual_features, mask)[0]  # [B, T, D]

        if return_visual:
            return encoded_features, video_lengths, visual_features

        return encoded_features, video_lengths

    def get_contexts(self, videos, video_lengths):
        encoded_features, input_lengths, _ = self.encode(
            videos, video_lengths, return_visual=True
        )
        _, logits = self.ctc(encoded_features, input_lengths)

        return {
            "visual_features": encoded_features,
            "encoded_features": encoded_features,
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
        d_model=256,
        num_heads=4,
        ffn_dim=1024,
        num_blocks=1,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        corruption_rate=0.1,
        corruption_sample_probability=0.5,
        top1_suppression=0.25,
        wer_weight=1.0,
        text_transform=None,
        checkpoint_dir=None,
        freeze_baseline=True,
    ):
        super().__init__()
        self.vsr_model = vsr_model
        self.refiner = VisualEvidenceCTCRefiner(
            vocab_size=vsr_model.vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            corruption_rate=corruption_rate,
            corruption_sample_probability=corruption_sample_probability,
            top1_suppression=top1_suppression,
        )
        if text_transform is None:
            raise ValueError("text_transform is required for WER-guided CTC loss.")

        self.loss_fn = WERGuidedCTCLoss(text_transform, wer_weight)
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

        refined = self.refiner(
            logits=contexts["logits"],
            visual_features=contexts["visual_features"],
            input_lengths=contexts["input_lengths"],
        )
        loss = None
        raw_ctc_loss = None

        if labels is not None:
            loss_output = self.loss_fn(
                self.vsr_model.ctc,
                refined["logits"],
                contexts["input_lengths"],
                labels,
                label_lengths,
            )
            loss = loss_output["loss"]
            raw_ctc_loss = loss_output["raw_ctc_loss"]

        return {
            "loss": loss,
            "logits": refined["logits"],
            "input_lengths": contexts["input_lengths"],
            "first_logits": contexts["logits"],
            "raw_ctc_loss": raw_ctc_loss,
            "refined_flip_rate": refined["refined_flip_rate"],
        }

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
