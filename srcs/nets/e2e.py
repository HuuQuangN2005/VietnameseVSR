# Source (modified): https://github.com/mpc001/auto_avsr/blob/main/espnet/nets/pytorch_backend/e2e_asr_conformer.py
# License: Apache-2.0 (https://github.com/mpc001/auto_avsr/blob/main/LICENSE)

import torch
import torch.nn as nn

from srcs.nets.backend.ctc import CTC
from srcs.nets.backend.encoder.conformer_encoder import ConformerEncoder
from srcs.nets.backend.frontend.resnet import video_resnet
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.backend.refiner.refiner import Refiner
from srcs.nets.utils import load_weights


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

        self._contexts = None
        self.encoder.encoders[1].register_forward_hook(
            self._make_context_hook("h2_features")
        )
        self.encoder.encoders[3].register_forward_hook(
            self._make_context_hook("h4_features")
        )

    def _make_context_hook(self, name):
        def hook(_module, _inputs, output):
            if self._contexts is None:
                return

            hidden = output[0]
            if isinstance(hidden, tuple):
                hidden = hidden[0]

            self._contexts[name] = hidden

        return hook

    def encode(self, videos, video_lengths):
        input_mask = make_non_pad_mask(video_lengths).to(videos.device).unsqueeze(1)
        visual_features = self.frontend(videos)
        encoder_inputs = self.proj_encoder(visual_features)
        encoder_features = self.encoder(encoder_inputs, input_mask)[0]

        return encoder_features, visual_features

    @torch.no_grad()
    def get_contexts(self, videos, video_lengths):
        self._contexts = {}

        try:
            encoder_features, visual_features = self.encode(videos, video_lengths)
            _, logits = self.ctc(encoder_features, video_lengths)

            visual_contexts = {
                "visual_features": visual_features,
                "h2_features": self._contexts["h2_features"],
                "h4_features": self._contexts["h4_features"],
                "input_lengths": video_lengths,
            }
            return logits, visual_contexts
        finally:
            self._contexts = None

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        encoder_features, _ = self.encode(videos, video_lengths)
        loss, logits = self.ctc(encoder_features, video_lengths, labels, label_lengths)

        return {"loss": loss, "logits": logits, "input_lengths": video_lengths}


def get_model(model: str, vocab_size: int, checkpoint=None, **model_config):
    if model == "auto-vsr":
        network = VSRModel(vocab_size=vocab_size, **model_config)
    elif model == "refiner":
        network = Refiner(vocab_size=vocab_size, **model_config)
    else:
        raise ValueError("model must be 'auto-vsr' or 'refiner'.")

    if checkpoint:
        load_weights(network, checkpoint)

    return network
