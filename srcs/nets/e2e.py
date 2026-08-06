# Source (modified): https://github.com/mpc001/auto_avsr/blob/main/espnet/nets/pytorch_backend/e2e_asr_conformer.py
# License: Apache-2.0 (https://github.com/mpc001/auto_avsr/blob/main/LICENSE)

from torch import nn
from transformers import PretrainedConfig

from srcs.nets.backend.ctc import CTC
from srcs.nets.backend.encoder.conformer_encoder import ConformerEncoder
from srcs.nets.backend.frontend.resnet import video_resnet
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.utils import ctc_decode, freeze, load_weights


class VSRConfig(PretrainedConfig):
    model_type = "vsr_conformer"

    def __init__(
        self,
        vocab_size=0,
        attention_dim=256,
        attention_heads=4,
        linear_units=1024,
        num_blocks=4,
        dropout_rate=0.1,
        attention_dropout_rate=0.0,
        cnn_module_kernel=31,
        blank_id=0,
        ignore_id=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.cnn_module_kernel = cnn_module_kernel
        self.blank_id = blank_id
        self.ignore_id = ignore_id


class Model(nn.Module):
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

        self.config = VSRConfig(
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            cnn_module_kernel=cnn_module_kernel,
            blank_id=blank_id,
            ignore_id=ignore_id,
        )

    def freeze_frontend(self):
        freeze(self.frontend)
        self.frontend_frozen = True

    def train(self, mode=True):
        super().train(mode)
        if self.frontend_frozen:
            self.frontend.eval()
        return self

    def encode(self, videos, video_lengths):
        mask = make_non_pad_mask(video_lengths).to(videos.device).unsqueeze(-2)
        encoded_features = self.proj_encoder(self.frontend(videos))
        encoded_features = self.encoder(encoded_features, mask)[0]
        return encoded_features, video_lengths

    def forward(self, videos, video_lengths, labels=None, label_lengths=None, **kwargs):
        del kwargs
        encoded_features, input_lengths = self.encode(videos, video_lengths)
        loss, logits = self.ctc(encoded_features, input_lengths, labels, label_lengths)
        return {"loss": loss, "logits": logits, "input_lengths": input_lengths}

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
    if model_name != "e2e":
        raise ValueError(f"Unsupported model: {model_name}")
    model = Model(vocab_size, **model_config)
    if pretrained_weights:
        load_weights(model, pretrained_weights, "frontend")
    if checkpoint_path:
        load_weights(model, checkpoint_path)
    if freeze_frontend:
        if not pretrained_weights and not checkpoint_path:
            raise ValueError("Frontend cannot be frozen before weights are loaded.")
        model.freeze_frontend()
    return model
