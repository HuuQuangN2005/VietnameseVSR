import torch.nn as nn

from srcs.nets.backend.frontend.shufflenet import video_shufflenet
from srcs.nets.backend.decoder.syllabic import SyllabicDecoder
from srcs.nets.backend.heads.mctc import MCTCHead
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.backend.TCN import TCN
from srcs.nets.loss.mctc import MCTCWELoss
from srcs.nets.utils import load_weights


class VisualEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=6,
        kernel_size=3,
        dropout=0.1,
        visual_pretrained=None,
    ):
        super().__init__()
        self.output_size = hidden_dim
        self.frontend = video_shufflenet(visual_pretrained)
        self.projection = nn.Linear(self.frontend.output_size, hidden_dim)
        self.tcn = TCN(
            num_inputs=hidden_dim,
            num_channels=[hidden_dim] * num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, videos, video_lengths):
        features = self.projection(self.frontend(videos))
        valid_mask = make_non_pad_mask(
            video_lengths.to(features.device),
            features.size(1),
        )
        features = features * valid_mask.unsqueeze(-1)

        return self.tcn(features, valid_mask)


class MCTCVSR(nn.Module):
    def __init__(self, vocab_size, dropout=0.1, **encoder_config):
        super().__init__()
        self.encoder = VisualEncoder(dropout=dropout, **encoder_config)
        self.head = MCTCHead(
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


class DecoderVSR(nn.Module):
    def __init__(
        self,
        vocab_size,
        dropout=0.1,
        ctc_weight=0.3,
        decoder_layers=2,
        decoder_heads=4,
        decoder_ffn_dim=512,
        label_smoothing=0.1,
        **encoder_config,
    ):
        super().__init__()
        self.encoder = VisualEncoder(dropout=dropout, **encoder_config)
        self.head = None
        self.loss_fn = None

        if ctc_weight > 0.0:
            self.head = MCTCHead(
                input_size=self.encoder.output_size,
                initial_size=vocab_size["initial"],
                rhyme_size=vocab_size["rhyme"],
                tone_size=vocab_size["tone"],
                dropout=dropout,
            )
            self.loss_fn = MCTCWELoss()

        self.decoder = SyllabicDecoder(
            hidden_dim=self.encoder.output_size,
            vocab_size=vocab_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            label_smoothing=label_smoothing,
        )
        self.ctc_weight = ctc_weight

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        features = self.encoder(videos, video_lengths)

        memory_mask = make_non_pad_mask(
            video_lengths.to(features.device),
            features.size(1),
        )

        outputs = {
            "loss": None,
            "input_lengths": video_lengths,
        }

        if self.head is not None:
            outputs["logits"] = self.head(features)

        if labels is not None:
            ce_loss, decoder_logits = self.decoder.compute_loss(
                features,
                memory_mask,
                labels,
                label_lengths,
            )
            loss = (1.0 - self.ctc_weight) * ce_loss

            if self.head is not None:
                loss = loss + self.ctc_weight * self.loss_fn(
                    outputs["logits"],
                    labels,
                    video_lengths,
                    label_lengths,
                )

            outputs["loss"] = loss

            if self.training:
                outputs["preds"] = self.decoder.teacher_forced_preds(
                    decoder_logits,
                    label_lengths,
                )

        if not self.training:
            outputs["preds"] = self.decoder.generate(features, memory_mask)

        return outputs


def get_model(model, vocab_size, ckpt=None, **model_config):
    model_classes = {
        "MCTCVSR": MCTCVSR,
        "DecoderVSR": DecoderVSR,
    }
    network = model_classes[model](vocab_size=vocab_size, **model_config)

    if ckpt:
        load_weights(network, ckpt)

    return network
