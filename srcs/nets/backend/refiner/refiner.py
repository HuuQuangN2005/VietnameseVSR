import torch
import torch.nn as nn

from srcs.nets.backend.decoder.transformer_decoder import TransformerDecoder
from srcs.nets.backend.nets_utils import make_non_pad_mask


class VisualEvidenceCTCRefiner(nn.Module):
    def __init__(
        self,
        vocab_size,
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
        correction_init_std=1e-3,
    ):
        super().__init__()

        if not 0.0 <= corruption_rate <= 1.0:
            raise ValueError("corruption_rate must be in [0, 1].")

        if not 0.0 <= corruption_sample_probability <= 1.0:
            raise ValueError("corruption_sample_probability must be in [0, 1].")

        if not 0.0 <= top1_suppression <= 1.0:
            raise ValueError("top1_suppression must be in [0, 1].")

        self.corruption_rate = corruption_rate
        self.corruption_sample_probability = corruption_sample_probability
        self.top1_suppression = top1_suppression

        self.posterior_proj = nn.Linear(vocab_size, d_model)

        self.decoder = TransformerDecoder(
            odim=vocab_size,
            attention_dim=d_model,
            attention_heads=num_heads,
            linear_units=ffn_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            self_attention_dropout_rate=attention_dropout_rate,
            src_attention_dropout_rate=attention_dropout_rate,
            input_layer=nn.Identity(),
            use_output_layer=False,
        )

        self.correction_head = nn.Linear(d_model, vocab_size)

        nn.init.normal_(self.correction_head.weight, mean=0.0, std=correction_init_std)
        nn.init.zeros_(self.correction_head.bias)

    def _corrupt_posterior(self, posterior, valid_mask):
        if (
            not self.training
            or self.corruption_rate == 0.0
            or self.corruption_sample_probability == 0.0
        ):
            return posterior

        batch_size = posterior.size(0)

        sample_mask = (
            torch.rand(batch_size, device=posterior.device)
            < self.corruption_sample_probability
        )

        frame_mask = (
            torch.rand(valid_mask.shape, device=posterior.device) < self.corruption_rate
        )

        corrupt_mask = valid_mask & frame_mask & sample_mask.unsqueeze(1)

        if not corrupt_mask.any():
            return posterior

        top2 = posterior.topk(2, dim=-1)

        top1_ids = top2.indices[..., 0:1]
        top2_ids = top2.indices[..., 1:2]

        top1_prob = top2.values[..., 0:1]
        top2_prob = top2.values[..., 1:2]

        new_top1 = top1_prob * self.top1_suppression
        new_top2 = top2_prob + (top1_prob - new_top1)

        corrupted = posterior.scatter(-1, top1_ids, new_top1)
        corrupted = corrupted.scatter(-1, top2_ids, new_top2)

        return torch.where(corrupt_mask.unsqueeze(-1), corrupted, posterior)

    def forward(self, logits, visual_features, input_lengths, visual_lengths=None):
        valid_mask = make_non_pad_mask(input_lengths).to(logits.device)

        if visual_lengths is None:
            visual_lengths = input_lengths

        visual_mask = make_non_pad_mask(visual_lengths).to(logits.device)

        posterior = torch.softmax(logits, dim=-1)
        posterior = self._corrupt_posterior(posterior, valid_mask)
        posterior_features = self.posterior_proj(posterior)

        reconsidered_features, _ = self.decoder(
            posterior_features,
            valid_mask.unsqueeze(1),
            visual_features,
            visual_mask.unsqueeze(1),
        )

        correction_logits = self.correction_head(reconsidered_features)
        refined_logits = logits + correction_logits

        with torch.no_grad():
            initial_ids = logits.argmax(dim=-1)
            refined_ids = refined_logits.argmax(dim=-1)

            refined_flip_rate = (
                refined_ids[valid_mask].ne(initial_ids[valid_mask]).float().mean()
            )

        return {"logits": refined_logits, "refined_flip_rate": refined_flip_rate}
