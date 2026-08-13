import torch
import torch.nn as nn

from srcs.nets.backend.decoder.transformer_decoder import TransformerDecoder
from srcs.nets.backend.nets_utils import make_non_pad_mask


class VisualEvidenceCTCRefiner(nn.Module):
    def __init__(
        self,
        vocab_size,
        attention_dim=256,
        attention_heads=4,
        linear_units=1024,
        num_blocks=1,
        dropout_rate=0.1,
        attention_dropout_rate=0.0,
        correction_init_std=1e-3,
    ):
        super().__init__()

        self.posterior_proj = nn.Linear(vocab_size, attention_dim)

        self.decoder = TransformerDecoder(
            odim=vocab_size,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            self_attention_dropout_rate=attention_dropout_rate,
            src_attention_dropout_rate=attention_dropout_rate,
            input_layer=nn.Identity(),
            use_output_layer=False,
        )

        self.correction_head = nn.Linear(attention_dim, vocab_size)

        nn.init.normal_(self.correction_head.weight, mean=0.0, std=correction_init_std)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, logits, visual_features, input_lengths):
        if logits.shape[:2] != visual_features.shape[:2]:
            raise ValueError(
                "logits and visual_features must have the same batch and time dimensions."
            )

        input_mask = make_non_pad_mask(input_lengths, logits[..., 0])
        attention_mask = input_mask.unsqueeze(1)

        posterior = torch.softmax(logits, dim=-1)
        posterior_features = self.posterior_proj(posterior)

        reconsidered_features, _ = self.decoder(
            posterior_features, attention_mask, visual_features, attention_mask
        )

        correction_logits = self.correction_head(reconsidered_features)
        refined_logits = logits + correction_logits

        with torch.no_grad():
            initial_ids = logits.argmax(dim=-1)
            refined_ids = refined_logits.argmax(dim=-1)
            refined_flip_rate = (
                refined_ids[input_mask].ne(initial_ids[input_mask]).float().mean()
            )

        return {"logits": refined_logits, "refined_flip_rate": refined_flip_rate}
