import torch
import torch.nn as nn


class TemperedVisualCTCRefiner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        visual_dim: int = 512,
        attn_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 2.0,
    ):
        super().__init__()

        if temperature <= 0.0:
            raise ValueError("temperature must be greater than zero.")

        self.temperature = temperature

        self.query_proj = nn.Linear(vocab_size, attn_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            kdim=visual_dim,
            vdim=visual_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(attn_dim * 2, attn_dim),
            nn.GELU(),
            nn.LayerNorm(attn_dim),
            nn.Dropout(dropout),
        )
        self.correction_head = nn.Linear(attn_dim, vocab_size)

        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, ctc_logits, visual_feats, padding_mask=None):
        soft_posterior = torch.softmax(ctc_logits / self.temperature, dim=-1)
        query = self.query_proj(soft_posterior)

        retrieved_visual, _ = self.cross_attn(
            query=query,
            key=visual_feats,
            value=visual_feats,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        hidden = self.fusion(torch.cat([query, retrieved_visual], dim=-1))
        delta_logits = self.correction_head(hidden)

        refined_logits = ctc_logits + delta_logits

        return {
            "logits": refined_logits,
            "delta_logits": delta_logits,
            "query": query,
            "retrieved_visual": retrieved_visual,
        }
