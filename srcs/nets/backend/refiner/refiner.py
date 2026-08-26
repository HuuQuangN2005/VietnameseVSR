import torch
import torch.nn as nn

from srcs.nets.backend.refiner.utils import make_soft_blank_posterior
from srcs.nets.backend.refiner.conv import RefinerConv

from srcs.nets.backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from srcs.nets.backend.transformer.layer_norm import LayerNorm
from srcs.nets.backend.transformer.repeat import repeat
from srcs.nets.backend.nets_utils import make_non_pad_mask


class RefinerLayer(nn.Module):
    def __init__(
        self,
        attn_dim: int = 512,
        attn_head: int = 8,
        ffn_dim: int = 2048,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm_self = LayerNorm(attn_dim)
        self.norm_cross = LayerNorm(attn_dim)
        self.norm_ffn = LayerNorm(attn_dim)

        self.self_mha = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=attn_head,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.cross_mha = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=attn_head,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ffn = PositionwiseFeedForward(
            idim=attn_dim, hidden_units=ffn_dim, dropout_rate=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_emb: torch.Tensor,
        k_emb: torch.Tensor,
        v_emb: torch.Tensor,
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ):
        padding_mask = ~mask
        query_mask = mask.unsqueeze(-1)

        residual = q_emb
        self_output, _ = self.self_mha(
            q_emb, q_emb, q_emb, key_padding_mask=padding_mask, need_weights=False
        )
        q_emb = self.norm_self(residual + self.dropout(self_output)) * query_mask

        residual = q_emb
        cross_output, _ = self.cross_mha(
            q_emb,
            k_emb,
            v_emb,
            attn_mask=local_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        q_emb = self.norm_cross(residual + self.dropout(cross_output)) * query_mask

        residual = q_emb
        q_emb = self.norm_ffn(residual + self.ffn(q_emb)) * query_mask

        return q_emb, k_emb, v_emb, mask, local_mask


class VisualRefiner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        visual_dim: int = 512,
        attn_dim: int = 512,
        attn_head: int = 8,
        ffn_dim: int = 2048,
        num_blocks: int = 1,
        window_size: int = 3,
        blank_id: int = 0,
        reduce_ratio: float = 0.5,
        conv_kernel: int = 3,
        bias: bool = False,
        activation: str = "gelu",
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert conv_kernel > 0 and conv_kernel % 2 == 1
        assert num_blocks > 0
        assert window_size > 0 and window_size % 2 == 1

        self.blank_id = blank_id
        self.reduce_ratio = reduce_ratio
        self.window_size = window_size // 2

        self.logits_proj = nn.Linear(vocab_size, attn_dim, bias=bias)
        self.posterior_proj = nn.Linear(vocab_size, attn_dim, bias=bias)
        self.visual_proj = nn.Linear(visual_dim, attn_dim, bias=bias)

        self.dwconv = nn.Sequential(
            nn.Conv1d(
                attn_dim,
                attn_dim,
                kernel_size=conv_kernel,
                padding=conv_kernel // 2,
                groups=attn_dim,
                bias=bias,
            ),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.visual_norm = LayerNorm(attn_dim)

        self.layers = repeat(
            num_blocks,
            lambda _: RefinerLayer(
                attn_dim=attn_dim,
                attn_head=attn_head,
                ffn_dim=ffn_dim,
                attn_dropout=attn_dropout,
                dropout=dropout,
            ),
        )

        self.final_head = nn.Sequential(
            nn.Linear(2 * attn_dim, attn_dim),
            LayerNorm(attn_dim),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def make_local_mask(self, time: int, device):
        positions = torch.arange(time, device=device)

        return (positions.unsqueeze(1) - positions.unsqueeze(0)).abs() > self.window_size

    def forward(self, logits: torch.Tensor, visual_feats: torch.Tensor, video_lengths):
        logits = logits.detach()
        visual_feats = visual_feats.detach()

        if logits.shape[:2] != visual_feats.shape[:2]:
            raise ValueError(
                "Logits and visual features must have the same batch and time dimensions."
            )

        with torch.no_grad():
            p_old = logits.softmax(dim=-1)
            p_soft = make_soft_blank_posterior(p_old, self.blank_id, self.reduce_ratio)

        valid_mask = make_non_pad_mask(video_lengths).to(
            device=logits.device, dtype=torch.bool
        )
        local_mask = self.make_local_mask(logits.size(1), logits.device)

        logits_emb = self.logits_proj(logits)  # [B,T,D]
        posterior_emb = self.posterior_proj(p_soft)  # [B,T,D]

        visual_emb = self.visual_proj(visual_feats)  # [B,T,D]

        visual_ctx = self.dwconv(visual_emb.transpose(1, 2)).transpose(
            1, 2
        )  # [B,T,D] -> [B,D,T] -> [B, D, T]
        visual_ctx = self.visual_norm(visual_emb + visual_ctx)

        hidden, _, _, _, _ = self.layers(
            posterior_emb, visual_ctx, visual_emb, valid_mask, local_mask
        )

        logits = self.final_head(torch.cat([logits_emb, hidden], dim=-1))

        return {"logits": logits, "inner_logits": hidden}


class Refiner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        visual_dim: int = 512,
        attn_dim: int = 512,
        attn_head: int = 8,
        ffn_dim: int = 2048,
        num_blocks: int = 1,
        window_size: int = 17,
        blank_id: int = 0,
        reduce_ratio: float = 0.5,
        conv_kernel: int = 31,
        conv_layers: int = 2,
        bias: bool = False,
        activation: str = "gelu",
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert conv_kernel > 0 and conv_kernel % 2 == 1
        assert num_blocks > 0
        assert conv_layers > 0
        assert window_size > 0 and window_size % 2 == 1

        self.blank_id = blank_id
        self.reduce_ratio = reduce_ratio
        self.window_size = window_size // 2

        self.logits_proj = nn.Linear(vocab_size, attn_dim, bias=bias)
        self.posterior_proj = nn.Linear(vocab_size, attn_dim, bias=bias)
        self.visual_proj = nn.Linear(visual_dim, attn_dim, bias=bias)

        self.visual_conv = nn.ModuleList(
            [
                RefinerConv(
                    attn_dim, kernel_size=conv_kernel, dropout=dropout, act=activation
                )
                for _ in range(conv_layers)
            ]
        )
        self.visual_norm = LayerNorm(attn_dim)

        self.layers = repeat(
            num_blocks,
            lambda _: RefinerLayer(
                attn_dim=attn_dim,
                attn_head=attn_head,
                ffn_dim=ffn_dim,
                attn_dropout=attn_dropout,
                dropout=dropout,
            ),
        )

        self.final_head = nn.Sequential(
            nn.Linear(2 * attn_dim, attn_dim),
            LayerNorm(attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_proj = nn.Linear(attn_dim, vocab_size, bias=False)
        nn.init.zeros_(self.delta_proj.weight)

        self.gate_proj = nn.Linear(attn_dim, 1)

        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -1.0)

    def make_local_mask(self, time: int, device):
        positions = torch.arange(time, device=device)
        return (positions.unsqueeze(1) - positions.unsqueeze(0)).abs() > self.window_size

    def forward(self, logits: torch.Tensor, visual_feats: torch.Tensor, video_lengths):
        logits = logits.detach()
        visual_feats = visual_feats.detach()

        if logits.shape[:2] != visual_feats.shape[:2]:
            raise ValueError(
                "Logits and visual features must have the same batch and time dimensions."
            )

        with torch.no_grad():
            p_old = logits.softmax(dim=-1)
            p_soft = make_soft_blank_posterior(p_old, self.blank_id, self.reduce_ratio)

        valid_mask = make_non_pad_mask(video_lengths).to(
            device=logits.device, dtype=torch.bool
        )  # [B,T]

        mask = valid_mask.unsqueeze(-1).to(logits.dtype)  # [B,T,1]
        local_mask = self.make_local_mask(logits.size(1), logits.device)

        logits_emb = self.logits_proj(logits)  # [B,T,D]
        posterior_emb = self.posterior_proj(p_soft)  # [B,T,D]
        visual_emb = self.visual_proj(visual_feats)  # [B,T,D]

        visual_ctx = visual_emb
        for conv in self.visual_conv:
            visual_ctx = conv(visual_ctx, mask)

        visual_ctx = self.visual_norm(visual_ctx) * mask

        hidden, _, _, _, _ = self.layers(
            posterior_emb, visual_ctx, visual_ctx, valid_mask, local_mask
        )

        fused = self.final_head(torch.cat([logits_emb, hidden], dim=-1))
        delta = self.delta_proj(fused)
        gate = torch.sigmoid(self.gate_proj(fused))

        return {"logits": logits + gate * delta, "inner_logits": hidden}
