import torch
import torch.nn as nn

from srcs.nets.backend.ctc import CTC
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.backend.refiner.conv import RefinerConv
from srcs.nets.backend.transformer.embedding import ScaledPositionalEncoding
from srcs.nets.backend.transformer.layer_norm import LayerNorm
from srcs.nets.backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from srcs.nets.backend.transformer.repeat import repeat


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
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_mask: torch.Tensor,
        local_mask: torch.Tensor,
    ):
        padding_mask = ~valid_mask
        query_mask = valid_mask.unsqueeze(-1).to(query.dtype)

        residual = query
        hidden = self.norm_self(query)
        hidden, _ = self.self_mha(
            hidden, hidden, hidden, key_padding_mask=padding_mask, need_weights=False
        )
        query = (residual + self.dropout(hidden)) * query_mask

        residual = query
        hidden, _ = self.cross_mha(
            self.norm_cross(query),
            key,
            value,
            attn_mask=local_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        query = (residual + self.dropout(hidden)) * query_mask

        residual = query
        hidden = self.ffn(self.norm_ffn(query))
        query = (residual + self.dropout(hidden)) * query_mask

        return query, key, value, valid_mask, local_mask


class Refiner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        visual_dim: int = 512,
        encoder_dim: int = 768,
        attn_dim: int = 512,
        attn_head: int = 8,
        ffn_dim: int = 2048,
        num_blocks: int = 2,
        window_size: int = 17,
        blank_id: int = 0,
        ignore_id: int = -1,
        conv_kernel: int = 7,
        conv_layers: int = 1,
        bias: bool = False,
        activation: str = "silu",
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero.")

        if conv_layers <= 0:
            raise ValueError("conv_layers must be greater than zero.")

        if conv_kernel <= 0 or conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be a positive odd number.")

        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd number.")

        self.window_radius = window_size // 2

        self.posterior_proj = nn.Linear(vocab_size, attn_dim, bias=bias)
        self.posterior_norm = LayerNorm(attn_dim)

        self.visual_projections = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(visual_dim), nn.Linear(visual_dim, attn_dim, bias=bias)
                ),
                nn.Sequential(
                    LayerNorm(encoder_dim), nn.Linear(encoder_dim, attn_dim, bias=bias)
                ),
                nn.Sequential(
                    LayerNorm(encoder_dim), nn.Linear(encoder_dim, attn_dim, bias=bias)
                ),
            ]
        )

        self.visual_fusion = nn.Sequential(
            nn.Linear(3 * attn_dim, attn_dim, bias=bias), LayerNorm(attn_dim)
        )
        self.visual_conv = nn.ModuleList(
            [
                RefinerConv(
                    attn_dim,
                    kernel_size=conv_kernel,
                    dropout=dropout,
                    act=activation,
                    bias=bias,
                )
                for _ in range(conv_layers)
            ]
        )
        self.visual_norm = LayerNorm(attn_dim)

        self.position = ScaledPositionalEncoding(attn_dim, dropout)

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

        self.final_norm = LayerNorm(attn_dim)
        self.ctc = CTC(
            output_size=vocab_size,
            input_size=attn_dim,
            dropout_rate=dropout,
            blank_id=blank_id,
            ignore_id=ignore_id,
            reduce=True,
        )

    def make_local_mask(self, time: int, device):
        positions = torch.arange(time, device=device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)

        return distance.abs() > self.window_radius

    def _fuse_visual_contexts(self, visual_contexts, mask):
        sources = [
            visual_contexts["visual_features"].detach(),
            visual_contexts["h2_features"].detach(),
            visual_contexts["h4_features"].detach(),
        ]
        batch_time = sources[0].shape[:2]

        if any(source.shape[:2] != batch_time for source in sources[1:]):
            raise ValueError(
                "All visual context features must share batch and time dimensions."
            )

        projected = [
            projection(source)
            for projection, source in zip(self.visual_projections, sources)
        ]
        memory = self.visual_fusion(torch.cat(projected, dim=-1)) * mask

        for conv in self.visual_conv:
            memory = conv(memory, mask)

        return self.visual_norm(memory) * mask

    def forward(self, logits, visual_contexts, labels=None, label_lengths=None):
        logits = logits.detach()
        input_lengths = visual_contexts["input_lengths"]

        with torch.no_grad():
            posterior = logits.softmax(dim=-1)

        valid_mask = make_non_pad_mask(input_lengths).to(
            device=logits.device, dtype=torch.bool
        )
        if valid_mask.size(1) != logits.size(1):
            raise ValueError("Input lengths do not match the context time dimension.")

        mask = valid_mask.unsqueeze(-1).to(logits.dtype)
        local_mask = self.make_local_mask(logits.size(1), logits.device)

        query = self.posterior_norm(self.posterior_proj(posterior))
        query = self.position(query) * mask

        memory = self._fuse_visual_contexts(visual_contexts, mask)
        memory = self.position(memory) * mask

        hidden, _, _, _, _ = self.layers(query, memory, memory, valid_mask, local_mask)
        hidden = self.final_norm(hidden) * mask

        loss, refined_logits = self.ctc(hidden, input_lengths, labels, label_lengths)

        return {"loss": loss, "logits": refined_logits, "input_lengths": input_lengths}
