import torch
import torch.nn as nn

from srcs.nets.backend.transformer.attention import (
    LocalMultiHeadedAttention,
    MultiHeadedAttention,
)
from srcs.nets.backend.refiner.utils import make_soft_blank_posterior
from srcs.nets.backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from srcs.nets.backend.transformer.layer_norm import LayerNorm
from srcs.nets.backend.nets_utils import make_non_pad_mask


class MyRefiner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        attn_dim: int = 512,
        attn_head: int = 8,
        visual_dim: int = None,
        window_size: int = 3,
        blank_id: int = 0,
        k: int = 0,
        blank_reduce: float = 0.5,
        residual_scale: float = 0.1,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert k >= 0

        if not 0.0 <= blank_reduce <= 1.0:
            raise ValueError(f"blank_reduce must be between 0 and 1, got {blank_reduce}.")

        self.vocab_size = vocab_size
        self.attn_dim = attn_dim
        self.visual_dim = visual_dim
        self.blank_id = blank_id
        self.blank_reduce = blank_reduce
        self.residual_scale = residual_scale
        self.k = k

        self.p_proj = nn.Linear(vocab_size, attn_dim)

        self.v_proj = (
            nn.Linear(visual_dim, attn_dim) if visual_dim is not None else nn.Identity()
        )

        self.cross_attn = LocalMultiHeadedAttention(
            n_head=attn_head,
            n_feat=attn_dim,
            window_size=window_size,
            dropout_rate=attn_dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)

        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 4, attn_dim),
        )

        self.output_head = nn.Linear(attn_dim, vocab_size)

    def normalize_blank(self, p_old: torch.Tensor) -> torch.Tensor:
        """
        Adaptive blank softening.

        Args:
            p_old: [B, T, V]

        Returns:
            p_soft: [B, T, V]
        """
        p_old = p_old.float()
        eps = torch.finfo(p_old.dtype).eps

        p_blank = p_old[..., self.blank_id : self.blank_id + 1]

        p_nonblank = p_old.clone()
        p_nonblank[..., self.blank_id] = 0.0

        nonblank_mass = 1.0 - p_blank
        p_nb = p_nonblank / nonblank_mass.clamp_min(eps)

        g = self.blank_reduce * p_blank

        # P_soft = (1-g) P_old + g P_nb
        p_soft = (1.0 - g) * p_old + g * p_nb

        return p_soft

    def refine_once(
        self,
        p_input: torch.Tensor,
        visual_memory: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_attn: bool = False,
    ):
        """
        One shared-weight refinement step.

        Args:
            p_input: [B, T, V]
            visual_memory: [B, T, D]
        """

        h_post = self.p_proj(p_input)

        q = h_post

        if visual_memory is None:
            k = h_post
            v = h_post
        else:
            k = visual_memory
            v = visual_memory

        if return_attn:
            h_attn, attn = self.cross_attn(
                query=q, key=k, value=v, mask=mask, rtn_attn=True
            )

        else:
            h_attn = self.cross_attn(query=q, key=k, value=v, mask=mask, rtn_attn=False)
            attn = None

        h = self.norm1(h_post + self.dropout(h_attn))
        h = self.norm2(h + self.dropout(self.ffn(h)))

        delta_logits = self.output_head(h)

        return delta_logits, attn

    def forward(
        self,
        logits: torch.Tensor,
        visual_feats: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_attn: bool = False,
    ):

        base_logits = logits.detach()

        with torch.no_grad():
            p_old = base_logits.softmax(dim=-1)
            p_soft = self.normalize_blank(p_old)

        if visual_feats is not None:
            visual_memory = self.v_proj(visual_feats)
        else:
            visual_memory = None

        p_current = p_soft
        current_logits = base_logits

        logits_steps = []
        posterior_steps = []
        delta_logits_steps = []
        attention_steps = []

        for _ in range(self.k + 1):
            delta_logits, attn = self.refine_once(
                p_input=p_current,
                visual_memory=visual_memory,
                mask=mask,
                return_attn=return_attn,
            )

            new_logits = current_logits + self.residual_scale * delta_logits
            p_new = new_logits.softmax(dim=-1)

            logits_steps.append(new_logits)
            posterior_steps.append(p_new)
            delta_logits_steps.append(delta_logits)

            if return_attn:
                attention_steps.append(attn)

            p_current = p_new
            current_logits = new_logits

        return {
            "logits_steps": logits_steps,
            "posterior_steps": posterior_steps,
            "delta_logits_steps": delta_logits_steps,
            "attention_steps": (attention_steps if return_attn else None),
        }


class RefinerLayer(nn.Module):
    def __init__(
        self,
        attn_dim: int = 512,
        attn_head: int = 8,
        ffn_dim: int = 2048,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.norm_self = LayerNorm(attn_dim)
        self.norm_cross = LayerNorm(attn_dim)
        self.norm_ffn = LayerNorm(attn_dim)

        self.self_mha = MultiHeadedAttention(
            n_head=attn_head, n_feat=attn_dim, dropout_rate=attn_dropout
        )
        self.cross_mha = MultiHeadedAttention(
            n_head=attn_head, n_feat=attn_dim, dropout_rate=attn_dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.tensor, k: torch.tensor, v: torch.tensor, mask: torch.Tensor
    ):
        ctx_q = self.self_mha(q, q, q, mask.unsqueeze(-2))


class Refiner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        visual_dim: int = 512,
        attn_dim: int = 512,
        blank_id: int = 0,
        reduce_ratio: float = 0.5,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.reduce_ratio = reduce_ratio

        self.visual_proj = nn.Linear(visual_dim, attn_dim)
        self.posterior_proj = nn.Linear(vocab_size, attn_dim)
        self.fusion = nn.Linear(2 * attn_dim, attn_dim)

        self.norm_k = LayerNorm(attn_dim)
        self.norm_v = LayerNorm(attn_dim)

    def forward(
        self,
        logits: torch.Tensor,
        visual_feats: torch.Tensor,
        video_lengths,
        return_attn: bool = False,
    ):
        logits = logits.detach()

        with torch.no_grad():
            p_old = logits.softmax(dim=-1)
            p_soft = make_soft_blank_posterior(p_old, self.blank_id, self.reduce_ratio)

        valid_mask = make_non_pad_mask(video_lengths).to(
            device=logits.device, dtype=torch.bool
        )  # [B,T]

        posterior_emb = self.posterior_proj(p_soft)
        visual_emb = self.visual_proj(visual_feats)

        k = self.norm_k(self.fusion(torch.cat([posterior_emb, visual_emb], dim=-1)))
        v = self.norm_v(visual_emb)
