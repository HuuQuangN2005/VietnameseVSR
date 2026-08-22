import torch


def make_soft_blank_posterior(
    posterior: torch.Tensor, blank_id: int = 0, ratio=0.3
) -> torch.Tensor:
    """
    Adaptive blank softening.

    Args:
        posterior: [B, T, V]

    Returns:
        new_posterior: [B, T, V]
    """

    eps = torch.finfo(posterior.dtype).eps

    p_blank = posterior[..., blank_id : blank_id + 1]

    p_nonblank = posterior.clone()
    p_nonblank[..., blank_id] = 0.0
    nonblank_mass = 1.0 - p_blank

    p_nb = p_nonblank / nonblank_mass.clamp_min(eps)
    g = ratio * p_blank

    # P_soft = (1-g) P_old + g P_nb
    return (1.0 - g) * posterior + g * p_nb
