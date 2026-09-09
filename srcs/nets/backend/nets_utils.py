import torch


def make_non_pad_mask(lengths, max_length=None):
    lengths = torch.as_tensor(lengths)
    if lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional.")
    if lengths.numel() == 0:
        raise ValueError("lengths must not be empty.")

    if max_length is None:
        max_length = int(lengths.max().item())

    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)
