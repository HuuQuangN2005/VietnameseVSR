# -*- coding: utf-8 -*-

# Source repository: https://github.com/mpc001/auto_avsr
# Source path: espnet/nets/pytorch_backend/nets_utils.py
# License: Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Network utilities required by the visual CTC model."""

import logging

import torch


def to_device(module_or_tensor, tensor):
    """Move a tensor to the device of a module or reference tensor."""
    if isinstance(module_or_tensor, torch.nn.Module):
        device = next(module_or_tensor.parameters()).device
    elif isinstance(module_or_tensor, torch.Tensor):
        device = module_or_tensor.device
    else:
        raise TypeError(
            "Expected torch.nn.Module or torch.Tensor, "
            f"but got: {type(module_or_tensor)}"
        )
    return tensor.to(device)


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Return a boolean mask whose padded positions are true."""
    if length_dim == 0:
        raise ValueError(f"length_dim cannot be 0: {length_dim}")

    if not isinstance(lengths, list):
        lengths = lengths.tolist()

    batch_size = len(lengths)
    if maxlen is None:
        maxlen = int(max(lengths)) if xs is None else xs.size(length_dim)
    else:
        if xs is not None:
            raise ValueError("xs must be None when maxlen is specified.")
        if maxlen < int(max(lengths)):
            raise ValueError("maxlen must be at least the maximum sequence length.")

    sequence_range = torch.arange(0, maxlen, dtype=torch.int64)
    sequence_range = sequence_range.unsqueeze(0).expand(batch_size, maxlen)
    sequence_lengths = sequence_range.new(lengths).unsqueeze(-1)
    mask = sequence_range >= sequence_lengths

    if xs is not None:
        if xs.size(0) != batch_size:
            raise ValueError("xs and lengths must have the same batch size.")
        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        indices = tuple(
            slice(None) if index in (0, length_dim) else None for index in range(xs.dim())
        )
        mask = mask[indices].expand_as(xs).to(xs.device)

    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Return a boolean mask whose non-padded positions are true."""
    return ~make_pad_mask(lengths, xs, length_dim)


def rename_state_dict(old_prefix, new_prefix, state_dict):
    """Rename matching state-dict key prefixes for backward compatibility."""
    old_keys = [key for key in state_dict if key.startswith(old_prefix)]
    if old_keys:
        logging.warning("Rename: %s -> %s", old_prefix, new_prefix)

    for old_key in old_keys:
        value = state_dict.pop(old_key)
        new_key = old_key.replace(old_prefix, new_prefix)
        state_dict[new_key] = value
