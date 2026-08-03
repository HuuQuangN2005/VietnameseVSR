# Project: VietnameseVSR
# License: CC BY-NC 4.0
# https://creativecommons.org/licenses/by-nc/4.0/

"""CTC loss scoring for batch-first model logits."""

import torch
from torch import nn


class CTCScorer(nn.Module):
    """Compute CTC loss with the same reduction used by Auto-AVSR."""

    def __init__(
        self, blank_id: int = 0, ignore_id: int = -1, zero_infinity: bool = True
    ) -> None:
        super().__init__()
        self.ignore_id = ignore_id
        self.criterion = nn.CTCLoss(
            blank=blank_id, reduction="sum", zero_infinity=zero_infinity
        )

    def forward(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the batch-averaged CTC loss for logits shaped (B, T, V)."""
        if logits.ndim != 3:
            raise ValueError("logits must have shape (batch, time, vocabulary).")
        if labels.ndim != 2:
            raise ValueError("labels must have shape (batch, max_label_length).")

        batch_size = logits.size(0)
        if input_lengths.shape != (batch_size,):
            raise ValueError("input_lengths must have shape (batch,).")
        if labels.size(0) != batch_size:
            raise ValueError("labels and logits must have the same batch size.")

        labels = labels.to(logits.device)
        if label_lengths is None:
            targets = [label[label != self.ignore_id] for label in labels]
            label_lengths = torch.tensor(
                [target.numel() for target in targets],
                device=logits.device,
                dtype=torch.long,
            )
        else:
            if label_lengths.shape != (batch_size,):
                raise ValueError("label_lengths must have shape (batch,).")
            label_lengths = label_lengths.to(logits.device, dtype=torch.long)
            targets = [
                label[:length] for label, length in zip(labels, label_lengths.tolist())
            ]

        flattened_targets = torch.cat(targets).to(dtype=torch.long)
        input_lengths = input_lengths.to(logits.device, dtype=torch.long)
        log_probs = logits.transpose(0, 1).log_softmax(dim=-1)

        with torch.backends.cudnn.flags(deterministic=True):
            loss = self.criterion(
                log_probs, flattened_targets, input_lengths, label_lengths
            )

        return loss / batch_size
