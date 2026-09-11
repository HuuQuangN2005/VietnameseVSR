import torch.nn as nn
import torch.nn.functional as F

from srcs.nets.backend.nets_utils import COMP_NAMES


class PhonemeCELoss(nn.Module):
    ignore_id = -1

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        loss = 0.0

        for index, name in enumerate(COMP_NAMES):
            loss = loss + F.cross_entropy(
                logits[name].flatten(0, 1).float(),
                targets[..., index].flatten(),
                ignore_index=self.ignore_id,
                label_smoothing=self.label_smoothing,
                reduction="sum",
            )

        return loss / targets.size(0)
