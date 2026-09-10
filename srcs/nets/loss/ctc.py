import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def ctc_decode(logits, input_lengths, blank=0):
    predicted_ids = logits.float().argmax(dim=-1).cpu()
    input_lengths = input_lengths.detach().cpu().tolist()

    sequences = []

    for sample_ids, length in zip(predicted_ids, input_lengths):
        sequence = []
        previous = None

        for current_id in sample_ids[:length]:
            current_id = int(current_id)

            if current_id == blank:
                previous = None
                continue

            if current_id != previous:
                sequence.append(current_id)

            previous = current_id

        sequences.append(sequence)

    return sequences


class CTCLoss(nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="sum", zero_infinity=True)

    def forward(self, logits, labels, input_lengths, label_lengths):
        positions = torch.arange(labels.size(1), device=labels.device)
        label_mask = positions.unsqueeze(0) < label_lengths.unsqueeze(1)

        log_probs = F.log_softmax(logits.float(), dim=-1)

        loss = self.ctc(
            log_probs.transpose(0, 1),
            labels[label_mask],
            input_lengths,
            label_lengths,
        )

        return loss / labels.size(0)
