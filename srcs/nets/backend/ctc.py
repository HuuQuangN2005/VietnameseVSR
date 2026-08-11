# Source (modified): https://github.com/mpc001/auto_avsr/blob/main/espnet/nets/pytorch_backend/ctc.py
# License: Apache-2.0 (https://github.com/mpc001/auto_avsr/blob/main/LICENSE)

import torch


class CTC(torch.nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        dropout_rate=0.1,
        reduce=True,
        blank_id=0,
        ignore_id=-1,
    ):
        super().__init__()
        self.ignore_id = ignore_id
        self.reduce = reduce
        self.ctc_lo = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.ctc_loss = torch.nn.CTCLoss(
            blank=blank_id, reduction="sum" if reduce else "none", zero_infinity=True
        )

    def forward(self, encoded_features, input_lengths, labels=None, label_lengths=None):
        # encoded_features: [B, T, D], input_lengths: [B]
        logits = self.ctc_lo(self.dropout(encoded_features))
        # logits: [B, T, V]
        if labels is None:
            return None, logits

        loss = self.loss_from_logits(logits, input_lengths, labels, label_lengths)
        return loss, logits

    def loss_from_logits(self, logits, input_lengths, labels, label_lengths=None):
        labels = labels.to(logits.device, dtype=torch.long)
        input_lengths = input_lengths.to(logits.device, dtype=torch.long)

        if label_lengths is None:
            label_mask = labels.ne(self.ignore_id)
            label_lengths = label_mask.sum(dim=1)
        else:
            label_lengths = label_lengths.to(logits.device, dtype=torch.long)
            positions = torch.arange(labels.size(1), device=logits.device)
            label_mask = positions.unsqueeze(0) < label_lengths.unsqueeze(1)

        # label_mask: [B, L], label_lengths: [B]
        targets = labels.masked_select(label_mask)
        # targets: [sum(label_lengths)]
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        # log_probs: [T, B, V]
        loss = self.ctc_loss(log_probs, targets, input_lengths, label_lengths)

        if self.reduce:
            loss = loss / logits.size(0)

        # loss: scalar when reduce is enabled, otherwise [B]
        return loss
