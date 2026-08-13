# Source (modified): https://github.com/mpc001/auto_avsr/blob/main/espnet/nets/pytorch_backend/ctc.py
# License: Apache-2.0 (https://github.com/mpc001/auto_avsr/blob/main/LICENSE)

import torch

from srcs.nets.utils import ctc_decode


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
        self.blank_id = blank_id
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

    def loss_from_logits(
        self, logits, input_lengths, labels, label_lengths=None, reduce=None
    ):
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
        use_reduce = self.reduce if reduce is None else reduce

        if use_reduce == self.reduce:
            loss = self.ctc_loss(log_probs, targets, input_lengths, label_lengths)
        else:
            loss = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                label_lengths,
                blank=self.blank_id,
                reduction="sum" if use_reduce else "none",
                zero_infinity=True,
            )

        if use_reduce:
            loss = loss / logits.size(0)

        # loss: scalar when reduce is enabled, otherwise [B]
        return loss


class CTCFrameCorrectionLoss(torch.nn.Module):
    def __init__(self, frame_weight=1.0, eps=1e-6):
        super().__init__()

        self.frame_weight = frame_weight
        self.eps = eps

    def _prepare_targets(self, ctc, labels, label_lengths, device):
        labels = labels.to(device=device, dtype=torch.long)

        if label_lengths is None:
            label_mask = labels.ne(ctc.ignore_id)
            label_lengths = label_mask.sum(dim=1)

        else:
            label_lengths = label_lengths.to(device=device, dtype=torch.long)
            positions = torch.arange(labels.size(1), device=device)
            label_mask = positions.unsqueeze(0) < label_lengths.unsqueeze(1)

        targets = labels.masked_select(label_mask)

        return labels, targets, label_lengths

    def _ctc_alignment_posterior(
        self, ctc, first_logits, input_lengths, targets, label_lengths
    ):
        with torch.enable_grad():
            teacher_logits = first_logits.detach().float().requires_grad_(True)
            log_probs = teacher_logits.log_softmax(dim=-1).transpose(0, 1)

            teacher_loss = ctc.ctc_loss(
                log_probs, targets, input_lengths, label_lengths
            )
            teacher_loss = teacher_loss.sum()

            gradient = torch.autograd.grad(
                teacher_loss, teacher_logits, create_graph=False, retain_graph=False
            )[0]

        with torch.no_grad():
            posterior = teacher_logits.softmax(dim=-1)

            gamma = (posterior - gradient).clamp_min(0.0)
            gamma = gamma / (gamma.sum(dim=-1, keepdim=True).clamp_min(self.eps))

        return posterior.detach(), gamma.detach()

    def forward(
        self, ctc, first_logits, logits, input_lengths, labels, label_lengths=None
    ):
        input_lengths = input_lengths.to(logits.device, dtype=torch.long)

        labels, targets, label_lengths = self._prepare_targets(
            ctc, labels, label_lengths, logits.device
        )

        ctc_loss = ctc.loss_from_logits(logits, input_lengths, labels, label_lengths)

        if ctc_loss.ndim == 0:
            ctc_loss = ctc_loss / label_lengths.float().mean().clamp_min(1.0)

        else:
            ctc_loss = (ctc_loss / label_lengths.float().clamp_min(1.0)).mean()

        posterior, gamma = self._ctc_alignment_posterior(
            ctc, first_logits, input_lengths, targets, label_lengths
        )

        valid_mask = torch.arange(logits.size(1), device=logits.device).unsqueeze(
            0
        ) < input_lengths.unsqueeze(1)

        with torch.no_grad():
            frame_error = 0.5 * (posterior - gamma).abs().sum(dim=-1)
            frame_weight = frame_error * valid_mask.float()

        refined_log_probs = logits.float().log_softmax(dim=-1)

        frame_ce = -(gamma * refined_log_probs).sum(dim=-1)
        frame_loss = (frame_weight * frame_ce).sum() / (
            frame_weight.sum().clamp_min(self.eps)
        )

        loss = ctc_loss + self.frame_weight * frame_loss

        return {
            "loss": loss,
            "ctc_loss": ctc_loss.detach(),
            "frame_loss": frame_loss.detach(),
        }
