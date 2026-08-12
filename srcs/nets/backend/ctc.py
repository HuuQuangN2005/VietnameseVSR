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


class WERGuidedCTCLoss(torch.nn.Module):
    def __init__(self, tokenizer, wer_weight=1.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.wer_weight = wer_weight

    @torch.no_grad()
    def _decode_logits(self, logits, input_lengths, blank_id):
        token_ids = ctc_decode(logits, input_lengths, blank_id)
        return [self.tokenizer.decode(ids) for ids in token_ids]

    @torch.no_grad()
    def _decode_labels(self, labels, label_lengths):
        labels = labels.detach().cpu()
        label_lengths = label_lengths.detach().cpu().tolist()
        texts = []

        for row, length in zip(labels, label_lengths):
            ids = row[: int(length)].tolist()

            texts.append(self.tokenizer.decode(ids))

        return texts

    @staticmethod
    def _edit_distance(reference, hypothesis):
        reference = reference.split()
        hypothesis = hypothesis.split()

        previous = list(range(len(hypothesis) + 1))

        for i, ref_word in enumerate(reference, start=1):
            current = [i]

            for j, hyp_word in enumerate(hypothesis, start=1):
                substitution = previous[j - 1] + int(ref_word != hyp_word)
                insertion = current[j - 1] + 1
                deletion = previous[j] + 1

                current.append(min(substitution, insertion, deletion))

            previous = current

        return previous[-1]

    @classmethod
    def _wer(cls, reference, hypothesis):
        words = reference.split()

        if len(words) == 0:
            return float(len(hypothesis.split()) > 0)

        errors = cls._edit_distance(reference, hypothesis)

        return errors / len(words)

    def forward(self, ctc, logits, input_lengths, labels, label_lengths=None):
        if label_lengths is None:
            label_lengths = labels.ne(ctc.ignore_id).sum(dim=1)

        ctc_loss = ctc.loss_from_logits(
            logits, input_lengths, labels, label_lengths, reduce=False
        )

        ctc_loss = ctc_loss / label_lengths.to(ctc_loss.device).float().clamp_min(1)

        with torch.no_grad():
            predictions = self._decode_logits(logits, input_lengths, ctc.blank_id)
            targets = self._decode_labels(labels, label_lengths)

            wer = torch.tensor(
                [
                    self._wer(target, prediction)
                    for target, prediction in zip(targets, predictions)
                ],
                device=logits.device,
                dtype=ctc_loss.dtype,
            )

            severity = wer / (1.0 + wer)
            weight = 1.0 + self.wer_weight * severity

        loss = (weight * ctc_loss).mean()

        return {"loss": loss, "raw_ctc_loss": ctc_loss.mean().detach()}
