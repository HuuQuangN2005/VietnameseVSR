# Project: VietnameseVSR
# License: CC BY-NC 4.0
# https://creativecommons.org/licenses/by-nc/4.0/

"""Word error rate scoring for word-level CTC vocabularies."""

import editdistance
import torch


class WERScorer:
    """Accumulate corpus WER from uncollapsed word-level CTC logits."""

    def __init__(self, blank_id: int = 0, ignore_id: int = -1) -> None:
        self.blank_id = blank_id
        self.ignore_id = ignore_id
        self.reset()

    def reset(self) -> None:
        """Clear accumulated edit and reference-word counts."""
        self.total_edits = 0
        self.total_reference_words = 0

    def collapse(
        self, logits: torch.Tensor, input_lengths: torch.Tensor
    ) -> list[list[int]]:
        """Greedily collapse batch-first CTC logits into token sequences."""
        if logits.ndim != 3:
            raise ValueError("logits must have shape (batch, time, vocabulary).")
        if input_lengths.shape != (logits.size(0),):
            raise ValueError("input_lengths must have shape (batch,).")

        frame_ids = logits.argmax(dim=-1)
        hypotheses = []
        for sequence, length in zip(frame_ids, input_lengths.tolist()):
            collapsed = torch.unique_consecutive(sequence[:length])
            collapsed = collapsed[collapsed != self.blank_id]
            hypotheses.append(collapsed.tolist())
        return hypotheses

    def update(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor | None = None,
    ) -> None:
        """Accumulate edit counts from one batch."""
        if labels.ndim != 2 or labels.size(0) != logits.size(0):
            raise ValueError("labels must have shape (batch, max_label_length).")
        if label_lengths is not None and label_lengths.shape != (labels.size(0),):
            raise ValueError("label_lengths must have shape (batch,).")

        hypotheses = self.collapse(logits, input_lengths)
        for index, hypothesis in enumerate(hypotheses):
            if label_lengths is None:
                reference = labels[index][labels[index] != self.ignore_id].tolist()
            else:
                reference_length = int(label_lengths[index])
                reference = labels[index, :reference_length].tolist()

            self.total_edits += editdistance.eval(reference, hypothesis)
            self.total_reference_words += len(reference)

    def compute(self) -> float:
        """Return accumulated corpus WER."""
        if self.total_reference_words == 0:
            raise RuntimeError("WER is undefined without reference words.")
        return self.total_edits / self.total_reference_words

    def __call__(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor | None = None,
    ) -> float:
        """Reset, score one batch, and return its corpus WER."""
        self.reset()
        self.update(logits, input_lengths, labels, label_lengths)
        return self.compute()
