import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def mctc_decode(logits, input_lengths):
    blank_logits = logits["blank"].float()
    component_logits = [logits[name].float() for name in ("initial", "rhyme", "tone")]

    nonblank_scores = F.logsigmoid(-blank_logits)

    for component in component_logits:
        component_scores = F.log_softmax(component, dim=-1)
        nonblank_scores = nonblank_scores + component_scores.max(dim=-1).values

    blank_mask = (F.logsigmoid(blank_logits) >= nonblank_scores).cpu()

    component_ids = torch.stack(
        [component.argmax(dim=-1) for component in component_logits],
        dim=-1,
    ).cpu()

    input_lengths = input_lengths.detach().cpu().tolist()

    sequences = []

    for sample_ids, sample_blank, length in zip(
        component_ids,
        blank_mask,
        input_lengths,
    ):
        sequence = []
        previous = None

        for current_ids, is_blank in zip(
            sample_ids[:length],
            sample_blank[:length],
        ):
            if is_blank:
                previous = None
                continue

            current_ids = current_ids.tolist()

            if current_ids != previous:
                sequence.append(current_ids)

            previous = current_ids

        sequences.append(sequence)

    return sequences


class MCTCWELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)

    def forward(self, logits, labels, input_lengths, label_lengths):

        positions = torch.arange(labels.size(1), device=labels.device)
        label_mask = positions.unsqueeze(0) < label_lengths.unsqueeze(1)
        target_classes, target_ids = torch.unique(
            labels[label_mask],
            dim=0,
            return_inverse=True,
        )

        initial_ids, rhyme_ids, tone_ids = target_classes.unbind(-1)

        blank_logits = logits["blank"].float()
        initial_log_probs = F.log_softmax(logits["initial"].float(), dim=-1)
        rhyme_log_probs = F.log_softmax(logits["rhyme"].float(), dim=-1)
        tone_log_probs = F.log_softmax(logits["tone"].float(), dim=-1)

        initial_log_probs = initial_log_probs[:, :, initial_ids]
        rhyme_log_probs = rhyme_log_probs[:, :, rhyme_ids]
        tone_log_probs = tone_log_probs[:, :, tone_ids]

        joint_log_probs = (
            F.logsigmoid(-blank_logits).unsqueeze(-1)
            + initial_log_probs
            + rhyme_log_probs
            + tone_log_probs
        )

        scores = torch.cat(
            [F.logsigmoid(blank_logits).unsqueeze(-1), joint_log_probs],
            dim=-1,
        )
        normalizer = scores.logsumexp(dim=-1)
        log_probs = scores - normalizer.unsqueeze(-1)

        loss = self.ctc(
            log_probs.transpose(0, 1),
            target_ids + 1,
            input_lengths,
            label_lengths,
        )

        frames = torch.arange(scores.size(1), device=scores.device)
        valid_frames = frames.unsqueeze(0) < input_lengths.to(scores.device).unsqueeze(1)
        loss = loss - (normalizer * valid_frames).sum()

        return loss / labels.size(0)
