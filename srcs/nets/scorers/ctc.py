import torch


def ctc_decode(logits, input_lengths=None, blank_id=0):
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, time, vocab)")
    if input_lengths is None:
        input_lengths = torch.full(
            (logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device
        )
    frame_ids = logits.argmax(-1)
    decoded_tokens = []
    for token_ids, length in zip(frame_ids, input_lengths.tolist()):
        token_ids = torch.unique_consecutive(token_ids[: int(length)])
        decoded_tokens.append(
            token_ids[token_ids.ne(blank_id)].detach().cpu().tolist()
        )
    return decoded_tokens


class CTCScorer:
    def __init__(self, blank_id=0):
        self.blank_id = blank_id

    def __call__(self, logits, input_lengths=None):
        return ctc_decode(logits, input_lengths, self.blank_id)
