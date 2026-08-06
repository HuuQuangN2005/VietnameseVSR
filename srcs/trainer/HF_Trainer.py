# Source (modified): https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/src/custom_trainer.py
# License: CC BY-NC 4.0 (https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/LICENSE)

from contextlib import contextmanager

import numpy as np
import torch
from torchmetrics.text import WordErrorRate
from transformers import Trainer

from srcs.nets.utils import ctc_decode


class HFTrainer(Trainer):
    def __init__(self, *args, validation_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.remove_unused_columns:
            raise ValueError("remove_unused_columns must be False.")
        self.validation_collator = validation_collator or self.data_collator

    @contextmanager
    def _use_validation_collator(self):
        collator = self.data_collator
        self.data_collator = self.validation_collator
        try:
            yield
        finally:
            self.data_collator = collator

    def get_eval_dataloader(self, eval_dataset=None):
        with self._use_validation_collator():
            return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset):
        with self._use_validation_collator():
            return super().get_test_dataloader(test_dataset)


def preprocess_logits_for_metrics(logits, labels):
    del labels
    if isinstance(logits, (tuple, list)):
        scores = logits[0]
        input_lengths = logits[1] if len(logits) > 1 else None
    else:
        scores = logits
        input_lengths = None
    frame_ids = scores.argmax(dim=-1)
    return frame_ids if input_lengths is None else (frame_ids, input_lengths)


def build_metric_fn(text_transform):
    def compute(eval_pred):
        predictions = eval_pred.predictions
        if isinstance(predictions, (tuple, list)):
            frame_ids = predictions[0]
            input_lengths = predictions[1] if len(predictions) > 1 else None
        else:
            frame_ids = predictions
            input_lengths = None
        frame_ids = np.asarray(frame_ids)
        if input_lengths is not None:
            input_lengths = np.asarray(input_lengths)
        token_ids = ctc_decode(
            torch.from_numpy(frame_ids),
            None if input_lengths is None else torch.from_numpy(input_lengths),
            text_transform.blank_id,
        )
        labels = eval_pred.label_ids
        label_lengths = None
        if isinstance(labels, (tuple, list)):
            labels, label_lengths = labels[0], labels[1]
        labels = np.asarray(labels)
        hypotheses = [text_transform.decode(item) for item in token_ids]
        if label_lengths is None:
            references = [text_transform.decode(item[item != -1]) for item in labels]
        else:
            references = [
                text_transform.decode(item[: int(size)])
                for item, size in zip(labels, label_lengths)
            ]
        return {"wer": WordErrorRate()(hypotheses, references).item()}

    return compute
