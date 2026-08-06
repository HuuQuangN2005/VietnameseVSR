# Source (modified): https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/src/custom_trainer.py
# License: CC BY-NC 4.0 (https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/LICENSE)

from contextlib import contextmanager

import numpy as np
import torch
from transformers import Trainer

from srcs.nets.scorers.cer import CER
from srcs.nets.scorers.ctc import ctc_decode
from srcs.nets.scorers.wer import WER


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


def build_metric_fn(text_transform):
    def compute(eval_pred):
        predictions = eval_pred.predictions
        if isinstance(predictions, (tuple, list)):
            logits = predictions[0]
            input_lengths = predictions[1] if len(predictions) > 1 else None
        else:
            logits = predictions
            input_lengths = None
        logits = np.asarray(logits)
        if input_lengths is not None:
            input_lengths = np.asarray(input_lengths)
        token_ids = ctc_decode(
            torch.from_numpy(logits),
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
            references = [
                text_transform.decode(item[item != -1]) for item in labels
            ]
        else:
            references = [
                text_transform.decode(item[: int(size)])
                for item, size in zip(labels, label_lengths)
            ]
        return {
            "wer": WER()(hypotheses, references),
            "cer": CER()(hypotheses, references),
        }

    return compute
