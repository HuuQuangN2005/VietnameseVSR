# Source (modified): https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/src/custom_trainer.py
# License: CC BY-NC 4.0 (https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/LICENSE)

from contextlib import contextmanager

import numpy as np
import torch
from torchmetrics.text import WordErrorRate

from transformers import Trainer, TrainingArguments

from srcs.nets.utils import ctc_decode


class HFTrainer(Trainer):
    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset,
        eval_dataset,
        train_collator,
        validation_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        callbacks=None,
        allowed_trainable_names=None,
    ):
        if validation_collator is None:
            raise ValueError("validation_collator must be provided.")

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=train_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=callbacks,
        )

        if self.args.remove_unused_columns:
            raise ValueError("remove_unused_columns must be False.")

        self.validation_collator = validation_collator
        self.allowed_trainable_names = allowed_trainable_names
        self._validate_trainable_parameters()

        self.diagnostic_totals = {
            "train": self._new_diagnostics(),
            "eval": self._new_diagnostics(),
        }

    @staticmethod
    def _new_diagnostics():
        return {"flip_count": 0.0, "frame_count": 0}

    def _validate_trainable_parameters(self):
        trainable_names = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]

        if not trainable_names:
            raise ValueError("The model has no trainable parameters.")

        if self.allowed_trainable_names is None:
            return

        unexpected = [
            name
            for name in trainable_names
            if not any(pattern in name for pattern in self.allowed_trainable_names)
        ]

        if unexpected:
            raise ValueError(
                "Unexpected trainable parameters: " + ", ".join(unexpected[:10])
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        if loss is None:
            raise RuntimeError("Model output does not contain a loss.")

        if loss.ndim != 0:
            raise RuntimeError(
                f"Expected scalar loss, received " f"shape {tuple(loss.shape)}."
            )

        if isinstance(outputs, dict):
            mode = "train" if model.training else "eval"
            self._update_diagnostics(outputs=outputs, mode=mode)

        if return_outputs:
            return loss, outputs

        return loss

    def _update_diagnostics(self, outputs, mode):
        refined_flip_rate = outputs.get("refined_flip_rate")
        input_lengths = outputs.get("input_lengths")

        if refined_flip_rate is None:
            return

        if input_lengths is None:
            raise RuntimeError(
                "Model output must contain input_lengths "
                "to aggregate refined_flip_rate."
            )

        frame_count = int(input_lengths.detach().sum().item())
        totals = self.diagnostic_totals[mode]

        totals["flip_count"] += refined_flip_rate.detach().float().item() * frame_count
        totals["frame_count"] += frame_count

    def log(self, logs, start_time=None):
        if "loss" in logs:
            self._add_diagnostics(logs=logs, mode="train")

        if "eval_loss" in logs:
            self._add_diagnostics(logs=logs, mode="eval")

        super().log(logs=logs, start_time=start_time)

    def _add_diagnostics(self, logs, mode):
        totals = self.diagnostic_totals[mode]
        frame_count = totals["frame_count"]

        if frame_count == 0:
            return

        prefix = "eval_" if mode == "eval" else ""
        logs[prefix + "refined_flip_rate"] = totals["flip_count"] / frame_count

        self.diagnostic_totals[mode] = self._new_diagnostics()

    @contextmanager
    def _use_validation_collator(self):
        current_collator = self.data_collator
        self.data_collator = self.validation_collator

        try:
            yield
        finally:
            self.data_collator = current_collator

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
        first_scores = logits[2] if len(logits) > 2 else None

    else:
        scores = logits
        input_lengths = None
        first_scores = None

    frame_ids = scores.argmax(dim=-1)

    if first_scores is not None and first_scores.ndim == 3:
        first_frame_ids = first_scores.argmax(dim=-1)
        return frame_ids, input_lengths, first_frame_ids

    return frame_ids if input_lengths is None else (frame_ids, input_lengths)


def build_metric_fn(text_transform):
    def compute(eval_pred):
        predictions = eval_pred.predictions

        if isinstance(predictions, (tuple, list)):
            frame_ids = predictions[0]
            input_lengths = predictions[1] if len(predictions) > 1 else None
            first_frame_ids = predictions[2] if len(predictions) > 2 else None
        else:
            frame_ids = predictions
            input_lengths = None
            first_frame_ids = None

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

        metric_name = "refined_wer" if first_frame_ids is not None else "wer"
        metrics = {metric_name: WordErrorRate()(hypotheses, references).item()}

        if first_frame_ids is not None:
            first_token_ids = ctc_decode(
                torch.from_numpy(np.asarray(first_frame_ids)),
                None if input_lengths is None else torch.from_numpy(input_lengths),
                text_transform.blank_id,
            )
            first_hypotheses = [text_transform.decode(item) for item in first_token_ids]
            metrics["first_pass_wer"] = WordErrorRate()(
                first_hypotheses, references
            ).item()

        return metrics

    return compute
