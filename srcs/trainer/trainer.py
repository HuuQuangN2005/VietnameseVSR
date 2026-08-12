# Source (modified): https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/src/custom_trainer.py
# License: CC BY-NC 4.0 (https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/LICENSE)

from contextlib import contextmanager
import time

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from torchmetrics.text import WordErrorRate
from transformers import Trainer
from transformers.trainer_pt_utils import LengthGroupedSampler

from srcs.nets.utils import ctc_decode


class HFTrainer(Trainer):
    def __init__(self, *args, validation_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.remove_unused_columns:
            raise ValueError("remove_unused_columns must be False.")

        self.validation_collator = validation_collator or self.data_collator
        self._train_lengths = None
        self._diagnostic_totals = {"train": {}, "eval": {}}

        if (
            self.train_dataset is not None
            and hasattr(self.train_dataset, "column_names")
            and self.args.length_column_name in self.train_dataset.column_names
        ):
            self._train_lengths = [
                int(value) for value in self.train_dataset[self.args.length_column_name]
            ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        if isinstance(outputs, dict):
            mode = "train" if model.training else "eval"
            totals = self._diagnostic_totals[mode]
            raw_ctc_loss = outputs.get("raw_ctc_loss")
            refined_logits = outputs.get("logits")
            first_logits = outputs.get("first_logits")
            input_lengths = outputs.get("input_lengths")

            if raw_ctc_loss is not None and refined_logits is not None:
                batch_size = refined_logits.size(0)
                totals["raw_ctc_loss_sum"] = (
                    totals.get("raw_ctc_loss_sum", 0.0)
                    + raw_ctc_loss.detach().float().item() * batch_size
                )
                totals["sample_count"] = totals.get("sample_count", 0) + batch_size

            if (
                refined_logits is not None
                and first_logits is not None
                and input_lengths is not None
            ):
                time = refined_logits.size(1)
                valid_mask = torch.arange(time, device=refined_logits.device).unsqueeze(
                    0
                ) < input_lengths.unsqueeze(1)
                refined_ids = refined_logits.detach().argmax(dim=-1)
                first_ids = first_logits.detach().argmax(dim=-1)
                totals["flip_count"] = totals.get("flip_count", 0) + int(
                    refined_ids[valid_mask].ne(first_ids[valid_mask]).sum().item()
                )
                totals["frame_count"] = totals.get("frame_count", 0) + int(
                    valid_mask.sum().item()
                )

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs and self._diagnostic_totals["train"]:
            self._log_diagnostics(logs, "train")

        if "eval_loss" in logs and self._diagnostic_totals["eval"]:
            self._log_diagnostics(logs, "eval")

        super().log(logs, start_time)

    def _log_diagnostics(self, logs, mode):
        totals = self._diagnostic_totals[mode]
        prefix = "eval_" if mode == "eval" else ""
        sample_count = totals.get("sample_count", 0)
        frame_count = totals.get("frame_count", 0)

        if sample_count:
            logs[prefix + "raw_ctc_loss"] = totals["raw_ctc_loss_sum"] / sample_count

        if frame_count:
            logs[prefix + "refined_flip_rate"] = totals["flip_count"] / frame_count

        totals.clear()

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset

        if self.args.train_sampling_strategy != "group_by_length" or dataset is None:
            return super()._get_train_sampler(train_dataset)

        if dataset is self.train_dataset and self._train_lengths is not None:
            lengths = self._train_lengths
        elif (
            hasattr(dataset, "column_names")
            and self.args.length_column_name in dataset.column_names
        ):
            lengths = [int(value) for value in dataset[self.args.length_column_name]]
        else:
            return super()._get_train_sampler(train_dataset)

        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=dataset,
            lengths=lengths,
        )

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset)

    def _save_checkpoint(self, model, trial):
        report = self.is_world_process_zero()

        if report:
            print(f"Saving checkpoint at step {self.state.global_step}...")

        start = time.perf_counter()
        super()._save_checkpoint(model, trial)

        if report:
            elapsed = time.perf_counter() - start
            print(f"Checkpoint saved in {elapsed:.2f} seconds")

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
