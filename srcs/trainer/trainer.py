# Source (modified): https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/src/custom_trainer.py
# License: CC BY-NC 4.0 (https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/LICENSE)

from contextlib import contextmanager

import torch
from torchmetrics.text import WordErrorRate

from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler

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

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        length_column = self.args.length_column_name

        if (
            self.args.train_sampling_strategy != "group_by_length"
            or dataset is None
            or not hasattr(dataset, "column_names")
            or length_column not in dataset.column_names
        ):
            return super()._get_train_sampler(train_dataset)

        lengths = [int(length) for length in dataset[length_column]]

        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=lengths,
        )

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


class FinetuneTrainer(HFTrainer):
    def __init__(self, *args, encoder_lr, ctc_head_lr, **kwargs):
        self.encoder_lr = encoder_lr
        self.ctc_head_lr = ctc_head_lr
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model if model is None else model
        encoder_parameters = list(model.encoder.encoders[-2:].parameters())
        ctc_head_parameters = list(model.ctc.ctc_lo.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": self.encoder_lr},
                {"params": ctc_head_parameters, "lr": self.ctc_head_lr},
            ],
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer


def preprocess_logits_for_metrics(logits, labels):
    del labels
    logits, input_lengths = logits
    return logits.argmax(dim=-1), input_lengths


def build_metric_fn(text_transform):
    def compute(eval_pred):
        frame_ids, input_lengths = eval_pred.predictions
        token_ids = ctc_decode(
            torch.as_tensor(frame_ids),
            torch.as_tensor(input_lengths),
            text_transform.blank_id,
        )
        labels, label_lengths = eval_pred.label_ids
        hypotheses = [text_transform.decode(item) for item in token_ids]
        references = [
            text_transform.decode(label[: int(length)])
            for label, length in zip(labels, label_lengths)
        ]

        return {"wer": WordErrorRate()(hypotheses, references).item()}

    return compute
