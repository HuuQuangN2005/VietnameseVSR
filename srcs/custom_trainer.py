from contextlib import contextmanager
from typing import Any, Iterator

from torch.utils.data import DataLoader
from transformers import Trainer


class AVSRTrainer(Trainer):
    def __init__(
        self,
        *args,
        valid_data_collator: Any | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if self.args.remove_unused_columns:
            raise ValueError(
                "remove_unused_columns must be False because the VSR data collator "
                "requires the raw video and label columns."
            )

        self.valid_data_collator = (
            valid_data_collator
            if valid_data_collator is not None
            else self.data_collator
        )

    @contextmanager
    def _use_valid_data_collator(self) -> Iterator[None]:
        train_data_collator = self.data_collator
        self.data_collator = self.valid_data_collator

        try:
            yield
        finally:
            self.data_collator = train_data_collator

    def get_eval_dataloader(self, eval_dataset: Any | None = None) -> DataLoader:
        with self._use_valid_data_collator():
            return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset: Any) -> DataLoader:
        with self._use_valid_data_collator():
            return super().get_test_dataloader(test_dataset)
