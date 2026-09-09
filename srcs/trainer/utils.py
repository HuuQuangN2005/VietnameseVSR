import json
import math
import random

import torch
import yaml
from torch.utils.data import DataLoader, Sampler
from torchmetrics.text import WordErrorRate

from srcs.nets.loss.mctc import mctc_decode


class LengthBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=False, seed=42):
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        lengths = [int(value) for value in lengths]
        indices = sorted(range(len(lengths)), key=lengths.__getitem__)

        self.batches = [
            indices[start : start + batch_size]
            for start in range(0, len(indices), batch_size)
        ]

    def __iter__(self):
        batches = self.batches.copy()
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
            self.epoch += 1
        yield from batches

    def __len__(self):
        return len(self.batches)


def load_configuration(file_path):
    with open(file_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loader(dataset, collator, configuration, shuffle=False):
    worker_count = configuration["num_workers"]
    batch_sampler = LengthBatchSampler(
        dataset["video_length"],
        configuration["batch_size"],
        shuffle,
        configuration.get("seed", 42),
    )

    options = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "collate_fn": collator,
        "num_workers": worker_count,
        "pin_memory": torch.cuda.is_available(),
    }

    if worker_count > 0:
        options.update(persistent_workers=True, prefetch_factor=2)

    return DataLoader(**options)


def move_batch(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def create_metrics(text_transform):
    return {name: WordErrorRate() for name in text_transform.metric_names}


def get_metric_results(metrics):
    return {name: metric.compute().item() for name, metric in metrics.items()}


def update_metrics(metrics, outputs, batch, text_transform):
    predicted_ids = mctc_decode(outputs["logits"], outputs["input_lengths"])
    predictions = [text_transform.decode_for_metrics(ids) for ids in predicted_ids]

    references = [
        text_transform.decode_for_metrics(label[: int(length)])
        for label, length in zip(
            batch["labels"].detach().cpu(),
            batch["label_lengths"].detach().cpu(),
        )
    ]

    for name, metric in metrics.items():
        metric.update(
            [prediction[name] for prediction in predictions],
            [reference[name] for reference in references],
        )


def create_learning_rate_scheduler(
    optimizer, data_loader, epoch_count, accumulation_steps, warmup
):
    total_steps = math.ceil(len(data_loader) / accumulation_steps) * epoch_count
    warmup_steps = round(total_steps * warmup) if warmup < 1.0 else int(warmup)

    def learning_rate_scale(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)


def save_checkpoint(model, file_path, epoch, metrics):
    state = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics}
    torch.save(state, file_path)


def save_history(history, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
