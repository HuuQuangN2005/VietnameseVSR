import json
import math
import os
import random

import torch
import yaml
from torch.utils.data import DataLoader, Sampler
from torchmetrics.text import WordErrorRate


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


def load_configs(file_path):
    with open(file_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loader(dataset, collator, configs, shuffle=False):
    worker_count = configs["num_workers"]
    batch_sampler = LengthBatchSampler(
        dataset["video_length"],
        configs["batch_size"],
        shuffle,
        configs.get("seed", 42),
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


def create_lr_scheduler(optimizer, loader, epoch_count, accum_steps, warmup):
    total_steps = math.ceil(len(loader) / accum_steps) * epoch_count
    warmup_steps = round(total_steps * warmup) if warmup < 1.0 else int(warmup)

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


def save_ckpt(model, file_path, epoch, metrics, trainer=None):
    state = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics}

    if trainer is not None:
        state["optimizer"] = trainer.optimizer.state_dict()
        state["scheduler"] = trainer.scheduler.state_dict()
        state["scaler"] = trainer.scaler.state_dict()

    torch.save(state, file_path)


def load_history(file_path):
    if not os.path.isfile(file_path):
        return []

    with open(file_path, encoding="utf-8") as file:
        return json.load(file)


def save_history(history, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
