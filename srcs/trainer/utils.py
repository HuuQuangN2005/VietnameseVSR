import json
import random


import math
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from srcs.datasets.vicocktail import Collator
from srcs.nets.backend.ctc import ctc_decode


class LengthBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, seed=42):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        lengths = [int(length) for length in lengths]
        indices = sorted(range(len(lengths)), key=lengths.__getitem__)
        self.batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]

    def __iter__(self):
        batches = self.batches.copy()
        generator = random.Random(self.seed + self.epoch)
        generator.shuffle(batches)
        self.epoch += 1

        yield from batches

    def __len__(self):
        return len(self.batches)


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(dataset, text_transform, split, config, shuffle=False):
    num_workers = config["num_workers"]
    common_args = {
        "dataset": dataset,
        "collate_fn": Collator(text_transform, split),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    if shuffle and "video_length" in dataset.column_names:
        video_lengths = list(dataset["video_length"])
        batch_sampler = LengthBatchSampler(
            video_lengths, config["batch_size"], config["seed"]
        )
        return DataLoader(batch_sampler=batch_sampler, **common_args)

    return DataLoader(batch_size=config["batch_size"], shuffle=shuffle, **common_args)


def move_batch(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def update_wer(metric, outputs, batch, text_transform):
    token_ids = ctc_decode(
        outputs["logits"], outputs["input_lengths"], text_transform.blank_id
    )
    hypotheses = [text_transform.decode(item) for item in token_ids]
    references = [
        text_transform.decode(label[: int(length)])
        for label, length in zip(
            batch["labels"].detach().cpu(), batch["label_lengths"].detach().cpu()
        )
    ]
    metric.update(hypotheses, references)


def create_grad_scaler(device, enabled=True):
    return torch.amp.GradScaler(
        "cuda", enabled=enabled and device.type == "cuda", init_scale=1024.0
    )


def resolve_warmup_steps(value, total_steps):
    if 0.0 <= value < 1.0:
        return round(total_steps * value)

    return int(value)


def create_cosine_scheduler(optimizer, total_steps, warmup_value=0):
    if total_steps <= 0:
        raise ValueError("total_steps must be greater than zero.")

    warmup_steps = resolve_warmup_steps(warmup_value, total_steps)

    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps.")

    def lr_scale(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        cosine_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / cosine_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


def optimizer_step_count(dataloader, epochs, accumulation_steps):
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be greater than zero.")

    return math.ceil(len(dataloader) / accumulation_steps) * epochs


def save_checkpoint(model, path, epoch, metrics):
    torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, path)


def save_history(history, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
