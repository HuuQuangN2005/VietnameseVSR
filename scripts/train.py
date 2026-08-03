import os
import sys
import argparse
import time

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from srcs.datasets.utils import setup_hf
from srcs.datasets.vicocktail_dataset import add_validation_split

CONFIGS = setup_hf()
import torch
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset as hf_load_dataset,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DATASETS = ["vicocktail"]
VALIDATION_FRACTION = 0.03
SEED = 42


def select_fraction(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Dataset fractions must be greater than 0 and at most 1.")
    if fraction == 1.0:
        return dataset

    sample_count = max(1, round(len(dataset) * fraction))
    return dataset.shuffle(seed=seed).select(range(sample_count))


def load_dataset(args: argparse.Namespace) -> DatasetDict:
    train_ds = []
    val_ds = []

    for name in DATASETS:
        if args.dataset != "all" and args.dataset != name:
            continue

        if name == "vicocktail":
            dataset = hf_load_dataset(
                "nguyenvulebinh/ViCocktail",
                streaming=False,
                cache_dir=CONFIGS["HF_DATASETS_CACHE"],
            )
            dataset = dataset.remove_columns(["__key__", "__url__"])
            dataset = add_validation_split(
                dataset, validation_fraction=VALIDATION_FRACTION, seed=SEED
            )

            train_ds.append(select_fraction(dataset["train"], args.train_fraction, SEED))
            val_ds.append(select_fraction(dataset["validation"], args.val_fraction, SEED))

    if not train_ds or not val_ds:
        raise ValueError(f"No dataset was loaded for: {args.dataset}")

    train_dataset = train_ds[0] if len(train_ds) == 1 else concatenate_datasets(train_ds)
    val_dataset = val_ds[0] if len(val_ds) == 1 else concatenate_datasets(val_ds)

    return DatasetDict({"train": train_dataset, "val": val_dataset})


class Trainer:
    def __init__(
        self,
        datasets: DatasetDict,
        model: torch.nn.Module,
        train_collator,
        val_collator,
        output_dir: str,
        epochs: int = 300,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.005,
        num_workers: int = 0,
    ):
        self.datasets = datasets
        self.model = model
        self.output_dir = output_dir
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_dataloader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collator,
        )
        self.val_dataloader = DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collator,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.best_val_loss = float("inf")
        os.makedirs(self.output_dir, exist_ok=True)

    def move_batch_to_device(self, batch: dict) -> dict:
        return {
            name: value.to(self.device) if torch.is_tensor(value) else value
            for name, value in batch.items()
        }

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        progress = tqdm(
            self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs} [train]"
        )

        for batch in progress:
            batch = self.move_batch_to_device(batch)
            batch_size = batch["videos"].size(0)

            self.optimizer.zero_grad()
            loss, _ = self.model(**batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            avg_loss = total_loss / total_samples
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        return total_loss / total_samples

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        progress = tqdm(self.val_dataloader, desc=f"Epoch {epoch}/{self.epochs} [val]")

        for batch in progress:
            batch = self.move_batch_to_device(batch)
            batch_size = batch["videos"].size(0)
            loss, _ = self.model(**batch)

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            avg_loss = total_loss / total_samples
            progress.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")

        return total_loss / total_samples

    def save_checkpoint(self, name: str, epoch: int, val_loss: float) -> None:
        checkpoint_path = os.path.join(self.output_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

    def run(self) -> None:
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.datasets['train']):,}")
        print(f"Validation samples: {len(self.datasets['val']):,}")

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            epoch_seconds = time.perf_counter() - epoch_start
            remaining_hours = epoch_seconds * (self.epochs - epoch) / 3600

            print(
                f"Epoch {epoch}/{self.epochs} - "
                f"train_loss: {train_loss:.6f} - "
                f"val_loss: {val_loss:.6f} - "
                f"epoch_time: {epoch_seconds / 60:.2f} min - "
                f"estimated_remaining: {remaining_hours:.2f} h"
            )
            self.save_checkpoint("last.pt", epoch, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt", epoch, val_loss)
                print(f"Saved best checkpoint with val_loss: {val_loss:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VSR model.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--output_dir", type=str, default="/checkpoints")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args)
    print(dataset)
    print("Train dataset:", dataset["train"])
    print("Validation dataset:", dataset["val"])
    print(f"Train samples: {len(dataset['train']):,}")
    print(f"Validation samples: {len(dataset['val']):,}")
