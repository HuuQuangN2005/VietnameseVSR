import argparse
import os

from srcs.datasets.vicocktail import load_vicocktail
from srcs.nets.backend.refiner.transform import RefinerTransform
from srcs.nets.e2e import get_model
from srcs.spm.spm_train import ensure_unigram
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.trainer import RefinerTrainer
from srcs.trainer.utils import create_dataloader, load_config, set_seed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "refiner")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--vsr_checkpoint", required=True)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)["training"]
    set_seed(config["seed"])

    datasets = load_vicocktail(
        train_fraction=args.train_fraction,
        validation_fraction=args.val_fraction,
        splits=("train", "val"),
        seed=config["seed"],
    )
    train_dataset = datasets["train"]
    validation_dataset = datasets["val"]

    model_path, units_path = ensure_unigram(train_dataset)
    text_transform = TextTransform(model_path, units_path)
    base_model = get_model(
        "auto-vsr", text_transform.vocab_size, checkpoint=args.vsr_checkpoint
    )
    model = get_model(
        "refiner",
        text_transform.vocab_size,
        blank_id=text_transform.blank_id,
        ignore_id=text_transform.ignore_id,
    )

    train_dataloader = create_dataloader(
        train_dataset, text_transform, "train", config, shuffle=True
    )
    validation_dataloader = create_dataloader(
        validation_dataset, text_transform, "val", config
    )

    amp = config.get("amp", True)
    transform = RefinerTransform(probability=0.1)

    trainer = RefinerTrainer(
        base_model,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        text_transform=text_transform,
        config=config,
        amp=amp,
        transform=transform,
    )
    trainer.train(train_dataloader, validation_dataloader, args.epochs, args.output_dir)


if __name__ == "__main__":
    main()
