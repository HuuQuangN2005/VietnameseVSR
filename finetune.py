import argparse
import os

from srcs.datasets.vicocktail import load_vicocktail
from srcs.nets.e2e import get_model
from srcs.spm.spm_train import ensure_unigram
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.trainer import FinetuneTrainer
from srcs.trainer.utils import (
    create_dataloader,
    load_config,
    set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
PRETRAINED_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "pretrain", "vsr_trlrs2lrs3vox2avsp_base.pth"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "finetune_vsr_12")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=PRETRAINED_CHECKPOINT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)["finetuning"]
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

    model = get_model("auto-vsr", text_transform.vocab_size, checkpoint=args.checkpoint)

    train_dataloader = create_dataloader(
        train_dataset, text_transform, "train", config, shuffle=True
    )
    validation_dataloader = create_dataloader(
        validation_dataset, text_transform, "val", config
    )

    amp = config.get("amp", True)

    trainer = FinetuneTrainer(
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        text_transform=text_transform,
        config=config,
        amp=amp,
    )
    trainer.train(train_dataloader, validation_dataloader, args.epochs, args.output_dir)


if __name__ == "__main__":
    main()
