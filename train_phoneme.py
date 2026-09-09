import argparse
import os

from srcs.datasets.vicocktail import load_vicocktail
from srcs.datasets.collator import PhonemeCollator
from srcs.nets.e2e import get_model
from srcs.nlp.text_transform import PhonemeTransform
from srcs.trainer.trainer import Trainer
from srcs.trainer.utils import create_dataloader, load_config, set_seed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument(
        "--model",
        choices=[
            "IndependentMCTCVSR",
            "CascadedMCTCVSR",
            "RhymeGuidedMCTCVSR",
        ],
        default="IndependentMCTCVSR",
    )
    parser.add_argument("--epochs", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)["training"]
    set_seed(config["seed"])

    datasets = load_vicocktail(
        split="train",
        fraction=args.train_fraction,
        seed=config["seed"],
    )

    text_transform = PhonemeTransform(datasets["train"])
    model = get_model(
        args.model,
        text_transform.vocab_size,
        checkpoint=args.checkpoint,
    )

    train_dataloader = create_dataloader(
        datasets["train"],
        PhonemeCollator("train", text_transform),
        config,
        shuffle=True,
    )
    validation_dataloader = create_dataloader(
        datasets["val"],
        PhonemeCollator("val", text_transform),
        config,
    )

    trainer = Trainer(
        model=model,
        text_transform=text_transform,
        config=config,
    )
    output_dir = args.output_dir or os.path.join(CHECKPOINT_DIR, args.model)
    trainer.train(train_dataloader, validation_dataloader, args.epochs, output_dir)


if __name__ == "__main__":
    main()
