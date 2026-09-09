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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_fraction", type=float, default=1.0)
    parser.add_argument(
        "--model",
        choices=[
            "IndependentMCTCVSR",
            "CascadedMCTCVSR",
            "RhymeGuidedMCTCVSR",
        ],
        default="IndependentMCTCVSR",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    evaluation_config = config["evaluation"]
    seed = config["training"]["seed"]
    set_seed(seed)

    test_dataset = load_vicocktail(
        split="test", fraction=args.test_fraction, seed=seed
    )["test"]
    text_transform = PhonemeTransform()
    model = get_model(
        args.model,
        text_transform.vocab_size,
        checkpoint=args.checkpoint,
    )

    dataloader = create_dataloader(
        test_dataset,
        PhonemeCollator("test", text_transform),
        evaluation_config,
    )
    trainer = Trainer(
        model=model,
        text_transform=text_transform,
        config=evaluation_config,
    )
    metrics = trainer.run_one_epoch(dataloader, training=False, description="Testing")

    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test loss: {metrics['loss']:.6f}")
    print(f"Test PER_I: {metrics['per_i']:.6f}")
    print(f"Test PER_R: {metrics['per_r']:.6f}")
    print(f"Test PER_T: {metrics['per_t']:.6f}")


if __name__ == "__main__":
    main()
