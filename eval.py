import argparse
import os

from srcs.datasets.vicocktail import load_vicocktail
from srcs.nets.e2e import get_model
from srcs.spm.spm_train import ensure_unigram
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.trainer import RefinerTrainer
from srcs.trainer.utils import create_dataloader, load_config
from train import CONFIG_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--vsr_checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_fraction", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    evaluation_config = config["evaluation"]
    seed = config["training"]["seed"]

    test_dataset = load_vicocktail(
        test_fraction=args.test_fraction,
        splits=("test",),
        seed=seed,
    )["test"]
    model_path, units_path = ensure_unigram()
    text_transform = TextTransform(model_path, units_path)

    base_model = get_model(
        "auto-vsr",
        text_transform.vocab_size,
        checkpoint=args.vsr_checkpoint,
    )
    refiner = get_model(
        "refiner",
        text_transform.vocab_size,
        checkpoint=args.checkpoint,
        blank_id=text_transform.blank_id,
        ignore_id=text_transform.ignore_id,
    )

    dataloader = create_dataloader(
        test_dataset,
        text_transform,
        "test",
        evaluation_config,
    )
    trainer = RefinerTrainer(
        base_model,
        refiner,
        optimizer=None,
        scheduler=None,
        scaler=None,
        text_transform=text_transform,
        config=evaluation_config,
        amp=evaluation_config.get("amp", True),
    )
    metrics = trainer.run_one_epoch(dataloader, training=False, description="Testing")

    print(f"VSR checkpoint: {os.path.abspath(args.vsr_checkpoint)}")
    print(f"Refiner checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test loss: {metrics['loss']:.6f}")
    print(f"Test WER: {metrics['wer']:.6f}")


if __name__ == "__main__":
    main()
