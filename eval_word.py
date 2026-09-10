import argparse
import os

from srcs.datasets.vicocktail import load_vicocktail
from srcs.datasets.collator import WordCollator
from srcs.nets.e2e import get_model
from srcs.nlp.text_transform import WordTransform, vocabulary_paths
from srcs.trainer.trainer import Trainer
from srcs.trainer.utils import create_data_loader, load_configuration, set_seed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIGURATION_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION_PATH)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--model", choices=["WordCTCVSR"], default="WordCTCVSR")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    configuration = load_configuration(arguments.configuration)
    evaluation_configuration = configuration["evaluation"]
    seed = configuration["training"]["seed"]
    set_seed(seed)

    test_dataset = load_vicocktail(
        split="test",
        fraction=arguments.fraction,
        seed=seed,
    )["test"]

    checkpoint_directory = (
        arguments.checkpoint
        if os.path.isdir(arguments.checkpoint)
        else os.path.dirname(os.path.abspath(arguments.checkpoint))
    )
    vocabulary_directory = os.path.join(checkpoint_directory, "vocab")

    if not os.path.isdir(vocabulary_directory):
        raise FileNotFoundError(
            f"No vocabulary next to the checkpoint: {vocabulary_directory}. "
            "It is written there by train_word.py; evaluating against a different "
            "vocabulary silently reports wrong metrics."
        )

    text_transform = WordTransform(
        word_path=vocabulary_paths(vocabulary_directory)["word_path"],
    )

    model = get_model(
        arguments.model,
        text_transform.vocab_size,
        checkpoint=arguments.checkpoint,
    )

    test_data_loader = create_data_loader(
        test_dataset,
        WordCollator("test", text_transform),
        evaluation_configuration,
    )

    trainer = Trainer(
        model=model,
        text_transform=text_transform,
        configuration=evaluation_configuration,
    )

    metrics = trainer.run_epoch(
        test_data_loader,
        training=False,
        description="Testing",
    )

    print(f"Checkpoint: {os.path.abspath(arguments.checkpoint)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test loss: {metrics['loss']:.6f}")
    print(f"Test WER: {metrics['wer']:.6f}")


if __name__ == "__main__":
    main()
