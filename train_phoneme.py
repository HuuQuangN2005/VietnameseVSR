import argparse
import os

from srcs.datasets.vicocktail import load_vicocktail
from srcs.datasets.collator import PhonemeCollator
from srcs.nets.e2e import get_model
from srcs.nlp.text_transform import PhonemeTransform
from srcs.trainer.trainer import Trainer
from srcs.trainer.utils import create_data_loader, load_configuration, set_seed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIGURATION_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DEFAULT_CHECKPOINT_DIRECTORY = os.path.join(PROJECT_ROOT, "checkpoints")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION_PATH)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_directory")
    parser.add_argument("--fraction", type=float, default=1.0)
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
    arguments = parse_arguments()
    training_configuration = load_configuration(arguments.configuration)["training"]
    set_seed(training_configuration["seed"])

    dataset_splits = load_vicocktail(
        split="train",
        fraction=arguments.fraction,
        seed=training_configuration["seed"],
    )

    text_transform = PhonemeTransform(dataset_splits["train"])
    model = get_model(
        arguments.model,
        text_transform.vocab_size,
        checkpoint=arguments.checkpoint,
    )

    training_data_loader = create_data_loader(
        dataset_splits["train"],
        PhonemeCollator("train", text_transform),
        training_configuration,
        shuffle=True,
    )
    validation_data_loader = create_data_loader(
        dataset_splits["val"],
        PhonemeCollator("val", text_transform),
        training_configuration,
    )

    trainer = Trainer(
        model=model,
        text_transform=text_transform,
        configuration=training_configuration,
    )

    output_directory = arguments.output_directory or os.path.join(
        DEFAULT_CHECKPOINT_DIRECTORY,
        arguments.model,
    )

    trainer.train(
        training_data_loader,
        validation_data_loader,
        arguments.epochs,
        output_directory,
    )


if __name__ == "__main__":
    main()
