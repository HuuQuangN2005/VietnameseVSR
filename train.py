import argparse
import os

from srcs.datasets.utils import filter_by_length
from srcs.datasets.vicocktail import load_vicocktail
from srcs.datasets.collator import PhonemeCollator
from srcs.nets.e2e import get_model
from srcs.nlp.text_transform import PhonemeTransform, vocab_paths
from srcs.trainer.trainer import DecoderTrainer, EncoderTrainer
from srcs.trainer.utils import create_loader, load_configs, set_seed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIGS_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
TRAINERS = {"MCTCVSR": EncoderTrainer, "DecoderVSR": DecoderTrainer}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=DEFAULT_CONFIGS_PATH)
    parser.add_argument("--ckpt")
    parser.add_argument("--output_dir")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--model",
        choices=[
            "MCTCVSR",
            "DecoderVSR",
        ],
        default="MCTCVSR",
    )
    parser.add_argument("--visual_pretrained")
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--ctc_weight", type=float)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    for path in (args.ckpt, args.visual_pretrained):
        if path and not os.path.exists(path):
            raise FileNotFoundError(path)

    train_configs = load_configs(args.configs)["training"]
    set_seed(train_configs["seed"])

    output_dir = args.output_dir or os.path.join(
        DEFAULT_CKPT_DIR,
        args.model,
    )
    vocab_dir = os.path.join(output_dir, "vocab")
    os.makedirs(vocab_dir, exist_ok=True)

    dataset_splits = load_vicocktail(
        split="train",
        fraction=args.fraction,
        seed=train_configs["seed"],
    )

    if args.max_frames:
        for split in ("train", "val"):
            dataset_splits[split] = filter_by_length(
                dataset_splits[split], args.max_frames
            )

    text_transform = PhonemeTransform(
        train_dataset=dataset_splits["train"],
        **vocab_paths(vocab_dir),
    )
    model_config = {}

    if args.ctc_weight is not None:
        model_config["ctc_weight"] = args.ctc_weight

    model = get_model(
        args.model,
        text_transform.vocab_size,
        ckpt=args.ckpt,
        visual_pretrained=args.visual_pretrained,
        **model_config,
    )

    train_loader = create_loader(
        dataset_splits["train"],
        PhonemeCollator("train", text_transform),
        train_configs,
        shuffle=True,
    )
    val_loader = create_loader(
        dataset_splits["val"],
        PhonemeCollator("val", text_transform),
        train_configs,
    )

    trainer = TRAINERS[args.model](
        model=model,
        text_transform=text_transform,
        configs=train_configs,
    )

    trainer.train(
        train_loader,
        val_loader,
        args.epochs,
        output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
