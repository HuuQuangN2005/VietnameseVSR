import argparse

import torch
from transformers import TrainingArguments

from srcs.datasets.vicocktail import Collator, load_vicocktail
from srcs.nets.e2e import get_model as create_model
from srcs.nets.utils import load_weights
from srcs.spm.spm_train import ensure_unigram
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.trainer import HFTrainer, build_metric_fn, preprocess_logits_for_metrics
from train import CONFIG_PATH, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--model", choices=("auto-vsr",), default="auto-vsr")
    parser.add_argument("--size", choices=("large", "small"), default="large")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_fraction", type=float, default=1.0)

    return parser.parse_args()


def get_model(args, text_transform):
    model = create_model(args.model, text_transform.vocab_size, size=args.size)

    report = load_weights(model, args.checkpoint)

    print(f"Loaded model checkpoint: {report['loaded']} tensors.")

    return model


def get_evaluation_args(config):
    return TrainingArguments(
        output_dir="checkpoints",
        label_names=["labels", "label_lengths"],
        per_device_eval_batch_size=config["batch_size"],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=config["num_workers"],
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=config["num_workers"] > 0,
        dataloader_prefetch_factor=2 if config["num_workers"] > 0 else None,
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    evaluation_config = config["evaluation"]
    test_dataset = load_vicocktail(test_fraction=args.test_fraction, splits=("test",))[
        "test"
    ]
    model_path, units_path = ensure_unigram()
    text_transform = TextTransform(model_path, units_path)
    model = get_model(args, text_transform)
    test_collator = Collator(text_transform, "test")

    trainer = HFTrainer(
        model=model,
        args=get_evaluation_args(evaluation_config),
        train_dataset=None,
        eval_dataset=test_dataset,
        train_collator=test_collator,
        validation_collator=test_collator,
        compute_metrics=build_metric_fn(text_transform),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    output = trainer.predict(test_dataset)

    print(f"Test samples: {len(test_dataset)}")
    trainer.log_metrics("test", output.metrics)


if __name__ == "__main__":
    main()
