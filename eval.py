import argparse
import os

import torch
import yaml
from transformers import TrainingArguments

from srcs.datasets.vicocktail import Collator, load_vicocktail
from srcs.nets.e2e import get_model
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.HF_Trainer import (
    HFTrainer,
    build_metric_fn,
    preprocess_logits_for_metrics,
)

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=("e2e",), required=True)
    parser.add_argument("--dataset", choices=("vicocktail",), default="vicocktail")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="/checkpoints")
    parser.add_argument("--test_fraction", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    evaluation_config = config["evaluation"]
    dataset_splits = load_vicocktail(
        train_fraction=1.0,
        validation_fraction=1.0,
        test_fraction=args.test_fraction,
        splits=("test",),
    )
    if "test" not in dataset_splits:
        raise ValueError("The selected dataset has no test split.")
    text_transform = TextTransform()
    model = get_model(
        args.model,
        text_transform.vocab_size,
        checkpoint_path=args.checkpoint,
        **config["model"],
    )
    evaluation_args = TrainingArguments(
        output_dir=args.output_dir,
        label_names=["labels", "label_lengths"],
        per_device_eval_batch_size=evaluation_config["batch_size"],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=evaluation_config["num_workers"],
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
    )
    test_collator = Collator(text_transform, "test")
    trainer = HFTrainer(
        model=model,
        args=evaluation_args,
        eval_dataset=dataset_splits["test"],
        data_collator=test_collator,
        validation_collator=test_collator,
        compute_metrics=build_metric_fn(text_transform),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    metrics = trainer.evaluate(metric_key_prefix="test")
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
