import argparse

import torch
from transformers import TrainingArguments

from srcs.datasets.vicocktail import Collator, load_vicocktail
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.trainer import HFTrainer, build_metric_fn, preprocess_logits_for_metrics
from train import CONFIG_PATH, PRETRAINED_PATH, get_model, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument(
        "--model", choices=("baseline", "teacher", "refiner"), required=True
    )
    parser.add_argument("--dataset", choices=("vicocktail",), default="vicocktail")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/eval")
    parser.add_argument("--test_fraction", type=float, default=1.0)
    parser.add_argument("--autoavsr_checkpoint", default=PRETRAINED_PATH)
    parser.add_argument("--pretrained_weights")
    parser.add_argument("--baseline_checkpoint")
    parser.add_argument(
        "--freeze_baseline", action=argparse.BooleanOptionalAction, default=True
    )
    parser.set_defaults(freeze_frontend=False, resume_checkpoint=None)

    return parser.parse_args()


def get_evaluation_args(args, config):
    return TrainingArguments(
        output_dir=args.output_dir,
        label_names=["labels", "label_lengths"],
        per_device_eval_batch_size=config["batch_size"],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=config["num_workers"],
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=config["num_workers"] > 0,
        dataloader_prefetch_factor=2 if config["num_workers"] > 0 else None,
        report_to="none",
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    evaluation_config = config["evaluation"]
    test_dataset = load_vicocktail(test_fraction=args.test_fraction, splits=("test",))[
        "test"
    ]
    text_transform = TextTransform()
    model = get_model(args, text_transform, config)
    test_collator = Collator(text_transform, "test")
    allowed_trainable_names = None

    if args.model == "teacher":
        allowed_trainable_names = ("ctc.", ".lora_a.", ".lora_b.")

    trainer = HFTrainer(
        model=model,
        args=get_evaluation_args(args, evaluation_config),
        train_dataset=None,
        eval_dataset=test_dataset,
        train_collator=test_collator,
        validation_collator=test_collator,
        compute_metrics=build_metric_fn(text_transform),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        allowed_trainable_names=allowed_trainable_names,
    )
    output = trainer.predict(test_dataset)

    print(f"Test samples: {len(test_dataset)}")
    trainer.log_metrics("test", output.metrics)


if __name__ == "__main__":
    main()
