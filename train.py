import argparse
import os
import torch
import yaml

from transformers import EarlyStoppingCallback, TrainingArguments
from srcs.datasets.vicocktail import Collator, load_vicocktail
from srcs.nets.e2e import get_model as create_model
from srcs.nets.lora import (
    LORA_TRAINABLE_PATTERNS,
    apply_lora_config,
    print_trainable_parameters,
)
from srcs.nets.utils import load_backbone_weights
from srcs.spm.spm_train import ensure_unigram
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.trainer import HFTrainer, build_metric_fn, preprocess_logits_for_metrics

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_args(args):
    initialization_count = sum(
        (bool(args.checkpoint), bool(args.resume_checkpoint), args.from_scratch)
    )

    if initialization_count != 1:
        raise ValueError(
            "Use exactly one of --checkpoint, --resume_checkpoint, or --from_scratch."
        )


def get_model(args, text_transform, config):
    model = create_model(
        args.model, text_transform.vocab_size, size=args.size, **config["vsr"]
    )

    if args.checkpoint:
        report = load_backbone_weights(model, args.checkpoint)
        print(
            f"Loaded backbone: {len(report['loaded'])} tensors; "
            f"skipped: {len(report['skipped'])}."
        )
    elif args.from_scratch:
        print("Training VSR from scratch.")

    if args.freeze_frontend:
        model.freeze_frontend()

    if args.lora:
        replaced = apply_lora_config(model, config["lora"])
        print(f"Applied LoRA to {len(replaced)} linear modules.")

    print_trainable_parameters(model)

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--model", choices=("auto-vsr",), default="auto-vsr")
    parser.add_argument("--size", choices=("large", "small"), default="large")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--output_dir", default="/checkpoints")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume_checkpoint")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--freeze_frontend", action="store_true")
    return parser.parse_args()


def get_training_args(args, config):

    return TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        label_names=["labels", "label_lengths"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        max_grad_norm=config["max_grad_norm"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config["logging_steps"],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=config["num_workers"],
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=config["num_workers"] > 0,
        dataloader_prefetch_factor=2 if config["num_workers"] > 0 else None,
        train_sampling_strategy="group_by_length",
        length_column_name="video_length",
        load_best_model_at_end=True,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        save_total_limit=config["save_total_limit"],
        seed=config["seed"],
        data_seed=config["seed"],
    )


def get_text_transform(train_dataset):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    rank = torch.distributed.get_rank() if distributed else 0
    model_path = None
    units_path = None

    if rank == 0:
        model_path, units_path = ensure_unigram(train_dataset)

    if distributed:
        torch.distributed.barrier()

    if rank != 0:
        model_path, units_path = ensure_unigram()

    return TextTransform(model_path, units_path)


def main():
    args = parse_args()
    validate_args(args)
    print(f"Config: {os.path.abspath(args.config)}")
    config = load_config(args.config)
    training_config = config["training"]
    torch.manual_seed(training_config["seed"])

    dataset_splits = load_vicocktail(
        train_fraction=args.train_fraction,
        validation_fraction=args.val_fraction,
        test_fraction=1.0,
        splits=("train", "val"),
    )

    text_transform = get_text_transform(dataset_splits["train"])
    model = get_model(args, text_transform, config)

    trainer = HFTrainer(
        model=model,
        args=get_training_args(args, training_config),
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["val"],
        train_collator=Collator(text_transform, "train"),
        validation_collator=Collator(text_transform, "val"),
        compute_metrics=build_metric_fn(text_transform),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        allowed_trainable_names=LORA_TRAINABLE_PATTERNS if args.lora else None,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config["early_stopping_patience"],
                early_stopping_threshold=training_config["early_stopping_threshold"],
            )
        ],
    )

    trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    final_dir = os.path.join(args.output_dir, "final")

    trainer.save_model(final_dir)
    trainer.save_state()

    if trainer.is_world_process_zero():
        completed_epoch = trainer.state.epoch or 0.0
        stopped_early = completed_epoch + 1e-6 < args.epochs
        status = "Early stopping" if stopped_early else "Training completed"

        print(f"{status} at epoch {completed_epoch:g}")

        if trainer.state.best_metric is not None:
            print(f"Best validation WER: {trainer.state.best_metric:.6f}")

        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"Final model: {final_dir}")


if __name__ == "__main__":
    main()
