import argparse
import os

import torch
import yaml
from transformers import EarlyStoppingCallback, TrainingArguments

from srcs.datasets.vicocktail import Collator, load_vicocktail
from srcs.nets.e2e import get_model as create_model
from srcs.nets.utils import load_backbone_weights
from srcs.spm.spm_train import ensure_unigram
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.trainer import (
    FinetuneTrainer,
    build_metric_fn,
    preprocess_logits_for_metrics,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
PRETRAINED_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "pretrain", "vsr_trlrs2lrs3vox2avsp_base.pth"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "finetune_vsr_12")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=PRETRAINED_CHECKPOINT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, required=True)
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(vocab_size, checkpoint_path):
    model = create_model("auto-vsr", vocab_size, size="large")
    load_backbone_weights(model, checkpoint_path)
    model.finetune()

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_parameters:,}")

    return model


def get_training_args(args, config):
    return TrainingArguments(
        output_dir=args.output_dir,
        label_names=["labels", "label_lengths"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["encoder_lr"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
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


def main():
    args = parse_args()
    config = load_config(args.config)
    finetuning_config = config["finetuning"]
    torch.manual_seed(finetuning_config["seed"])

    datasets = load_vicocktail(
        train_fraction=args.train_fraction,
        validation_fraction=args.val_fraction,
        splits=("train", "val"),
        seed=finetuning_config["seed"],
    )
    train_dataset = datasets["train"]
    validation_dataset = datasets["val"]

    model_path, units_path = ensure_unigram(train_dataset)
    text_transform = TextTransform(model_path, units_path)
    model = load_model(text_transform.vocab_size, args.checkpoint)

    trainer = FinetuneTrainer(
        model=model,
        args=get_training_args(args, finetuning_config),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        train_collator=Collator(text_transform, "train"),
        validation_collator=Collator(text_transform, "val"),
        compute_metrics=build_metric_fn(text_transform),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        encoder_lr=finetuning_config["encoder_lr"],
        ctc_head_lr=finetuning_config["ctc_head_lr"],
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=finetuning_config["early_stopping_patience"],
                early_stopping_threshold=finetuning_config["early_stopping_threshold"],
            )
        ],
    )

    trainer.train()
    trainer.save_state()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)

    print(f"Best validation WER: {trainer.state.best_metric}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Final model: {final_dir}")


if __name__ == "__main__":
    main()
