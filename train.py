import argparse
import os

import torch
import yaml
from transformers import TrainingArguments

from srcs.datasets.vicocktail import Collator, load_vicocktail
from srcs.nets.e2e import get_model as create_model
from srcs.nets.utils import parameter_count
from srcs.spm.spm_train import ensure_unigram, get_paths
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.HF_Trainer import (
    HFTrainer,
    build_metric_fn,
    preprocess_logits_for_metrics,
)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_model(args, text_transform, model_config):
    model = create_model(
        args.model,
        text_transform.vocab_size,
        pretrained_weights=args.pretrained_weights,
        freeze_frontend=args.freeze_frontend,
        **model_config,
    )
    print(f"Model parameters: {parameter_count(model):,}")
    print(f"Trainable parameters: {parameter_count(model, True):,}")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--model", choices=("e2e",), required=True)
    parser.add_argument("--dataset", choices=("vicocktail",), default="vicocktail")
    parser.add_argument("--output_dir", default="/checkpoints")
    parser.add_argument("--train_fraction", type=float, default=0.2)
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--pretrained_weights")
    parser.add_argument("--resume_checkpoint")
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
        train_sampling_strategy="group_by_length",
        length_column_name="video_length",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=config["save_total_limit"],
        report_to="none",
        seed=config["seed"],
        data_seed=config["seed"],
    )


def get_text_transform(train_dataset):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    model_path = None
    units_path = None
    if rank == 0:
        model_path, units_path = get_paths()
        if not os.path.isfile(model_path):
            tokenizer_dataset = load_vicocktail(train_fraction=1.0, splits=("train",))[
                "train"
            ]
        else:
            tokenizer_dataset = train_dataset
        model_path, units_path = ensure_unigram(tokenizer_dataset)
    if distributed:
        torch.distributed.barrier()
    if rank != 0:
        model_path, units_path = ensure_unigram()
    return TextTransform(model_path, units_path)


def main():
    args = parse_args()
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
    model = get_model(args, text_transform, config["model"])

    trainer = HFTrainer(
        model=model,
        args=get_training_args(args, training_config),
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["val"],
        data_collator=Collator(text_transform, "train"),
        validation_collator=Collator(text_transform, "val"),
        compute_metrics=build_metric_fn(text_transform),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "final"))
    trainer.save_state()


if __name__ == "__main__":
    main()
