import argparse
import os
import torch
import yaml

from transformers import EarlyStoppingCallback, TrainingArguments
from srcs.datasets.vicocktail import Collator, load_vicocktail
from srcs.nets.e2e import VisualRefinerVSRModel, get_model as create_model
from srcs.nets.lora import (
    apply_lora,
    print_trainable_parameters,
    trainable_parameter_names,
)
from srcs.nets.utils import load_backbone_weights, load_weights, parameter_count
from srcs.spm.spm_train import ensure_unigram
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.trainer import HFTrainer, build_metric_fn, preprocess_logits_for_metrics

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_args(args):
    initialization_count = sum(
        (
            bool(args.checkpoint),
            bool(args.resume_checkpoint),
            args.from_scratch,
        )
    )

    if initialization_count != 1:
        raise ValueError(
            "Use exactly one of --checkpoint, --resume_checkpoint, or --from_scratch."
        )

    if args.from_scratch and args.model != "baseline":
        raise ValueError("--from_scratch is only supported for the baseline model.")

    if args.freeze_frontend and args.model != "baseline":
        raise ValueError("--freeze_frontend is only supported for the baseline model.")

    if not args.freeze_baseline and args.model != "refiner":
        raise ValueError("--no-freeze_baseline is only supported for the refiner model.")


def get_model(args, text_transform, config):
    if args.model == "baseline":
        model = create_model(
            args.model,
            text_transform.vocab_size,
            **config["baseline"],
        )

        if args.checkpoint:
            report = load_backbone_weights(model, args.checkpoint)
            print(
                f"Loaded backbone: {len(report['loaded'])} tensors; "
                f"skipped: {len(report['skipped'])}."
            )
        elif args.from_scratch:
            print("Training baseline from scratch.")

        if args.freeze_frontend:
            model.freeze_frontend()

    elif args.model == "teacher":
        model = create_model(
            args.model,
            text_transform.vocab_size,
            **config["teacher"],
        )

        if args.checkpoint:
            report = load_backbone_weights(model, args.checkpoint)
            print(
                f"Loaded backbone: {len(report['loaded'])} tensors; "
                f"skipped: {len(report['skipped'])}."
            )

        lora_config = config["lora"]

        if not lora_config["enabled"]:
            raise ValueError("LoRA must be enabled when training the teacher.")

        print(
            f"Applying LoRA: start_block={lora_config['start_block']}, "
            f"rank={lora_config['rank']}, alpha={lora_config['alpha']}, "
            f"targets={lora_config['target_modules']}"
        )

        replaced = apply_lora(
            model=model,
            start_block=lora_config["start_block"],
            rank=lora_config["rank"],
            alpha=lora_config["alpha"],
            dropout_rate=lora_config["dropout_rate"],
            target_modules=tuple(lora_config["target_modules"]),
        )
        print(f"Applied LoRA to {len(replaced)} linear modules.")

    else:
        baseline = create_model(
            "baseline",
            text_transform.vocab_size,
            **config["baseline"],
        )

        if args.checkpoint:
            report = load_weights(baseline, args.checkpoint)
            print(f"Loaded baseline: {report['loaded']} tensors.")

        model = VisualRefinerVSRModel(
            baseline,
            freeze_baseline=args.freeze_baseline,
            **config["refiner"],
        )

    if args.model == "teacher":
        print_trainable_parameters(model)
        print(f"Trainable tensors: {len(trainable_parameter_names(model))}")
    else:
        print(f"Model parameters: {parameter_count(model):,}")
        print(f"Trainable parameters: {parameter_count(model, True):,}")

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument(
        "--model", choices=("baseline", "teacher", "refiner"), required=True
    )
    parser.add_argument("--dataset", choices=("vicocktail",), default="vicocktail")
    parser.add_argument("--output_dir", default="/checkpoints")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume_checkpoint")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--freeze_frontend", action="store_true")
    parser.add_argument(
        "--freeze_baseline", action=argparse.BooleanOptionalAction, default=True
    )
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
        metric_for_best_model=(
            "eval_refined_wer" if args.model == "refiner" else "eval_wer"
        ),
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

    allowed_trainable_names = None

    if args.model == "teacher":
        allowed_trainable_names = ("ctc.", ".lora_a.", ".lora_b.")

    trainer = HFTrainer(
        model=model,
        args=get_training_args(args, training_config),
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["val"],
        train_collator=Collator(text_transform, "train"),
        validation_collator=Collator(text_transform, "val"),
        compute_metrics=build_metric_fn(text_transform),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        allowed_trainable_names=allowed_trainable_names,
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
