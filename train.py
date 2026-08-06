import argparse
import os

import torch
from transformers import TrainingArguments

from srcs.datasets.vicocktail import Collator, DATASETS, load_dataset
from srcs.nets.e2e import get_model as create_model
from srcs.nets.utils import parameter_count
from srcs.spm.spm_train import ensure_unigram, get_paths
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.HF_Trainer import HFTrainer, build_metric_fn

SEED = 42
BATCH_SIZE = 4
NUM_WORKERS = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.005
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 5.0
LOGGING_STEPS = 25
SAVE_TOTAL_LIMIT = 3
FREEZE_FRONTEND = False
PRETRAINED_WEIGHTS_PATH = None
RESUME_CHECKPOINT_PATH = None
MODEL_CONFIG = {
    "attention_dim": 256,
    "attention_heads": 4,
    "linear_units": 1024,
    "num_blocks": 4,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.0,
    "cnn_module_kernel": 31,
}


def get_model(model_name, text_transform):
    model = create_model(
        model_name,
        text_transform.vocab_size,
        pretrained_weights=PRETRAINED_WEIGHTS_PATH,
        freeze_frontend=FREEZE_FRONTEND,
        **MODEL_CONFIG,
    )
    print(f"Model parameters: {parameter_count(model):,}")
    print(f"Trainable parameters: {parameter_count(model, True):,}")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("e2e",), required=True)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--output_dir", default="/checkpoints")
    parser.add_argument("--train_fraction", type=float, default=0.2)
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    return parser.parse_args()


def get_training_args(args):
    return TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        label_names=["labels", "label_lengths"],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )


def get_text_transform(dataset_name, train_dataset):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    model_path = None
    units_path = None
    if rank == 0:
        model_path, units_path = get_paths()
        if not os.path.isfile(model_path):
            tokenizer_dataset = load_dataset(
                dataset_name, train_fraction=1.0, splits=("train",)
            )["train"]
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
    torch.manual_seed(SEED)

    dataset_splits = load_dataset(
        args.dataset,
        train_fraction=args.train_fraction,
        validation_fraction=args.val_fraction,
        test_fraction=1.0,
        splits=("train", "val"),
    )

    text_transform = get_text_transform(args.dataset, dataset_splits["train"])
    model = get_model(args.model, text_transform)

    trainer = HFTrainer(
        model=model,
        args=get_training_args(args),
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["val"],
        data_collator=Collator(text_transform, "train"),
        validation_collator=Collator(text_transform, "val"),
        compute_metrics=build_metric_fn(text_transform),
    )
    trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT_PATH)
    trainer.save_model(os.path.join(args.output_dir, "final"))
    trainer.save_state()


if __name__ == "__main__":
    main()
