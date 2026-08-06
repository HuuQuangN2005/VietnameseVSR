import argparse

import torch
from transformers import TrainingArguments

from srcs.datasets.vicocktail import Collator, DATASETS, load_dataset
from srcs.nets.e2e import get_model
from srcs.spm.text_transofm import TextTransform
from srcs.trainer.HF_Trainer import HFTrainer, build_metric_fn


BATCH_SIZE = 2
NUM_WORKERS = 0
MODEL_CONFIG = {
    "attention_dim": 256,
    "attention_heads": 4,
    "linear_units": 1024,
    "num_blocks": 4,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.0,
    "cnn_module_kernel": 31,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("e2e",), required=True)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="/checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_splits = load_dataset(
        args.dataset,
        train_fraction=1.0,
        validation_fraction=1.0,
        test_fraction=1.0,
        splits=("test",),
    )
    if "test" not in dataset_splits:
        raise ValueError("The selected dataset has no test split.")
    text_transform = TextTransform()
    model = get_model(
        args.model,
        text_transform.vocab_size,
        checkpoint_path=args.checkpoint,
        **MODEL_CONFIG,
    )
    evaluation_args = TrainingArguments(
        output_dir=args.output_dir,
        label_names=["labels", "label_lengths"],
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=NUM_WORKERS,
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
    )
    metrics = trainer.evaluate(metric_key_prefix="test")
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
