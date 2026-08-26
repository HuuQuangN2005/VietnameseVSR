import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.text import WordErrorRate
from tqdm.auto import tqdm

from srcs.datasets.vicocktail import Collator, load_vicocktail
from srcs.nets.e2e import VSRRefinerModel, get_model as create_model
from srcs.nets.utils import ctc_decode, load_weights
from srcs.spm.spm_train import ensure_unigram
from srcs.spm.text_transofm import TextTransform
from train_refiner import CONFIG_PATH, load_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "checkpoint-54908", "checkpoint-54908"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--test_fraction", type=float, default=1.0)
    return parser.parse_args()


def load_model(vocab_size, checkpoint_path):
    base_model = create_model("auto-vsr", vocab_size, size="large")
    model = VSRRefinerModel(base_model=base_model, vocab_size=vocab_size)
    load_weights(model, checkpoint_path)
    return model


def get_test_dataloader(dataset, text_transform, config):
    num_workers = config["num_workers"]

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=Collator(text_transform, "test"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def evaluate(model, dataloader, text_transform, device):
    model.eval()
    inner_loss_sum = 0.0
    final_loss_sum = 0.0
    sample_count = 0
    inner_wer = WordErrorRate()
    final_wer = WordErrorRate()

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {
                name: value.to(device, non_blocking=True) for name, value in batch.items()
            }
            contexts = model.base_model.get_contexts(
                batch["videos"], batch["video_lengths"]
            )
            outputs = model.refiner(
                contexts["logits"], contexts["visual_features"], contexts["input_lengths"]
            )
            inner_loss, inner_logits = model.inner_ctc(
                outputs["inner_logits"],
                contexts["input_lengths"],
                batch["labels"],
                batch["label_lengths"],
            )
            final_loss, final_logits = model.final_ctc(
                outputs["logits"],
                contexts["input_lengths"],
                batch["labels"],
                batch["label_lengths"],
            )

            batch_size = batch["videos"].size(0)
            inner_loss_sum += inner_loss.item() * batch_size
            final_loss_sum += final_loss.item() * batch_size
            sample_count += batch_size

            inner_token_ids = ctc_decode(
                inner_logits, contexts["input_lengths"], text_transform.blank_id
            )
            final_token_ids = ctc_decode(
                final_logits, contexts["input_lengths"], text_transform.blank_id
            )
            inner_hypotheses = [text_transform.decode(item) for item in inner_token_ids]
            final_hypotheses = [text_transform.decode(item) for item in final_token_ids]
            references = [
                text_transform.decode(label[: int(length)])
                for label, length in zip(
                    batch["labels"].detach().cpu(), batch["label_lengths"].detach().cpu()
                )
            ]
            inner_wer.update(inner_hypotheses, references)
            final_wer.update(final_hypotheses, references)

    return {
        "inner_loss": inner_loss_sum / sample_count,
        "final_loss": final_loss_sum / sample_count,
        "inner_wer": inner_wer.compute().item(),
        "final_wer": final_wer.compute().item(),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    evaluation_config = config["evaluation"]
    seed = config["training"]["seed"]

    test_dataset = load_vicocktail(
        test_fraction=args.test_fraction, splits=("test",), seed=seed
    )["test"]
    model_path, units_path = ensure_unigram()
    text_transform = TextTransform(model_path, units_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(text_transform.vocab_size, args.checkpoint).to(device)
    dataloader = get_test_dataloader(test_dataset, text_transform, evaluation_config)
    metrics = evaluate(model, dataloader, text_transform, device)

    print(f"Test samples: {len(test_dataset)}")
    print(f"Test inner loss: {metrics['inner_loss']:.6f}")
    print(f"Test final loss: {metrics['final_loss']:.6f}")
    print(f"Test inner WER: {metrics['inner_wer']:.6f}")
    print(f"Test final WER: {metrics['final_wer']:.6f}")


if __name__ == "__main__":
    main()
