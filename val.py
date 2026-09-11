import argparse
import os

import torch
from torchmetrics.text import WordErrorRate

from srcs.datasets.collator import PhonemeCollator
from srcs.datasets.utils import filter_by_length, to_text
from srcs.datasets.vicocktail import load_vicocktail
from srcs.nets.e2e import get_model
from srcs.nets.loss.mctc import mctc_decode
from srcs.nlp.text_transform import PhonemeTransform, vocab_paths
from srcs.nlp.tokenizer import PhonemeTokenizer
from srcs.trainer.utils import (
    create_loader,
    load_configs,
    move_batch,
    set_seed,
)


class WithText:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, items):
        return self.collator(items), [to_text(item["label"]) for item in items]


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIGS_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
MODELS = ("MCTCVSR", "DecoderVSR")
METRIC_NAMES = ("per_i", "per_r", "per_t", "ser", "wer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a checkpoint against the raw transcript."
    )
    parser.add_argument("--configs", default=DEFAULT_CONFIGS_PATH)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_configs(args.configs)
    seed = configs["training"]["seed"]
    set_seed(seed)

    ckpt = os.path.abspath(args.ckpt)
    dir_path = ckpt if os.path.isdir(ckpt) else os.path.dirname(ckpt)
    paths = vocab_paths(os.path.join(dir_path, "vocab"))
    analyser = PhonemeTokenizer()

    transform = PhonemeTransform(**paths)

    def to_hypothesis(ids):
        triples = [
            [
                transform.id2token[name].get(int(index), transform.unk_token)
                for name, index in zip(transform.comp_names, syllable)
            ]
            for syllable in ids
        ]
        return triples, transform.tokenizer.detokenize(triples).split()

    model = get_model(args.model, transform.vocab_size, ckpt=ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataset = load_vicocktail(split="all", fraction=args.fraction, seed=seed)[
        args.split
    ]

    if args.max_frames:
        dataset = filter_by_length(dataset, args.max_frames)

    loader = create_loader(
        dataset,
        WithText(PhonemeCollator(args.split, transform)),
        configs["evaluation"],
    )

    metrics = {name: WordErrorRate() for name in METRIC_NAMES}
    total_loss = 0.0
    sample_count = 0

    def update(name, hypothesis, reference):
        metrics[name].update([" ".join(hypothesis)], [" ".join(reference)])

    with torch.no_grad():
        for batch, batch_texts in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)

            batch_size = batch["videos"].size(0)
            total_loss += outputs["loss"].detach().float().item() * batch_size
            sample_count += batch_size

            samples = outputs.get("preds")

            if samples is None:
                samples = mctc_decode(outputs["logits"], outputs["input_lengths"])

            for sample, text in zip(samples, batch_texts):
                reference = analyser.tokenize(text)
                hypothesis, words = to_hypothesis(sample)

                for index, name in enumerate(METRIC_NAMES[:3]):
                    update(
                        name,
                        [t[index] for t in hypothesis],
                        [t[index] for t in reference],
                    )

                update(
                    "ser",
                    ["|".join(t) for t in hypothesis],
                    ["|".join(t) for t in reference],
                )
                update("wer", words, analyser.to_word(text))

    print(f"{args.model}  {args.split}  {dataset.num_rows:,} clips")
    print(f"  loss   {total_loss / max(1, sample_count):.4f}")

    for name in METRIC_NAMES:
        print(f"  {name:6s} {metrics[name].compute().item():.4f}")


if __name__ == "__main__":
    main()
