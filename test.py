import argparse
import unicodedata

import editdistance
import numpy as np
import torch
from tqdm.auto import tqdm

from eval_refiner import CHECKPOINT, get_test_dataloader, load_model
from srcs.datasets.vicocktail import load_vicocktail
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.utils import ctc_decode
from srcs.spm.spm_train import ensure_unigram
from srcs.spm.text_transofm import TextTransform
from train_refiner import CONFIG_PATH, load_config

TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--test_fraction", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_length_matched_permutation(dataset):
    sample_count = len(dataset)
    if sample_count < 2:
        raise ValueError("At least two test samples are required for visual shuffle.")

    lengths = np.asarray(dataset["video_length"], dtype=np.int64)
    sorted_indices = np.argsort(lengths, kind="stable")
    permutation = np.empty(sample_count, dtype=np.int64)
    paired_count = sample_count if sample_count % 2 == 0 else sample_count - 3

    for index in range(0, paired_count, 2):
        first = sorted_indices[index]
        second = sorted_indices[index + 1]
        permutation[first] = second
        permutation[second] = first

    if sample_count % 2 == 1:
        first, second, third = sorted_indices[-3:]
        permutation[first] = second
        permutation[second] = third
        permutation[third] = first

    if np.any(permutation == np.arange(sample_count)):
        raise RuntimeError("Length-matched permutation contains a fixed point.")

    length_differences = np.abs(lengths - lengths[permutation])
    return permutation.tolist(), length_differences


def match_time(features, target_time):
    matched = features.new_zeros(features.size(0), target_time, features.size(2))
    copy_time = min(features.size(1), target_time)
    matched[:, :copy_time] = features[:, :copy_time]
    return matched


def get_refiner_logits(model, contexts, visual_features):
    outputs = model.refiner(
        contexts["logits"],
        visual_features,
        contexts["input_lengths"],
    )
    _, logits = model.final_ctc(
        outputs["logits"],
        contexts["input_lengths"],
    )
    return logits


def collect_predictions(
    model,
    dataloader,
    shuffled_dataloader,
    text_transform,
    device,
):
    model.eval()
    references = []
    baseline_hypotheses = []
    inner_hypotheses = []
    normal_hypotheses = []
    shuffled_hypotheses = []
    zero_hypotheses = []
    inner_loss_sum = 0.0
    final_loss_sum = 0.0
    sample_count = 0

    with torch.inference_mode():
        batches = zip(dataloader, shuffled_dataloader)
        for batch, shuffled_batch in tqdm(
            batches,
            total=len(dataloader),
            desc="Testing",
        ):
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
            }
            shuffled_batch = {
                name: value.to(device, non_blocking=True)
                for name, value in shuffled_batch.items()
            }
            contexts = model.base_model.get_contexts(
                batch["videos"], batch["video_lengths"]
            )

            shuffled_visual = model.base_model.frontend(shuffled_batch["videos"])
            shuffled_visual = shuffled_visual * make_non_pad_mask(
                shuffled_batch["video_lengths"],
                shuffled_visual,
                length_dim=1,
            )
            shuffled_visual = match_time(
                shuffled_visual,
                contexts["logits"].size(1),
            )

            normal_outputs = model.refiner(
                contexts["logits"],
                contexts["visual_features"],
                contexts["input_lengths"],
            )
            inner_loss, inner_logits = model.inner_ctc(
                normal_outputs["inner_logits"],
                contexts["input_lengths"],
                batch["labels"],
                batch["label_lengths"],
            )
            final_loss, normal_logits = model.final_ctc(
                normal_outputs["logits"],
                contexts["input_lengths"],
                batch["labels"],
                batch["label_lengths"],
            )
            shuffled_logits = get_refiner_logits(
                model,
                contexts,
                shuffled_visual,
            )
            zero_logits = get_refiner_logits(
                model,
                contexts,
                torch.zeros_like(contexts["visual_features"]),
            )

            baseline_tokens = ctc_decode(
                contexts["logits"],
                contexts["input_lengths"],
                text_transform.blank_id,
            )
            normal_tokens = ctc_decode(
                normal_logits,
                contexts["input_lengths"],
                text_transform.blank_id,
            )
            inner_tokens = ctc_decode(
                inner_logits,
                contexts["input_lengths"],
                text_transform.blank_id,
            )
            shuffled_tokens = ctc_decode(
                shuffled_logits,
                contexts["input_lengths"],
                text_transform.blank_id,
            )
            zero_tokens = ctc_decode(
                zero_logits,
                contexts["input_lengths"],
                text_transform.blank_id,
            )

            baseline_hypotheses.extend(
                text_transform.decode(item) for item in baseline_tokens
            )
            normal_hypotheses.extend(
                text_transform.decode(item) for item in normal_tokens
            )
            inner_hypotheses.extend(
                text_transform.decode(item) for item in inner_tokens
            )
            shuffled_hypotheses.extend(
                text_transform.decode(item) for item in shuffled_tokens
            )
            zero_hypotheses.extend(
                text_transform.decode(item) for item in zero_tokens
            )
            references.extend(
                text_transform.decode(label[: int(length)])
                for label, length in zip(
                    batch["labels"].detach().cpu(),
                    batch["label_lengths"].detach().cpu(),
                )
            )
            batch_size = batch["videos"].size(0)
            inner_loss_sum += inner_loss.item() * batch_size
            final_loss_sum += final_loss.item() * batch_size
            sample_count += batch_size

    return {
        "references": references,
        "baseline": baseline_hypotheses,
        "inner": inner_hypotheses,
        "normal": normal_hypotheses,
        "shuffled": shuffled_hypotheses,
        "zero": zero_hypotheses,
        "inner_loss": inner_loss_sum / sample_count,
        "final_loss": final_loss_sum / sample_count,
    }


def word_errors(references, hypotheses):
    errors = np.asarray(
        [
            editdistance.eval(reference.split(), hypothesis.split())
            for reference, hypothesis in zip(references, hypotheses)
        ],
        dtype=np.int64,
    )
    reference_words = np.asarray(
        [len(reference.split()) for reference in references],
        dtype=np.int64,
    )
    return errors, reference_words


def remove_tones(text):
    decomposed = unicodedata.normalize("NFD", text)
    without_tones = "".join(
        character for character in decomposed if character not in TONE_MARKS
    )
    return unicodedata.normalize("NFC", without_tones)


def tone_stripped_errors(references, hypotheses):
    stripped_references = [remove_tones(item) for item in references]
    stripped_hypotheses = [remove_tones(item) for item in hypotheses]
    return word_errors(stripped_references, stripped_hypotheses)


def align_reference_words(reference, hypothesis):
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    reference_count = len(reference_words)
    hypothesis_count = len(hypothesis_words)

    distances = np.zeros(
        (reference_count + 1, hypothesis_count + 1),
        dtype=np.int32,
    )
    distances[:, 0] = np.arange(reference_count + 1)
    distances[0, :] = np.arange(hypothesis_count + 1)

    for ref_index in range(1, reference_count + 1):
        for hyp_index in range(1, hypothesis_count + 1):
            substitution_cost = (
                reference_words[ref_index - 1] != hypothesis_words[hyp_index - 1]
            )
            distances[ref_index, hyp_index] = min(
                distances[ref_index - 1, hyp_index] + 1,
                distances[ref_index, hyp_index - 1] + 1,
                distances[ref_index - 1, hyp_index - 1] + substitution_cost,
            )

    correct = np.zeros(reference_count, dtype=np.bool_)
    insertions = 0
    ref_index = reference_count
    hyp_index = hypothesis_count

    while ref_index > 0 or hyp_index > 0:
        if (
            ref_index > 0
            and hyp_index > 0
            and reference_words[ref_index - 1] == hypothesis_words[hyp_index - 1]
            and distances[ref_index, hyp_index]
            == distances[ref_index - 1, hyp_index - 1]
        ):
            correct[ref_index - 1] = True
            ref_index -= 1
            hyp_index -= 1
        elif (
            ref_index > 0
            and hyp_index > 0
            and distances[ref_index, hyp_index]
            == distances[ref_index - 1, hyp_index - 1] + 1
        ):
            ref_index -= 1
            hyp_index -= 1
        elif (
            ref_index > 0
            and distances[ref_index, hyp_index]
            == distances[ref_index - 1, hyp_index] + 1
        ):
            ref_index -= 1
        else:
            hyp_index -= 1
            insertions += 1

    return correct, insertions


def word_transitions(references, baseline_hypotheses, refiner_hypotheses):
    wrong_to_correct = 0
    correct_to_wrong = 0
    wrong_to_wrong = 0
    correct_to_correct = 0
    baseline_insertions = 0
    refiner_insertions = 0

    for reference, baseline, refiner in zip(
        references,
        baseline_hypotheses,
        refiner_hypotheses,
    ):
        baseline_correct, baseline_insertion_count = align_reference_words(
            reference,
            baseline,
        )
        refiner_correct, refiner_insertion_count = align_reference_words(
            reference,
            refiner,
        )

        wrong_to_correct += int(np.sum(~baseline_correct & refiner_correct))
        correct_to_wrong += int(np.sum(baseline_correct & ~refiner_correct))
        wrong_to_wrong += int(np.sum(~baseline_correct & ~refiner_correct))
        correct_to_correct += int(np.sum(baseline_correct & refiner_correct))
        baseline_insertions += baseline_insertion_count
        refiner_insertions += refiner_insertion_count

    baseline_wrong = wrong_to_correct + wrong_to_wrong
    baseline_correct = correct_to_wrong + correct_to_correct

    return {
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_wrong": wrong_to_wrong,
        "correct_to_correct": correct_to_correct,
        "correction_rate": wrong_to_correct / baseline_wrong,
        "damage_rate": correct_to_wrong / baseline_correct,
        "baseline_insertions": baseline_insertions,
        "refiner_insertions": refiner_insertions,
    }


def paired_bootstrap(
    baseline_errors,
    refiner_errors,
    reference_words,
    bootstrap_samples,
    seed,
):
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be greater than zero.")

    rng = np.random.default_rng(seed)
    sample_count = len(reference_words)
    differences = np.empty(bootstrap_samples, dtype=np.float64)

    for index in range(bootstrap_samples):
        sample_indices = rng.integers(0, sample_count, size=sample_count)
        word_count = reference_words[sample_indices].sum()
        baseline_wer = baseline_errors[sample_indices].sum() / word_count
        refiner_wer = refiner_errors[sample_indices].sum() / word_count
        differences[index] = refiner_wer - baseline_wer

    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "improvement_probability": float(np.mean(differences < 0.0)),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    evaluation_config = config["evaluation"]

    test_dataset = load_vicocktail(
        test_fraction=args.test_fraction,
        splits=("test",),
        seed=args.seed,
    )["test"]
    model_path, units_path = ensure_unigram()
    text_transform = TextTransform(model_path, units_path)

    permutation, length_differences = create_length_matched_permutation(test_dataset)
    shuffled_dataset = test_dataset.select(permutation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(text_transform.vocab_size, args.checkpoint).to(device)
    dataloader = get_test_dataloader(
        test_dataset,
        text_transform,
        evaluation_config,
    )
    shuffled_dataloader = get_test_dataloader(
        shuffled_dataset,
        text_transform,
        evaluation_config,
    )
    predictions = collect_predictions(
        model,
        dataloader,
        shuffled_dataloader,
        text_transform,
        device,
    )

    baseline_errors, reference_words = word_errors(
        predictions["references"],
        predictions["baseline"],
    )
    normal_errors, _ = word_errors(
        predictions["references"],
        predictions["normal"],
    )
    shuffled_errors, _ = word_errors(
        predictions["references"],
        predictions["shuffled"],
    )
    zero_errors, _ = word_errors(
        predictions["references"],
        predictions["zero"],
    )
    baseline_tone_stripped_errors, _ = tone_stripped_errors(
        predictions["references"],
        predictions["baseline"],
    )
    normal_tone_stripped_errors, _ = tone_stripped_errors(
        predictions["references"],
        predictions["normal"],
    )
    shuffled_tone_stripped_errors, _ = tone_stripped_errors(
        predictions["references"],
        predictions["shuffled"],
    )
    zero_tone_stripped_errors, _ = tone_stripped_errors(
        predictions["references"],
        predictions["zero"],
    )

    word_count = reference_words.sum()
    baseline_wer = baseline_errors.sum() / word_count
    normal_wer = normal_errors.sum() / word_count
    shuffled_wer = shuffled_errors.sum() / word_count
    zero_wer = zero_errors.sum() / word_count
    baseline_tone_stripped_wer = baseline_tone_stripped_errors.sum() / word_count
    normal_tone_stripped_wer = normal_tone_stripped_errors.sum() / word_count
    shuffled_tone_stripped_wer = shuffled_tone_stripped_errors.sum() / word_count
    zero_tone_stripped_wer = zero_tone_stripped_errors.sum() / word_count
    baseline_tone_error_fraction = (
        baseline_errors.sum() - baseline_tone_stripped_errors.sum()
    ) / baseline_errors.sum()
    normal_tone_error_fraction = (
        normal_errors.sum() - normal_tone_stripped_errors.sum()
    ) / normal_errors.sum()
    difference = normal_wer - baseline_wer
    relative_change = difference / baseline_wer

    refiner_wins = int(np.sum(normal_errors < baseline_errors))
    baseline_wins = int(np.sum(baseline_errors < normal_errors))
    ties = int(np.sum(normal_errors == baseline_errors))
    transitions = word_transitions(
        predictions["references"],
        predictions["baseline"],
        predictions["normal"],
    )

    baseline_bootstrap = paired_bootstrap(
        baseline_errors,
        normal_errors,
        reference_words,
        args.bootstrap_samples,
        args.seed,
    )
    zero_bootstrap = paired_bootstrap(
        zero_errors,
        normal_errors,
        reference_words,
        args.bootstrap_samples,
        args.seed,
    )
    shuffled_bootstrap = paired_bootstrap(
        shuffled_errors,
        normal_errors,
        reference_words,
        args.bootstrap_samples,
        args.seed,
    )

    visual_gain = zero_wer - normal_wer
    total_gain = baseline_wer - normal_wer
    visual_gain_ratio = (
        visual_gain / total_gain if total_gain != 0.0 else float("nan")
    )

    print(f"Test samples: {len(predictions['references'])}")
    print(f"Baseline WER: {baseline_wer:.6f}")
    print(f"Refiner correct visual WER: {normal_wer:.6f}")
    print(f"Refiner shuffled visual WER: {shuffled_wer:.6f}")
    print(f"Refiner zero visual WER: {zero_wer:.6f}")
    print(f"Baseline tone-stripped WER: {baseline_tone_stripped_wer:.6f}")
    print(
        "Refiner correct visual tone-stripped WER: "
        f"{normal_tone_stripped_wer:.6f}"
    )
    print(
        "Refiner shuffled visual tone-stripped WER: "
        f"{shuffled_tone_stripped_wer:.6f}"
    )
    print(
        "Refiner zero visual tone-stripped WER: "
        f"{zero_tone_stripped_wer:.6f}"
    )
    print(
        "Baseline tone-stripping WER reduction: "
        f"{(baseline_wer - baseline_tone_stripped_wer) * 100:.3f} points"
    )
    print(
        "Correct visual tone-stripping WER reduction: "
        f"{(normal_wer - normal_tone_stripped_wer) * 100:.3f} points"
    )
    print(
        "Baseline errors removed by tone stripping: "
        f"{baseline_tone_error_fraction * 100:.3f}%"
    )
    print(
        "Correct visual errors removed by tone stripping: "
        f"{normal_tone_error_fraction * 100:.3f}%"
    )
    print(f"Correct - zero WER: {normal_wer - zero_wer:.6f}")
    print(f"Correct - shuffled WER: {normal_wer - shuffled_wer:.6f}")
    print(f"Descriptive visual gain ratio: {visual_gain_ratio * 100:.3f}%")
    print(
        "Shuffle length difference: "
        f"mean={length_differences.mean():.3f}, "
        f"max={length_differences.max()} frames"
    )
    print(
        "Correct - baseline WER: "
        f"{difference:.6f} ({difference * 100:.3f} points)"
    )
    print(f"Relative WER change: {relative_change * 100:.3f}%")
    print(f"Refiner wins: {refiner_wins}")
    print(f"Baseline wins: {baseline_wins}")
    print(f"Ties: {ties}")
    print(f"Wrong -> Correct: {transitions['wrong_to_correct']}")
    print(f"Correct -> Wrong: {transitions['correct_to_wrong']}")
    print(f"Wrong -> Wrong: {transitions['wrong_to_wrong']}")
    print(f"Correct -> Correct: {transitions['correct_to_correct']}")
    print(f"Correction rate: {transitions['correction_rate'] * 100:.3f}%")
    print(f"Damage rate: {transitions['damage_rate'] * 100:.3f}%")
    print(f"Baseline insertions: {transitions['baseline_insertions']}")
    print(f"Refiner insertions: {transitions['refiner_insertions']}")
    print(
        "Correct - baseline bootstrap 95% CI: "
        f"[{baseline_bootstrap['lower']:.6f}, "
        f"{baseline_bootstrap['upper']:.6f}]"
    )
    print(
        "Correct better than baseline probability: "
        f"{baseline_bootstrap['improvement_probability']:.4f}"
    )
    print(
        "Correct - zero bootstrap 95% CI: "
        f"[{zero_bootstrap['lower']:.6f}, {zero_bootstrap['upper']:.6f}]"
    )
    print(
        "Correct better than zero probability: "
        f"{zero_bootstrap['improvement_probability']:.4f}"
    )
    print(
        "Correct - shuffled bootstrap 95% CI: "
        f"[{shuffled_bootstrap['lower']:.6f}, "
        f"{shuffled_bootstrap['upper']:.6f}]"
    )
    print(
        "Correct better than shuffled probability: "
        f"{shuffled_bootstrap['improvement_probability']:.4f}"
    )

    print(
        "Correct better than baseline: "
        f"{'yes' if baseline_bootstrap['upper'] < 0.0 else 'no'}"
    )
    print(
        "Correct better than zero: "
        f"{'yes' if zero_bootstrap['upper'] < 0.0 else 'no'}"
    )
    print(
        "Correct better than shuffled: "
        f"{'yes' if shuffled_bootstrap['upper'] < 0.0 else 'no'}"
    )


if __name__ == "__main__":
    main()
