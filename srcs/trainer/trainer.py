import os

import torch
from tqdm.auto import tqdm

from srcs.nets.loss.mctc import mctc_decode

from srcs.trainer.utils import (
    create_lr_scheduler,
    create_metrics,
    get_metric_results,
    move_batch,
    load_history,
    save_ckpt,
    save_history,
)


class Trainer:
    def __init__(self, model, text_transform, configs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.text_transform = text_transform
        self.configs = configs
        self.use_amp = configs.get("amp", True) and self.device.type == "cuda"
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

    def setup_training(self, loader, epoch_count):
        parameters = [param for param in self.model.parameters() if param.requires_grad]

        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.configs["lr"],
            weight_decay=self.configs.get("weight_decay", 0.0),
        )

        accum_steps = self.configs.get("gradient_accumulation_steps", 1)

        self.scheduler = create_lr_scheduler(
            self.optimizer,
            loader,
            epoch_count,
            accum_steps,
            self.configs.get("warmup_steps", 0),
        )

        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

    def decode(self, outputs):
        raise NotImplementedError

    def update_metrics(self, metrics, outputs, batch):
        preds = [
            self.text_transform.decode_for_metrics(ids) for ids in self.decode(outputs)
        ]
        references = [
            self.text_transform.decode_for_metrics(label[: int(length)])
            for label, length in zip(
                batch["labels"].detach().cpu(),
                batch["label_lengths"].detach().cpu(),
            )
        ]

        for name, metric in metrics.items():
            metric.update(
                [prediction[name] for prediction in preds],
                [reference[name] for reference in references],
            )

    def update_model(self, loss, update_params):
        self.scaler.scale(loss).backward()
        if not update_params:
            return

        self.scaler.unscale_(self.optimizer)
        max_gradient_norm = self.configs.get("max_grad_norm", 0.0)

        if max_gradient_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_gradient_norm)

        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.get_scale() >= previous_scale:
            self.scheduler.step()

    def run_epoch(self, loader, training, desc):
        self.model.train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        accum_steps = self.configs.get("gradient_accumulation_steps", 1)
        logging_steps = self.configs.get("logging_steps", 25)
        batch_count = len(loader)
        sample_count = 0
        total_loss = 0.0

        metrics = create_metrics(self.text_transform)

        progress = tqdm(loader, desc=desc)

        with torch.set_grad_enabled(training):
            for batch_number, batch in enumerate(progress, start=1):
                batch = move_batch(batch, self.device)
                with torch.amp.autocast(
                    self.device.type, dtype=torch.float16, enabled=self.use_amp
                ):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]

                if training:
                    group_start = (batch_number - 1) // accum_steps
                    remaining_batches = batch_count - group_start * accum_steps
                    group_size = min(accum_steps, remaining_batches)

                    update_params = (
                        batch_number % accum_steps == 0 or batch_number == batch_count
                    )

                    self.update_model(loss / group_size, update_params)

                batch_size = batch["videos"].size(0)
                total_loss += loss.detach().float().item() * batch_size
                sample_count += batch_size

                self.update_metrics(metrics, outputs, batch)

                if batch_number % logging_steps == 0 or batch_number == batch_count:
                    progress_values = get_metric_results(metrics)
                    progress_values["loss"] = total_loss / sample_count
                    if training:
                        progress_values["lr"] = self.optimizer.param_groups[0]["lr"]

                    progress.set_postfix(progress_values)

        return {"loss": total_loss / sample_count, **get_metric_results(metrics)}

    def resume_from(self, file_path, history):
        state = torch.load(file_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])

        epoch = state["epoch"]
        history[:] = [entry for entry in history if entry["epoch"] <= epoch]
        best_score = (float("inf"), float("inf"))

        for entry in history:
            score = (
                sum(
                    entry[f"validation_{name}"]
                    for name in self.text_transform.metric_names
                )
                / len(self.text_transform.metric_names),
                entry["validation_loss"],
            )
            best_score = min(best_score, score)

        return epoch, best_score

    def train(
        self,
        train_loader,
        val_loader,
        epoch_count,
        output_dir,
        resume=False,
    ):
        self.setup_training(train_loader, epoch_count)
        os.makedirs(output_dir, exist_ok=True)

        best_ckpt_path = os.path.join(output_dir, "best.pt")
        last_ckpt_path = os.path.join(output_dir, "last.pt")
        history_path = os.path.join(output_dir, "history.json")

        best_score = (float("inf"), float("inf"))
        history = []
        start_epoch = 0

        if resume and os.path.isfile(last_ckpt_path):
            history = load_history(history_path)
            start_epoch, best_score = self.resume_from(last_ckpt_path, history)
            print(f"resumed from epoch {start_epoch}, best score {best_score}")

        for epoch in tqdm(
            range(start_epoch + 1, epoch_count + 1),
            initial=start_epoch,
            total=epoch_count,
            desc="Epochs",
        ):
            train_metrics = self.run_epoch(train_loader, True, "Training")

            val_metrics = self.run_epoch(val_loader, False, "Validation")

            epoch_metrics = {
                "epoch": epoch,
                **{f"training_{name}": value for name, value in train_metrics.items()},
                **{f"validation_{name}": value for name, value in val_metrics.items()},
            }

            history.append(epoch_metrics)
            save_history(history, history_path)
            save_ckpt(self.model, last_ckpt_path, epoch, epoch_metrics, self)

            val_score = (
                sum(val_metrics[name] for name in self.text_transform.metric_names)
                / len(self.text_transform.metric_names),
                val_metrics["loss"],
            )

            if val_score < best_score:
                best_score = val_score
                save_ckpt(self.model, best_ckpt_path, epoch, epoch_metrics)

        return history


class EncoderTrainer(Trainer):
    def decode(self, outputs):
        return mctc_decode(outputs["logits"], outputs["input_lengths"])


class DecoderTrainer(Trainer):
    def decode(self, outputs):
        return outputs["preds"]
