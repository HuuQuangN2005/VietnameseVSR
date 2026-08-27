import os

import torch
from torchmetrics.text import WordErrorRate
from tqdm.auto import tqdm

from srcs.nets.utils import freeze
from srcs.trainer.utils import (
    create_cosine_scheduler,
    create_grad_scaler,
    move_batch,
    optimizer_step_count,
    save_checkpoint,
    save_history,
    update_wer,
)


class Trainer:
    def __init__(
        self, model, optimizer, scheduler, scaler, text_transform, config, amp=True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.text_transform = text_transform
        self.config = config
        self.amp = amp
        self.model = model.to(self.device)

    def build_optimizer(self):
        parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("The model does not contain trainable parameters.")

        return torch.optim.AdamW(
            parameters,
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 0.0),
        )

    def build_scheduler(self, train_dataloader, epochs):
        total_steps = optimizer_step_count(
            train_dataloader, epochs, self.config.get("gradient_accumulation_steps", 1)
        )
        return create_cosine_scheduler(
            self.optimizer, total_steps, self.config.get("warmup_steps", 0)
        )

    def setup_training(self, train_dataloader, epochs):
        if self.optimizer is None:
            self.optimizer = self.build_optimizer()

        if self.scheduler is None:
            self.scheduler = self.build_scheduler(train_dataloader, epochs)

        if self.scaler is None:
            self.scaler = create_grad_scaler(self.device, self.amp)

    def _forward(self, batch):
        return self.model(**batch)

    def set_model_mode(self, training):
        self.model.train(training)

    def _run_one_epoch(self, dataloader, training, description, forward):
        amp_enabled = self.amp and self.device.type == "cuda"

        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        max_grad_norm = self.config.get("max_grad_norm", 0.0)
        logging_steps = self.config.get("logging_steps", 25)

        if training and self.optimizer is None:
            raise ValueError("An optimizer is required for training.")

        if training and amp_enabled and self.scaler is None:
            raise ValueError("A GradScaler is required when AMP is enabled.")

        if accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be greater than zero.")

        if logging_steps <= 0:
            raise ValueError("logging_steps must be greater than zero.")

        if len(dataloader) == 0:
            raise ValueError("The dataloader must contain at least one batch.")

        self.set_model_mode(training)

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        sample_count = 0
        loss_sum = 0.0

        wer = WordErrorRate()

        progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        grad_context = torch.enable_grad if training else torch.inference_mode

        with grad_context():
            for step, batch in progress:
                batch = move_batch(batch, self.device)

                with torch.amp.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    outputs = forward(batch)
                    loss = outputs["loss"]

                if loss is None:
                    raise RuntimeError("The model did not return a loss.")

                if training:
                    group_start = (step // accumulation_steps) * accumulation_steps
                    group_size = min(accumulation_steps, len(dataloader) - group_start)

                    scaled_loss = loss / group_size

                    if amp_enabled:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    should_update = (
                        step + 1
                    ) % accumulation_steps == 0 or step + 1 == len(dataloader)

                    if should_update:
                        optimizer_updated = True

                        if amp_enabled:
                            self.scaler.unscale_(self.optimizer)

                        if max_grad_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_grad_norm
                            )

                        if amp_enabled:
                            previous_scale = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            optimizer_updated = self.scaler.get_scale() >= previous_scale
                        else:
                            self.optimizer.step()

                        self.optimizer.zero_grad(set_to_none=True)

                        if self.scheduler is not None and optimizer_updated:
                            self.scheduler.step()

                batch_size = batch["videos"].size(0)
                loss_sum += loss.detach().float().item() * batch_size
                sample_count += batch_size

                update_wer(wer, outputs, batch, self.text_transform)

                if (step + 1) % logging_steps == 0 or step + 1 == len(dataloader):
                    postfix = {
                        "loss": f"{loss_sum / sample_count:.4f}",
                        "wer": f"{wer.compute().item():.4f}",
                    }
                    if training:
                        postfix["lr"] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    progress.set_postfix(postfix)

        return {"loss": loss_sum / sample_count, "wer": wer.compute().item()}

    def run_one_epoch(self, dataloader, training, description):
        return self._run_one_epoch(dataloader, training, description, self._forward)

    def train(self, train_dataloader, validation_dataloader, epochs, output_dir):
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        self.setup_training(train_dataloader, epochs)
        os.makedirs(output_dir, exist_ok=True)

        best_path = os.path.join(output_dir, "best.pt")
        last_path = os.path.join(output_dir, "last.pt")
        history_path = os.path.join(output_dir, "history.json")

        best_wer = float("inf")
        stale_epochs = 0
        history = []

        for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
            train_metrics = self.run_one_epoch(
                train_dataloader, training=True, description="Training"
            )
            validation_metrics = self.run_one_epoch(
                validation_dataloader, training=False, description="Validation"
            )

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_wer": train_metrics["wer"],
                "validation_loss": validation_metrics["loss"],
                "validation_wer": validation_metrics["wer"],
            }

            history.append(epoch_metrics)

            save_history(history, history_path)
            save_checkpoint(self.model, last_path, epoch, epoch_metrics)

            improved = validation_metrics["wer"] < (
                best_wer - self.config["early_stopping_threshold"]
            )
            if improved:
                best_wer = validation_metrics["wer"]
                stale_epochs = 0

                save_checkpoint(self.model, best_path, epoch, epoch_metrics)

            else:
                stale_epochs += 1

            if stale_epochs >= self.config["early_stopping_patience"]:
                break

        return history


class FinetuneTrainer(Trainer):
    def __init__(self, model, *args, **kwargs):
        freeze(model)
        blocks = model.encoder.encoders
        blocks[-2:].requires_grad_(True)
        model.ctc.ctc_lo.requires_grad_(True)

        self.frozen_modules = [
            model.frontend,
            model.proj_encoder,
            model.encoder.embed,
            *blocks[:-2],
            model.encoder.after_norm,
        ]
        super().__init__(model, *args, **kwargs)

    def set_model_mode(self, training):
        self.model.train(training)

        for module in self.frozen_modules:
            module.eval()

    def build_optimizer(self):
        return torch.optim.AdamW(
            [
                {
                    "params": self.model.encoder.encoders[-2:].parameters(),
                    "lr": self.config["encoder_lr"],
                },
                {
                    "params": self.model.ctc.ctc_lo.parameters(),
                    "lr": self.config["ctc_head_lr"],
                },
            ],
            weight_decay=self.config.get("weight_decay", 0.0),
        )


class RefinerTrainer(Trainer):
    def __init__(self, base_model, refiner, *args, transform=None, **kwargs):
        super().__init__(refiner, *args, **kwargs)

        self.base_model = base_model
        freeze(self.base_model)
        self.base_model.to(self.device)
        self.transform = transform

    def _forward_refiner(self, batch):
        logits, visual_contexts = self.base_model.get_contexts(
            batch["videos"], batch["video_lengths"]
        )

        if self.transform is not None and self.model.training:
            visual_contexts = self.transform(visual_contexts)

        return self.model(
            logits,
            visual_contexts,
            labels=batch.get("labels"),
            label_lengths=batch.get("label_lengths"),
        )

    def run_one_epoch(self, dataloader, training, description):
        self.base_model.eval()
        return self._run_one_epoch(
            dataloader, training, description, self._forward_refiner
        )
