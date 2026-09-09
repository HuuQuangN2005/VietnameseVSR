import os

import torch
from tqdm.auto import tqdm

from srcs.trainer.utils import (
    create_learning_rate_scheduler,
    create_metrics,
    get_metric_results,
    move_batch,
    save_checkpoint,
    save_history,
    update_metrics,
)


class Trainer:
    def __init__(self, model, text_transform, configuration):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.text_transform = text_transform
        self.configuration = configuration
        self.use_amp = configuration.get("amp", True) and self.device.type == "cuda"
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

    def setup_training(self, data_loader, epoch_count):
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.configuration["learning_rate"],
            weight_decay=self.configuration.get("weight_decay", 0.0),
        )

        accumulation_steps = self.configuration.get("gradient_accumulation_steps", 1)

        self.scheduler = create_learning_rate_scheduler(
            self.optimizer,
            data_loader,
            epoch_count,
            accumulation_steps,
            self.configuration.get("warmup_steps", 0),
        )

        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

    def update_model(self, loss, update_parameters):
        self.scaler.scale(loss).backward()
        if not update_parameters:
            return

        self.scaler.unscale_(self.optimizer)
        max_gradient_norm = self.configuration.get("max_grad_norm", 0.0)

        if max_gradient_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_gradient_norm)

        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.get_scale() >= previous_scale:
            self.scheduler.step()

    def run_epoch(self, data_loader, training, description):
        self.model.train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        accumulation_steps = self.configuration.get("gradient_accumulation_steps", 1)
        logging_steps = self.configuration.get("logging_steps", 25)
        batch_count = len(data_loader)
        sample_count = 0
        total_loss = 0.0

        metrics = create_metrics(self.text_transform)

        progress = tqdm(data_loader, desc=description)

        with torch.set_grad_enabled(training):
            for batch_number, batch in enumerate(progress, start=1):
                batch = move_batch(batch, self.device)
                with torch.amp.autocast(
                    self.device.type, dtype=torch.float16, enabled=self.use_amp
                ):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]

                if training:
                    group_start = (batch_number - 1) // accumulation_steps
                    remaining_batches = batch_count - group_start * accumulation_steps
                    group_size = min(accumulation_steps, remaining_batches)

                    update_parameters = (
                        batch_number % accumulation_steps == 0
                        or batch_number == batch_count
                    )

                    self.update_model(loss / group_size, update_parameters)

                batch_size = batch["videos"].size(0)
                total_loss += loss.detach().float().item() * batch_size
                sample_count += batch_size

                update_metrics(metrics, outputs, batch, self.text_transform)

                if batch_number % logging_steps == 0 or batch_number == batch_count:
                    progress_values = get_metric_results(metrics)
                    progress_values["loss"] = total_loss / sample_count
                    if training:
                        progress_values["learning_rate"] = self.optimizer.param_groups[
                            0
                        ]["lr"]

                    progress.set_postfix(progress_values)

        return {"loss": total_loss / sample_count, **get_metric_results(metrics)}

    def train(
        self,
        training_data_loader,
        validation_data_loader,
        epoch_count,
        output_directory,
    ):
        self.setup_training(training_data_loader, epoch_count)
        os.makedirs(output_directory, exist_ok=True)

        best_checkpoint_path = os.path.join(output_directory, "best.pt")
        last_checkpoint_path = os.path.join(output_directory, "last.pt")
        history_path = os.path.join(output_directory, "history.json")

        best_score = float("inf")
        stale_epochs = 0
        history = []

        for epoch in tqdm(range(1, epoch_count + 1), desc="Epochs"):
            training_metrics = self.run_epoch(training_data_loader, True, "Training")

            validation_metrics = self.run_epoch(
                validation_data_loader, False, "Validation"
            )

            epoch_metrics = {
                "epoch": epoch,
                **{
                    f"training_{name}": value
                    for name, value in training_metrics.items()
                },
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                },
            }

            history.append(epoch_metrics)
            save_history(history, history_path)
            save_checkpoint(self.model, last_checkpoint_path, epoch, epoch_metrics)

            validation_score = sum(
                validation_metrics[name] for name in self.text_transform.metric_names
            ) / len(self.text_transform.metric_names)
            improved = validation_score < (
                best_score - self.configuration["early_stopping_threshold"]
            )
            if improved:
                best_score = validation_score
                stale_epochs = 0
                save_checkpoint(self.model, best_checkpoint_path, epoch, epoch_metrics)

            else:
                stale_epochs += 1

            if stale_epochs >= self.configuration["early_stopping_patience"]:
                break

        return history
