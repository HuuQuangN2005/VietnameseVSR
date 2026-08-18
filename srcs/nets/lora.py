import math

import torch.nn as nn

from srcs.nets.utils import freeze

LORA_TRAINABLE_PATTERNS = ("ctc.", ".lora_a.", ".lora_b.")


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16, dropout_rate=0.05):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be positive.")

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")

        self.linear = linear
        self.lora_a = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = alpha / rank
        self.is_lora = True

        freeze(self.linear)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs):
        output = self.linear(inputs)
        update = self.lora_b(self.lora_a(self.dropout(inputs)))
        return output + self.scale * update


def apply_lora(
    model,
    start_block=8,
    rank=8,
    alpha=16,
    dropout_rate=0.05,
    target_modules=("linear_q", "linear_v"),
):
    blocks = model.encoder.encoders

    if not 0 <= start_block < len(blocks):
        raise ValueError(f"start_block must be between 0 and {len(blocks) - 1}.")

    freeze(model)
    replaced = []

    for block_index in range(start_block, len(blocks)):
        attention = blocks[block_index].self_attn

        for name in target_modules:
            linear = getattr(attention, name, None)

            if not isinstance(linear, nn.Linear):
                raise ValueError(
                    f"encoder block {block_index} has no linear module named {name}."
                )

            setattr(
                attention,
                name,
                LoRALinear(
                    linear=linear, rank=rank, alpha=alpha, dropout_rate=dropout_rate
                ),
            )
            replaced.append(f"encoder.encoders.{block_index}.self_attn.{name}")

    model.ctc.requires_grad_(True)
    model.lora_finetuning = True

    return replaced


def apply_lora_config(model, config):
    return apply_lora(
        model=model,
        start_block=config["start_block"],
        rank=config["rank"],
        alpha=config["alpha"],
        dropout_rate=config["dropout_rate"],
        target_modules=tuple(config["target_modules"]),
    )


def print_trainable_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    ratio = 100.0 * trainable / total

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,} ({ratio:.4f}%)")

    return {"total": total, "trainable": trainable, "ratio": ratio}
