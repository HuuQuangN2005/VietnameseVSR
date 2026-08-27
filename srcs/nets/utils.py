import os
import torch

from safetensors.torch import load_file


def _weight_path(path):
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        for name in ("model.safetensors", "pytorch_model.bin", "last.pt", "best.pt"):
            candidate = os.path.join(path, name)

            if os.path.isfile(candidate):
                return candidate

    raise FileNotFoundError(f"Weights not found: {path}")


def _load_state(path):
    path = _weight_path(path)

    if path.endswith(".safetensors"):
        state = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)

    for key in ("model_state_dict", "state_dict", "model"):
        if key in state:
            state = state[key]
            break

    return {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in state.items()
    }


def load_weights(model, path):
    state = _load_state(path)
    target = model.state_dict()
    state = {
        key: value
        for key, value in state.items()
        if key in target and value.shape == target[key].shape
    }
    missing = [key for key in target if key not in state]

    if not state or any(not key.startswith("ctc.") for key in missing):
        raise RuntimeError(f"No compatible weights found: {path}")

    model.load_state_dict(state, strict=False)


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

    module.eval()
