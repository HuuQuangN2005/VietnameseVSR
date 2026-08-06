import os

import torch
from safetensors.torch import load_file


def _state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _strip(key):
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def _weight_path(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ("model.safetensors", "pytorch_model.bin", "last.pt", "best.pt"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(f"Weights not found: {path}")


def load_weights(model, path, part=None):
    if not path:
        raise FileNotFoundError(f"Weights not found: {path}")
    path = _weight_path(path)
    if path.endswith(".safetensors"):
        checkpoint = load_file(path, device="cpu")
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_state = {
        _strip(key): value for key, value in _state(checkpoint).items()
    }
    module = getattr(model, part) if part else model
    target_state = module.state_dict()
    loaded = {}
    for key, value in source_state.items():
        names = [key]
        if part and key.startswith(part + "."):
            names.insert(0, key[len(part) + 1 :])
        for name in names:
            if name in target_state and target_state[name].shape == value.shape:
                loaded[name] = value
                break
    if not loaded:
        raise RuntimeError(f"No compatible weights found for {part or 'model'}.")
    info = module.load_state_dict(loaded, strict=False)
    return {
        "loaded": len(loaded),
        "missing": list(info.missing_keys),
        "unexpected": list(info.unexpected_keys),
    }


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def parameter_count(model, trainable=False):
    params = model.parameters()
    if trainable:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)
