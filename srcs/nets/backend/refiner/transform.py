import torch
from torchvision.transforms import RandomApply
from torchvision.transforms.v2 import Transform

VISUAL_KEYS = ("visual_features", "h2_features", "h4_features")


class TimeMask(Transform):
    def __init__(self, window=5, stride=25):
        super().__init__()

        if window <= 0:
            raise ValueError("window must be greater than zero.")
        if stride <= 0:
            raise ValueError("stride must be greater than zero.")

        self.window = int(window)
        self.stride = int(stride)

    def forward(self, mask):
        length = mask.size(0)
        mask_count = int((length + self.stride - 0.1) // self.stride)
        width_limit = min(length, self.window)
        transformed = mask.clone()

        for _ in range(mask_count):
            width = torch.randint(0, width_limit + 1, ()).item()

            if width == 0:
                continue

            start = torch.randint(0, length - width + 1, ()).item()
            transformed[start : start + width] = 0.0

        return transformed


class RefinerTransform(Transform):
    def __init__(self, probability=0.1, window=5, stride=25):
        super().__init__()
        self._validate_probability(probability)
        self.time_mask = RandomApply([TimeMask(window, stride)], p=probability)

    @staticmethod
    def _validate_probability(value):
        if not 0.0 <= value <= 1.0:
            raise ValueError("probability must be between zero and one.")

    def _make_mask(self, reference, input_lengths):
        batch_size, time = reference.shape[:2]
        mask = reference.new_ones(batch_size, time, 1)

        for batch_index, length in enumerate(input_lengths):
            mask[batch_index, :length] = self.time_mask(mask[batch_index, :length])

        return mask

    def forward(self, visual_contexts):
        if not self.training:
            return visual_contexts

        input_lengths = visual_contexts["input_lengths"].tolist()
        visual_mask = self._make_mask(visual_contexts["visual_features"], input_lengths)

        transformed_contexts = visual_contexts.copy()
        for key in VISUAL_KEYS:
            transformed_contexts[key] = visual_contexts[key] * visual_mask

        return transformed_contexts
