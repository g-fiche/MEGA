import torch
import numpy as np
from einops import repeat
import math


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


class PatchShuffleScheduled(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.noise_schedule = cosine_schedule

    def forward(self, patches: torch.Tensor, fixed_ratio: float = None):
        T, B, C = patches.shape
        if fixed_ratio is None:
            rand_time = torch.rand(1)
            ratio = self.noise_schedule(rand_time)
        else:
            ratio = fixed_ratio
        remain_T = int(T * (1 - ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes
