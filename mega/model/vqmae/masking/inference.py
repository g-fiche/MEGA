import torch
import numpy as np
from einops import repeat


def random_indexes(masks):
    size = len(masks)
    forward_indexes_global = np.arange(size)
    forward_indexes_unmask = forward_indexes_global[
        masks.clone().detach().cpu().numpy().astype(bool)
    ]
    np.random.shuffle(forward_indexes_unmask)
    forward_indexes_mask = forward_indexes_global[
        np.logical_not(masks.clone().detach().cpu().numpy().astype(bool))
    ]
    np.random.shuffle(forward_indexes_mask)
    forward_indexes = np.concatenate((forward_indexes_unmask, forward_indexes_mask))
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


class PatchShuffleInference(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, patches: torch.Tensor, masks: torch.Tensor):
        B = patches.shape[1]

        T = torch.sum(masks[0], dtype=torch.int)

        indexes = [random_indexes(masks[b]) for b in range(B)]
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)

        patches = take_indexes(patches, forward_indexes)

        patches = patches[:T]

        return patches, forward_indexes, backward_indexes
