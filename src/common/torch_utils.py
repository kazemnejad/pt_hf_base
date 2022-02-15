from typing import List

import torch


def pad_batches_to_same_length(
    batches: List[torch.Tensor], padding_value: int = 0
) -> torch.Tensor:
    num_batches = len(batches)
    batch_size = batches[0].shape[0]
    max_length = max(b.shape[1] for b in batches)
    out_dims = (batch_size * num_batches, max_length, *batches[0].shape[2:])
    out_tensor = batches[0].new(*out_dims).fill_(padding_value)
    for i, b in enumerate(batches):
        length = b.shape[1]
        out_tensor[i * batch_size : (i + 1) * batch_size, :length] = b
    return out_tensor
