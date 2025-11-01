import torch
import numpy.typing as npt
import numpy as np

import einx

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    length = len(dataset)
    data = torch.from_numpy(dataset)
    windows = data.unfold(0, context_length, 1)  # (N - context_length + 1, context_length)
    Xwindows = windows[:-1]  # (N - context_length, context_length)
    Ywindows = windows[1:]  # (N - context_length, context_length)

    indices = torch.randint(0, Xwindows.shape[0], (batch_size,1))
    X = einx.get_at("[l] c, b [1] -> b c", Xwindows, indices).to(device)
    Y = einx.get_at("[l] c, b [1] -> b c", Ywindows, indices).to(device)
    return X, Y


