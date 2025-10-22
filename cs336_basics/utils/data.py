import torch
import numpy.typing as npt
import numpy as np

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    length = len(dataset)
    starting_indices = np.random.randint(0, length - context_length, size=batch_size)

    X = torch.stack(
        [torch.tensor(dataset[i : i + context_length], device=device) for i in starting_indices]
    )
    Y = torch.stack(
        [torch.tensor(dataset[i + 1 : i + context_length + 1], device=device) for i in starting_indices]
    )
    return X, Y


