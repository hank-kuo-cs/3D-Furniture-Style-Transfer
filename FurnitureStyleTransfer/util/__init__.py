import torch
import numpy as np
from ..config import config


def tensor_to_numpy(tensor_array: torch.Tensor) -> np.ndarray:
    assert isinstance(tensor_array, torch.Tensor)

    tensor_array = tensor_array.clone()
    if tensor_array.requires_grad:
        tensor_array = tensor_array.detach()
    if config.cuda.device != 'cpu':
        tensor_array = tensor_array.cpu()

    numpy_array = tensor_array.numpy()
    return numpy_array
