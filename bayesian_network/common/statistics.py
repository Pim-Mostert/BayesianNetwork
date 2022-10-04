from typing import Tuple, Iterable, Union

import torch


def generate_random_probability_matrix(size: Union[int, Iterable[int]], device: torch.device, dtype: torch.dtype):
    p: torch.tensor = torch.rand(size, dtype=dtype, device=device)

    return p / p.sum(dim=-1, keepdim=True)


def is_valid_probability_matrix(p: torch.Tensor, tolerance=None):
    if tolerance is None:
        if p.dtype == torch.float32:
            tolerance = 1e-6
        else:
            tolerance = 1e-15

    def _is_approximately_one(x: torch.Tensor):
        return torch.abs(x - 1) < tolerance

    return torch.all(_is_approximately_one(p.sum(dim=-1)))