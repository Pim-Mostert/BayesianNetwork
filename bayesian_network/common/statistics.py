from typing import Tuple, Iterable, Union

import torch


def generate_random_probability_matrix(size: Union[int, Iterable[int]], device: torch.device):
    p: torch.tensor = torch.rand(size, dtype=torch.double, device=device)

    return p / p.sum(dim=-1, keepdim=True)


def is_valid_probability_matrix(p: torch.Tensor, tolerance=1e-15):
    def _is_approximately_one(x: torch.Tensor):
        return torch.abs(x - 1) < tolerance

    return torch.all(_is_approximately_one(p.sum(dim=-1)))