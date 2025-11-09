import torch
from typing_extensions import deprecated

from common.extensions import extension

from .torch_settings import TorchSettings


def generate_random_probability_matrix(
    size,
    torch_settings: TorchSettings,
):
    p: torch.Tensor = torch.rand(size, dtype=torch_settings.dtype, device=torch_settings.device)

    return p / p.sum(dim=-1, keepdim=True)


@deprecated("Use is_probability_matrix extension method")
def is_valid_probability_matrix(p: torch.Tensor):
    return torch.isclose(p.sum(dim=-1), torch.tensor(1.0)).all().item()


@extension(to=torch.Tensor)
def is_probability_matrix(p: torch.Tensor):
    return is_valid_probability_matrix(p)
