import torch

from common.extensions import extension

from .torch_settings import TorchSettings


def generate_random_probability_matrix(
    size,
    torch_settings: TorchSettings,
):
    p: torch.Tensor = torch.rand(size, dtype=torch_settings.dtype, device=torch_settings.device)

    return p / p.sum(dim=-1, keepdim=True)


def is_valid_probability_matrix(p: torch.Tensor):
    total = p.sum(dim=-1)
    return torch.isclose(total, torch.ones_like(total)).all().item()


@extension(to=torch.Tensor)
def is_probability_matrix(p: torch.Tensor):
    return is_valid_probability_matrix(p)
