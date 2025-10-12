import torch

from bayesian_network.common.torch_settings import TorchSettings


def generate_random_probability_matrix(
    size,
    torch_settings: TorchSettings,
):
    p: torch.Tensor = torch.rand(size, dtype=torch_settings.dtype, device=torch_settings.device)

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
