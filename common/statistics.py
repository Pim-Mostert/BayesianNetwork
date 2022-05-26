import torch


def generate_random_probability_matrix(size):
    p = torch.rand(size, dtype=torch.double)

    return p / p.sum(dim=-1, keepdims=True)