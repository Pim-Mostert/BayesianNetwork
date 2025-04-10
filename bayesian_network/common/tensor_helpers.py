from typing import List

import torch


def rescale_tensor(tensor: torch.Tensor, gamma=0.999999999):
    return tensor * gamma + (1 - gamma) / 2


def rescale_tensors(tensors: List[torch.Tensor], gamma=0.999999999):
    return [rescale_tensor(t, gamma) for t in tensors]
