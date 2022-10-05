from typing import List
import torch

def rescale_tensor(tensor: torch.Tensor, gamma=0.999999999):
        return tensor*gamma + (1-gamma)/2

def rescale_tensors(tensors: List[torch.Tensor], gamma=0.999999999):
    return [rescale_tensor(t, gamma) for t in tensors]

def get_min_pos_value(dtype: torch.dtype):
    if dtype == torch.float32: 
        return torch.tensor(1.4013e-45, dtype=torch.float32)
    else:
        return torch.tensor(4.9407e-324, dtype=torch.float64)