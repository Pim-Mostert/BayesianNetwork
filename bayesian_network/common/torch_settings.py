import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings


class TorchSettings(BaseSettings):
    device: torch.device
    dtype: torch.dtype

    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, v):
        valid_devices = {"cpu", "cuda", "mps"}
        if isinstance(v, str) and v in valid_devices:
            return torch.device(v)
        raise ValueError(f"Invalid device '{v}'. Must be one of {valid_devices}.")

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v):
        valid_dtypes = {
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if isinstance(v, str) and v in valid_dtypes:
            return valid_dtypes[v]
        raise ValueError(f"Invalid dtype '{v}'. Must be one of {list(valid_dtypes.keys())}.")

    def __init__(self, *args, **kwargs):
        if args:
            raise TypeError(f"Positional arguments are not allowed. Got: {args}")
        super().__init__(**kwargs)

    class Config:
        env_prefix = "BN__TORCH_SETTINGS__"
