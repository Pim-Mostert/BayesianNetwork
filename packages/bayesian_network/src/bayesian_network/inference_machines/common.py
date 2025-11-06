from dataclasses import dataclass

from common.torch_settings import TorchSettings


@dataclass
class InferenceMachineSettings:
    torch_settings: TorchSettings
    average_log_likelihood: bool
