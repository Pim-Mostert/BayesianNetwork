import os

from common.torch_settings import TorchSettings
from pytest import fixture


@fixture(autouse=True)
def setup_environment():
    os.environ["BN__TORCH_SETTINGS__DEVICE"] = "cpu"
    os.environ["BN__TORCH_SETTINGS__DTYPE"] = "float64"


@fixture
def torch_settings():
    return TorchSettings()
