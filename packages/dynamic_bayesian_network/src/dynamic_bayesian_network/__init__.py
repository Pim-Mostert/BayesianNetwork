from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dynamic_bayesian_network")
except PackageNotFoundError:
    __version__ = "noinstall"
