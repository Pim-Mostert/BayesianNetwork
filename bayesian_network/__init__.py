from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bayesian_network")
except PackageNotFoundError:
    __version__ = "local"
