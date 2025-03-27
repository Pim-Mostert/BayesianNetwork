from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bayesian_network")
except PackageNotFoundError:
    __version__ = "local"
