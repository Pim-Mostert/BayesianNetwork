# %%

from common.torch_settings import TorchSettings
from dynamic_bayesian_network.builder import DynamicBayesianNetworkBuilder
from dynamic_bayesian_network.dynamic_bayesian_network import Node

torch_settings = TorchSettings(device="cpu", dtype="float64")

builder = DynamicBayesianNetworkBuilder()

Q = Node.random(
    cpt_size=(2, 2),
    is_sequential=True,
    torch_settings=torch_settings,
    prior_size=(2),
)
Y = Node.random(
    cpt_size=(3, 2, 3),
    is_sequential=True,
    torch_settings=torch_settings,
    prior_size=(2, 3),
)

builder.add_node(Q, sequential_parents=Q)
builder.add_node(Y, parents=Q, sequential_parents=Y)

network = builder.build()
