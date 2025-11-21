# %%

from common.torch_settings import TorchSettings
from dynamic_bayesian_network.builder import DynamicBayesianNetworkBuilder
from dynamic_bayesian_network.dynamic_bayesian_network import Node
from dynamic_bayesian_network.unroller import unroll

torch_settings = TorchSettings(device="cpu", dtype="float64")

builder = DynamicBayesianNetworkBuilder()

Q = Node.random(
    cpt_size=(2, 2),
    is_sequential=True,
    torch_settings=torch_settings,
    prior_size=(2),
    name="Q",
)
Y = Node.random(
    cpt_size=(2, 3),
    is_sequential=False,
    torch_settings=torch_settings,
    name="Y",
)

builder.add_node(Q, sequential_parents=Q)
builder.add_node(Y, parents=Q)

network = builder.build()
network.nodes

# %%

bn, map = unroll(network, sequence_length=3)


# %%

[Q_t0, Y_t0, Q_t1, Y_t1, Q_t2, Y_t2] = list(bn.nodes)

TODO: WRITE UNIT TESTS