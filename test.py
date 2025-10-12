# %%

import matplotlib.pyplot as plt
import networkx as nx
import torch

from bayesian_network.bayesian_network import BayesianNetworkBuilder, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)
from bayesian_network.inference_machines.spa_v3.utils.extensions_to_factor_graph import to_networkx

# %%

device = torch.device("cpu")
dtype = torch.float64

torch_settings = TorchSettings(device="cpu", dtype="float64")

# %%

Q1 = Node(generate_random_probability_matrix((2), device=device, dtype=dtype), name="Q1")
Q2 = Node(generate_random_probability_matrix((2, 3), device=device, dtype=dtype), name="Q2")
Y2 = Node(generate_random_probability_matrix((3, 3), device=device, dtype=dtype), name="Y2")

network = (
    BayesianNetworkBuilder()
    .add_node(Q1)
    .add_node(Q2, parents=Q1)
    .add_node(Y2, parents=[Q2])
    .build()
)

# %%

# Q1 = Node(generate_random_probability_matrix((2), device=device, dtype=dtype), name="Q1")
# Q2 = Node(generate_random_probability_matrix((2, 3), device=device, dtype=dtype), name="Q2")
# Q3 = Node(generate_random_probability_matrix((3, 2), device=device, dtype=dtype), name="Q3")
# Y1 = Node(generate_random_probability_matrix((2, 2), device=device, dtype=dtype), name="Y1")
# Y2 = Node(generate_random_probability_matrix((3, 3), device=device, dtype=dtype), name="Y2")
# Y3 = Node(generate_random_probability_matrix((3, 4), device=device, dtype=dtype), name="Y3")
# Y4 = Node(generate_random_probability_matrix((2, 2), device=device, dtype=dtype), name="Y4")
# Y5 = Node(generate_random_probability_matrix((2, 3), device=device, dtype=dtype), name="Y5")
# network = (
#     BayesianNetworkBuilder()
#     .add_node(Q1)
#     .add_node(Q2, parents=Q1)
#     .add_node(Q3, parents=Q2)
#     .add_node(Y1, parents=Q1)
#     .add_node(Y2, parents=Q2)
#     .add_node(Y3, parents=Q2)
#     .add_node(Y4, parents=Q3)
#     .add_node(Y5, parents=Q3)
#     .build()
# )

# %%

plt.figure()
nx.draw_networkx(network._G)

# %%

inference_machine = SpaInferenceMachine(
    settings=SpaInferenceMachineSettings(
        torch_settings=torch_settings,
        average_log_likelihood=True,
    ),
    bayesian_network=network,
    num_observations=1,
    observed_nodes=[Y2],
)

print(inference_machine._num_iterations)

# %%

G = inference_machine.factor_graph >> to_networkx()

nx.draw_networkx(G)

# %%
