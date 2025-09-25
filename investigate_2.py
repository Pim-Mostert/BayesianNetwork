# %%


from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.common.torch_settings import TorchSettings


import torch

from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.inference_machines.spa_v1.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)

# %%

torch_settings = TorchSettings(
    device="cpu",
    dtype="float64",
)
device = torch.device(torch_settings.device)
dtype = torch.float64

# %%

Q1 = Node(
    generate_random_probability_matrix((2), device=device, dtype=dtype),
    name="Q1",
)
Y1 = Node(
    generate_random_probability_matrix((2, 2), device=device, dtype=dtype),
    name="Y",
)
Y2 = Node(
    generate_random_probability_matrix((2, 2), device=device, dtype=dtype),
    name="Y",
)
Ys = [Y1, Y2]

nodes = [Q1, Y1, Y2]
parents = {
    Q1: [],
    Y1: [Q1],
    Y2: [Q1],
}
network = BayesianNetwork(nodes, parents)

# %%

evidence = Evidence(
    [torch.tensor([[0.995, 0.005]]), torch.tensor([[0.8, 0.2]])],
    torch_settings,
)

# %%

inference_machine = SpaInferenceMachine(
    settings=SpaInferenceMachineSettings(
        torch_settings=torch_settings,
        num_iterations=10,
        average_log_likelihood=True,
    ),
    bayesian_network=network,
    observed_nodes=Ys,
    num_observations=1,
)

for a in range(100):
    inference_machine.enter_evidence(evidence)
    inference_machine.log_likelihood()
    print(inference_machine.factor_graph.variable_nodes[Q1].output_messages[0].value)

# %%

# inference_machine.factor_graph.variable_nodes[Q1].output_messages[0].value

# %%
