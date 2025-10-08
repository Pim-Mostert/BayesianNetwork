# %% Imports
import logging
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot


from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.common import InferenceMachineSettings
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.inference_machines.naive.naive_inference_machine import NaiveInferenceMachine
from bayesian_network.inference_machines.spa_v1_1.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler

logging.basicConfig(level=logging.INFO)

# %% Configuration
torch_settings = TorchSettings(
    device="cpu",
    dtype="float64",
)


# %% Create network


def generate_random_network() -> Tuple[BayesianNetwork, List[Node]]:
    device = torch_settings.device
    dtype = torch_settings.dtype

    Q0_cpt = generate_random_probability_matrix((2), device=device, dtype=dtype)
    Q1_cpt = generate_random_probability_matrix((2, 3), device=device, dtype=dtype)
    Q2_cpt = generate_random_probability_matrix((2, 3, 4), device=device, dtype=dtype)
    Y_cpt = generate_random_probability_matrix((2, 4, 5), device=device, dtype=dtype)
    Q0 = Node(Q0_cpt, name="Q0")
    Q1 = Node(Q1_cpt, name="Q1")
    Q2 = Node(Q2_cpt, name="Q2")
    Y = Node(Y_cpt, name="Y")

    nodes = [Q0, Q1, Q2, Y]
    parents = {
        Q0: [],
        Q1: [Q0],
        Q2: [Q0, Q1],
        Y: [Q0, Q2],
    }

    observed_nodes = [Y]
    bayesian_network = BayesianNetwork(nodes, parents)

    return bayesian_network, observed_nodes


network, observed_nodes = generate_random_network()

# Create training data
sampler = TorchBayesianNetworkSampler(
    bayesian_network=network,
    torch_settings=torch_settings,
)

N = 5
data = sampler.sample(N, observed_nodes)

evidence = Evidence(
    [one_hot(x.long(), node.num_states) for node, x in zip(observed_nodes, data.T)],
    torch_settings,
)

# %% Inference


naive_inference_machine = NaiveInferenceMachine(
    settings=InferenceMachineSettings(
        torch_settings,
        average_log_likelihood=True,
    ),
    bayesian_network=network,
    observed_nodes=observed_nodes,
)


all_messages = []
log_likelihood = []


def logger(iteration):
    nodes = [
        *[node for node in spa_inference_machine.factor_graph.variable_nodes.values()],
        *[node for node in spa_inference_machine.factor_graph.factor_nodes.values()],
    ]

    def flatten(l):
        return [x for xs in l for x in xs]

    messages = flatten(
        [
            *[node.input_messages for node in nodes],
            *[node.output_messages for node in nodes],
        ]
    )
    values = torch.concat([message.value.flatten() for message in messages])
    all_messages.append(values.flatten())

    def get_log_likelihood(factor_graph) -> float:
        # local_likelihoods: [num_observations, num_nodes]
        log_likelihoods = torch.stack(
            [node.local_log_likelihood for node in factor_graph.variable_nodes.values()], dim=1
        )
        return log_likelihoods.sum(dim=1).mean().item()

    log_likelihood.append(get_log_likelihood(spa_inference_machine.factor_graph))


spa_inference_machine = SpaInferenceMachine(
    settings=SpaInferenceMachineSettings(
        torch_settings,
        average_log_likelihood=True,
        num_iterations=10,
        callback=logger,
    ),
    bayesian_network=network,
    observed_nodes=observed_nodes,
    num_observations=N,
)

naive_inference_machine.enter_evidence(evidence)
naive_ll = naive_inference_machine.log_likelihood()

spa_inference_machine.enter_evidence(evidence)
spa_inference_machine.log_likelihood()


# %% Check messages over iterations

messages = torch.stack(all_messages)
total_ll = torch.tensor(log_likelihood)

plt.figure()
plt.plot(messages)

plt.figure()
plt.plot(total_ll)
plt.axhline(naive_ll, color="red")

# %%
