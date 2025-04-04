# %% Imports
import matplotlib.pyplot as plt
import torch
import torchvision

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import Evidence, EvidenceBatches
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine import (
    TorchSumProductAlgorithmInferenceMachine,
)
from bayesian_network.interfaces import IInferenceMachine
from bayesian_network.optimizers.em_optimizer import EmOptimizer, EmOptimizerSettings

# %% Configuration
torch_settings = TorchSettings(
    device="cpu",
    dtype="float64",
)

# %% Load data
mnist = torchvision.datasets.MNIST(
    "./experiments/mnist",
    train=True,
    # transform=transforms.ToTensor(),
    download=True,
)
data = mnist.data / 255

height, width = data.shape[1:3]
num_features = height * width
num_observations = data.shape[0]

# Morph into evidence structure
data = data.reshape([num_observations, num_features])

gamma = 0.001
evidence = Evidence(
    [torch.stack([1 - x, x]).T for x in data.T * (1 - gamma) + gamma / 2],
    torch_settings,
)


# %%

batches = EvidenceBatches(evidence, 100)

# DIT LIJTK TE WERKEN. TODO:
# - IMPLEMENT BATCH OPTIMIZER
# - IMPLEMENT MNISTEVIDENCE FOR LAZY LOADING

# %%

plt.imshow(evidence[1][:, 1].reshape(28, 28))

# %% Define network
num_classes = 10

# Create network
Q = Node(
    torch.ones(
        (num_classes),
        device=torch_settings.device,
        dtype=torch_settings.dtype,
    )
    / num_classes,
    name="Q",
)
mu = (
    torch.rand(
        (height, width, num_classes),
        device=torch_settings.device,
        dtype=torch_settings.dtype,
    )
    * 0.2
    + 0.4
)
mu = torch.stack([1 - mu, mu], dim=3)
Ys = [Node(mu[iy, ix], name=f"Y_{iy}x{ix}") for iy in range(height) for ix in range(width)]

nodes = [Q] + Ys
parents = {Y: [Q] for Y in Ys}
parents[Q] = []

network = BayesianNetwork(nodes, parents)

num_iterations = 10


# %% Fit network
def inference_machine_factory(
    bayesian_network: BayesianNetwork,
) -> IInferenceMachine:
    return TorchSumProductAlgorithmInferenceMachine(
        bayesian_network=bayesian_network,
        observed_nodes=Ys,
        torch_settings=torch_settings,
        num_iterations=3,
        num_observations=evidence.num_observations,
        callback=lambda *args: None,
    )


def callback(ll, iteration, duration):
    print(f"Finished iteration {iteration}/{num_iterations} - ll: {ll}" " - it took: {duration} s")


em_optimizer = EmOptimizer(
    network,
    inference_machine_factory,
    settings=EmOptimizerSettings(
        num_iterations=num_iterations,
        iteration_callback=callback,
    ),
)
em_optimizer.optimize(evidence)
